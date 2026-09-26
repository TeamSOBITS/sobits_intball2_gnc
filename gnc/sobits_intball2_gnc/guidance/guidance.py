#!/usr/bin/env python3
"""Guidance orchestrator node for IntBall2 (the single guidance-system node).

The only ``rclpy`` node in the guidance system (1-file-1-node rule, matching
``control/control.py``). Wires the ROS I/O wrappers (``guidance/ros``,
``common/ros``) to :class:`~sobits_intball2_gnc.guidance.utils.guidance_executor.GuidanceExecutor`
via dependency injection, and serves it as the ``execute_fn`` behind
``CtlCommandActionServer`` (``docs/guidance_node_implementation_plan.md``).

Configuration comes from ``config/gnc_params.yaml`` (loaded by
``launch/guidance.launch.py``); a bare ``ros2 run`` uses in-code defaults.
"""
import os
import threading

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.imu_subscriber import ImuSubscriber
from sobits_intball2_gnc.control.utils.singleton_lock import (
    SingletonLockError,
    acquire_singleton_lock,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.guidance_params import (
    ATTITUDE_REFERENCE_MODES,
    CAMERA_NAMES,
    declare_guidance_params,
    goal_execute_kwargs,
)
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.ros.checkpoint_publisher import CheckpointPublisher
from sobits_intball2_gnc.guidance.ros.ctl_command_action_server import (
    TERMINATE_ABORTED,
    TERMINATE_SUCCESS,
    CtlCommandActionServer,
)
from sobits_intball2_gnc.guidance.ros.multi_dof_joint_trajectory_publisher import (
    MultiDOFJointTrajectoryPublisher,
)
from sobits_intball2_gnc.guidance.ros.marker_array_publisher import MarkerArrayPublisher
from sobits_intball2_gnc.guidance.ros.marker_array_subscriber import MarkerArraySubscriber
from sobits_intball2_gnc.guidance.ros.speed_path_publisher import SpeedPathPublisher
from sobits_intball2_gnc.guidance.trajectory_tracking.tracker_builder import (
    TRAJECTORY_TRACKING_MODES,
)
from sobits_intball2_gnc.guidance.utils.guidance_executor import (
    STATUS_CANCELED,
    STATUS_PLANNING_FAILED,
    STATUS_SUCCESS,
    GuidanceExecutor,
)
from sobits_intball2_gnc.guidance.utils.velocity_estimator import VelocityEstimator

ACTION_NAME = "/gnc/move_to"
TRAJECTORY_SPEED_PATH_TOPIC = "/gnc/trajectory_path_speed"
LOCAL_TRAJECTORY_SPEED_PATH_TOPIC = "/gnc/trajectory_path_speed_local"
TF_STARTUP_TIMEOUT = 5.0
# Separate lock file from control_node's: a leftover process once survived
# kill as a child and answered /gnc/move_to alongside the new one,
# corrupting goal feedback.
GUIDANCE_LOCK_PATH = "/tmp/intball2_guidance_node.lock"

class GuidanceNode(Node):
    """Single orchestrator node: wire wrappers to logic and serve the action."""

    def __init__(self) -> None:
        # Default use_sim_time=True since this node's whole timing model
        # (clock_seconds_fn/spin_fn below) assumes self.get_clock() is sim
        # time. guidance.launch.py also injects it, but this keeps a bare
        # `ros2 run` on sim time too. A parameter_override is a default, not a lock: an
        # explicit `--ros-args -p use_sim_time:=false` still wins over it.
        super().__init__(
            "guidance_node",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        static_descriptor = ParameterDescriptor(read_only=True)
        declare_guidance_params(self, static_descriptor)
        g = lambda name: self.get_parameter("guidance." + name).value  # noqa: E731

        # Same frame names as control.py's tf_correction section (shared TF
        # tree, iss_body <- body) -- declared here too since this is a
        # separate node/parameter namespace.
        self.declare_parameter("tf_correction.reference_frame", "iss_body", static_descriptor)
        self.declare_parameter("tf_correction.target_frame", "body", static_descriptor)
        reference_frame = str(self.get_parameter("tf_correction.reference_frame").value)

        # Same values as control.py's trajectory_controller section (shared
        # with HoverController's TrajectoryController) -- declared here too,
        # read-only, so trajectory generation knows the same force/mass
        # budget the control side will actually track with
        # (docs/guidance_move_to_debug_2026-08-20.md).
        self.declare_parameter("trajectory_controller.max_force", [0.181, 0.0996, 0.122])
        self.declare_parameter("trajectory_controller.mass", 3.216, static_descriptor)
        # Only used by replanning_minco_v3's heuristic segment times (a
        # scalar 1-D model) -- the static/TOPP-RA path below uses the real
        # fan-derived wrench envelope instead (wrench_envelope below), see
        # docs/2026-08-28_constrained_trajectory_generation_research.md and
        # docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md
        # "追記（2026-08-28 その2）".
        trajectory_max_force = min(
            self.get_parameter("trajectory_controller.max_force").value
        )
        trajectory_mass = float(self.get_parameter("trajectory_controller.mass").value)
        # Physical constant for the static/TOPP-RA path's wrench-envelope
        # constraint (mass above, inertia here -- inv_dyn's M matrix).
        self.declare_parameter("trajectory_controller.inertia", 0.0136, static_descriptor)
        trajectory_inertia = float(
            self.get_parameter("trajectory_controller.inertia").value
        )
        # Same 8-fan geometry/fj_max control_node's ThrustAllocator uses
        # (shared /**: params file -- see gnc_params.yaml's thrust_allocator
        # section), declared here too since this is a separate node/
        # parameter namespace. wrench_envelope_halfspaces is static given
        # this geometry, so it's computed once here rather than per
        # move_to call (ToppraTrajectory just consumes the (F, g) pair).
        thrust_allocator = ThrustAllocator.from_node(self)
        wrench_envelope_safety_margin = float(
            self.get_parameter("guidance.wrench_envelope_safety_margin").value
        )
        wrench_envelope = wrench_envelope_halfspaces(
            thrust_allocator.A, thrust_allocator.fj_max,
            safety_margin=wrench_envelope_safety_margin,
        )
        # control_node's per-axis output clamp; the brake must stay under it.
        self.declare_parameter("hover_control.max_force", 0.1, static_descriptor)
        hover_max_force = float(self.get_parameter("hover_control.max_force").value)
        target_frame = str(self.get_parameter("tf_correction.target_frame").value)

        self._tf = TfClient(self, reference_frame, target_frame)
        if not self._tf.wait_for_frame(TF_STARTUP_TIMEOUT):
            self.get_logger().warn(
                "TF frames unavailable at startup; goals will abort until "
                "they appear"
            )

        self._setpoint_pub = MultiDOFJointTrajectoryPublisher(
            self, reference_frame=reference_frame
        )
        self._checkpoint_pub = CheckpointPublisher(
            self, reference_frame=reference_frame
        )
        self._speed_path_pub = SpeedPathPublisher(
            self, TRAJECTORY_SPEED_PATH_TOPIC, reference_frame=reference_frame,
            max_speed=float(g("target_speed")),
        )
        self._local_speed_path_pub = SpeedPathPublisher(
            self, LOCAL_TRAJECTORY_SPEED_PATH_TOPIC, reference_frame=reference_frame,
            max_speed=float(g("target_speed")), line_width=0.03,
            low_rgb=(0.0, 1.0, 0.0), high_rgb=(1.0, 1.0, 0.0),
        )

        # Guidance-side TF velocity estimate (docs/
        # guidance_velocity_estimator_design.md): driven by its own low-rate
        # timer, deliberately slower than Control's 50Hz P+D loop (frequency
        # separation, docs/guidance_realtime_replanning_design.md 3-2 節).
        # max_dt reuses tf_staleness_timeout (Category A, forwarded in
        # _on_set_parameters below) rather than a new parameter, so a
        # stall-then-burst TF gap is treated the same "stale" way here as it
        # already is by GuidanceExecutor's own staleness check.
        self._vel_estimator = VelocityEstimator(
            alpha=float(g("velocity_estimate_alpha")),
            max_dt=float(g("tf_staleness_timeout")),
        )
        self._vel_timer = self.create_timer(
            1.0 / float(g("velocity_estimate_rate")), self._on_velocity_timer
        )

        self._imu = ImuSubscriber(self)
        self._braking = False

        self._obstacle_map = self._load_obstacle_map(
            str(g("obstacle_map_file")), float(g("obstacle_grid_resolution")),
            float(g("obstacle_grid_inflation")))

        self._executor_logic = GuidanceExecutor(
            self._tf, self._setpoint_pub, self._checkpoint_pub,
            clock_seconds_fn=lambda: self.get_clock().now().nanoseconds * 1e-9,
            # Do NOT call rclpy.spin_once(self, ...) here: execute() runs
            # synchronously inside the action server's own execute_callback,
            # which is already being spun by main()'s MultiThreadedExecutor.
            # Spinning the same node from a second, ad-hoc executor while the
            # first is mid-spin corrupts callback-group bookkeeping shared
            # between the two and can silently starve other callbacks (e.g.
            # the next goal's goal_callback never fires) -- see
            # docs/guidance_move_to_debug_2026-08-20.md. self.get_clock() is
            # sim time (use_sim_time), so pace on it via Clock.sleep_for, not
            # time.sleep (wall clock) -- see CLAUDE.md. sleep_for blocks on
            # /clock updates without this thread spinning the node itself;
            # the ReentrantCallbackGroup lets the MultiThreadedExecutor's
            # other threads keep servicing TF/goal callbacks concurrently.
            spin_fn=lambda seconds: self.get_clock().sleep_for(
                Duration(seconds=seconds)
            ),
            logger=self.get_logger(),
            target_speed=float(g("target_speed")),
            attitude_speed_threshold=float(g("attitude_speed_threshold")),
            align_tolerance_deg=float(g("align_tolerance_deg")),
            align_timeout=float(g("align_timeout")),
            align_settle_time=float(g("align_settle_time")),
            align_pos_tolerance_m=float(g("align_pos_tolerance_m")),
            align_pos_settle_time=float(g("align_pos_settle_time")),
            align_pos_timeout=float(g("align_pos_timeout")),
            tf_staleness_timeout=float(g("tf_staleness_timeout")),
            rate=float(g("rate")),
            camera_forward_axis={
                "main": g("camera_forward_axis.main"),
                "stereo": g("camera_forward_axis.stereo"),
            },
            speed_path_publisher=self._speed_path_pub,
            local_speed_path_publisher=self._local_speed_path_pub,
            max_accel=trajectory_max_force / trajectory_mass,
            wrench_envelope=wrench_envelope,
            mass=trajectory_mass,
            inertia=trajectory_inertia,
            velocity_fn=self._vel_estimator.get,
            max_angular_rate=np.radians(float(g("max_angular_rate_deg"))),
            align_angular_speed_deg=float(g("align_angular_speed_deg")),
            align_angular_accel_deg=float(g("align_angular_accel_deg")),
            align_traj_publish_rate_hz=float(g("align_traj_publish_rate_hz")),
            angular_velocity_fn=lambda: self._imu.gyro,
            allocator=thrust_allocator,
            stopping_profile_kwargs={
                "eta": wrench_envelope_safety_margin,
                "f_max": float(g("stopping.f_max")),
                "t_max": float(g("stopping.t_max")),
                "x_threshold": float(g("stopping.x_threshold")),
                "theta_threshold": float(g("stopping.theta_threshold")),
                "max_axis_force": hover_max_force,
            },
            stopping_tolerance_pos=float(g("stopping.tolerance_pos")),
            stopping_tolerance_att=float(g("stopping.tolerance_att")),
            stopping_duration_goal=float(g("stopping.duration_goal")),
            stopping_wait_cancel=float(g("stopping.wait_cancel")),
            obstacle_map=self._obstacle_map,
        )
        if self._obstacle_map is not None:
            self._active_obstacle_pub = MarkerArrayPublisher(self, reference_frame=reference_frame)
            self._obstacle_map.add_listener(
                lambda _grid: self._active_obstacle_pub.publish(self._obstacle_map.boxes()))
            self._active_obstacle_pub.publish(self._obstacle_map.boxes())
            self._virtual_obstacle_sub = MarkerArraySubscriber(
                self, self._on_virtual_obstacles, expected_frame=reference_frame,
                callback_group=ReentrantCallbackGroup())

        self._action_server = CtlCommandActionServer(
            self, ACTION_NAME, self._execute_fn,
            expected_frame=reference_frame,
            callback_group=ReentrantCallbackGroup(),
            busy_fn=lambda: self._braking,
        )
        # Effective use_sim_time, logged at startup (same reasoning as
        # control.py's equivalent log): this node's default-True
        # parameter_override can still be overridden by an explicit
        # --ros-args flag, and a mismatch with control_node's use_sim_time
        # silently desyncs their clocks -- see docs/archive/achieved/
        # 2026-08-25_guidance_attitude_saturation_investigation.md.
        self.get_logger().info(
            "GuidanceNode up: serving %s, use_sim_time=%s"
            % (ACTION_NAME, self.get_parameter("use_sim_time").value)
        )

        # Category-A dynamic parameter
        # (docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md):
        # align_tolerance_deg/align_timeout/align_settle_time and their
        # position-error counterparts align_pos_tolerance_m/
        # align_pos_settle_time/align_pos_timeout, plus tf_staleness_timeout,
        # are plain threshold checks the align-wait/position-convergence/
        # TF-liveness logic re-reads every iteration, so they're safe to
        # change at runtime. Same for velocity_estimate_alpha (VelocityEstimator
        # re-reads it every update() call, docs/
        # guidance_velocity_estimator_design.md 5 節). target_speed/
        # attitude_speed_threshold are category B (only take effect via a
        # fresh trajectory generation) and are intentionally left unhandled
        # here; velocity_estimate_rate is a timer period, only read at node
        # construction, so it's read-only (declared with static_descriptor
        # above) rather than handled here.
        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _on_virtual_obstacles(self, clear, set_boxes, remove) -> None:
        self._obstacle_map.apply(clear=clear, set_boxes=set_boxes, remove=remove)
        self.get_logger().info(
            "[GuidanceNode] %d virtual obstacle box(es) in the obstacle map"
            % len(self._obstacle_map.boxes()))

    def _load_obstacle_map(self, map_file, resolution, inflation):
        if map_file and not os.path.isabs(map_file):
            map_file = os.path.join(
                get_package_share_directory("sobits_intball2_gnc"), "maps", map_file)
        try:
            obstacle_map = ObstacleMap(resolution, inflation, map_file or None)
        except Exception as exc:  # noqa: BLE001 -- guidance must still serve goals without it
            self.get_logger().error(
                "[GuidanceNode] obstacle map '%s' failed to load (%r) -- obstacle "
                "avoidance unavailable" % (map_file, exc))
            return None
        self.get_logger().info(
            "[GuidanceNode] obstacle map: %d static points from '%s', resolution %.2fm, "
            "inflation %.2fm" % (obstacle_map.static_point_count, map_file, resolution, inflation))
        return obstacle_map

    def _on_velocity_timer(self) -> None:
        """Feed the latest TF pose into ``VelocityEstimator`` (docs/
        guidance_velocity_estimator_design.md). Runs on this node's
        MultiThreadedExecutor on whatever thread services this timer's
        callback group -- concurrently with ``execute()``'s own TF reads on
        the action server's ReentrantCallbackGroup thread. Safe: tf2's
        BufferCore is internally mutex-protected against concurrent lookups,
        and VelocityEstimator.update()/get() hand off state via a single
        atomic attribute reassignment (see that module's docstring).
        """
        pose = self._tf.get_pose()
        if pose is None:
            return
        pos, _quat, stamp = pose
        self._vel_estimator.update(pos, stamp)

    def _on_set_parameters(self, params) -> SetParametersResult:
        """Route the align-loop's Category-A parameters to GuidanceExecutor.

        ``execute()`` runs inside the action server's execute_callback under
        a ``ReentrantCallbackGroup`` on ``main()``'s ``MultiThreadedExecutor``,
        so this callback can run concurrently with an in-progress align-wait
        on a different thread. Each affected attribute is a single float,
        reassigned atomically under the GIL, so no lock is needed here.
        """
        for p in params:
            if p.name == "guidance.align_tolerance_deg":
                self._executor_logic.set_gains(align_tolerance_deg=float(p.value))
            elif p.name == "guidance.align_timeout":
                self._executor_logic.set_gains(align_timeout=float(p.value))
            elif p.name == "guidance.align_settle_time":
                self._executor_logic.set_gains(align_settle_time=float(p.value))
            elif p.name == "guidance.align_pos_tolerance_m":
                self._executor_logic.set_gains(align_pos_tolerance_m=float(p.value))
            elif p.name == "guidance.align_pos_settle_time":
                self._executor_logic.set_gains(align_pos_settle_time=float(p.value))
            elif p.name == "guidance.align_pos_timeout":
                self._executor_logic.set_gains(align_pos_timeout=float(p.value))
            elif p.name == "guidance.tf_staleness_timeout":
                self._executor_logic.set_gains(tf_staleness_timeout=float(p.value))
                # Also VelocityEstimator's max_dt: this param doubles as its
                # stale-gap threshold too (docs/
                # guidance_velocity_estimator_design.md 5 節), so both
                # consumers must stay in sync.
                self._vel_estimator.set_gains(max_dt=float(p.value))
            elif p.name == "guidance.velocity_estimate_alpha":
                self._vel_estimator.set_gains(alpha=float(p.value))
            elif p.name == "guidance.attitude_reference_mode":
                if str(p.value) not in ATTITUDE_REFERENCE_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason="guidance.attitude_reference_mode must be one "
                        "of %s" % sorted(ATTITUDE_REFERENCE_MODES),
                    )
            elif p.name in ("guidance.face_travel_camera",
                             "guidance.align_at_arrival_camera"):
                if str(p.value) not in CAMERA_NAMES:
                    return SetParametersResult(
                        successful=False,
                        reason="%s must be one of %s"
                        % (p.name, sorted(CAMERA_NAMES)),
                    )
            elif p.name == "guidance.trajectory_tracking_mode":
                if str(p.value) not in TRAJECTORY_TRACKING_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason="guidance.trajectory_tracking_mode must be "
                        "one of %s" % sorted(TRAJECTORY_TRACKING_MODES),
                    )
            # Any other declared parameter is Category B/C (latched or
            # static) -- accepted but intentionally not applied at runtime.
        return SetParametersResult(successful=True)

    def _execute_fn(self, p_target, q_target, feedback_cb, is_cancel_requested):
        """``CtlCommandActionServer``'s ``execute_fn`` (module docstring).

        All mode/option params are read from ``guidance.*`` here, at goal
        receipt, so each goal latches the values in effect at that moment
        (Category B,
        docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md)
        rather than reacting to a change mid-trajectory.
        """
        mode = str(self.get_parameter("guidance.attitude_reference_mode").value)
        if mode == "look_at":
            self.get_logger().warn(
                "[GuidanceNode] attitude_reference_mode=look_at is not yet "
                "implemented; falling back to face_travel"
            )
            mode = "face_travel"

        via_waypoint_names = [
            name for name in self.get_parameter("guidance.via_waypoints").value
            if name
        ]
        via_waypoints = []
        for via_waypoint_name in via_waypoint_names:
            # Reuse self._tf's already-populated buffer instead of standing
            # up a fresh TfClient here: its /tf subscription has been
            # accumulating every published transform (not just
            # reference_frame<-target_frame) since node startup, so this is
            # a non-blocking lookup with no wait_for_frame() needed -- that
            # matters because wait_for_frame() calls rclpy.spin_once() on
            # this node, which would corrupt callback-group bookkeeping if
            # called from inside _execute_fn (already being spun by the
            # action server's own MultiThreadedExecutor, see
            # GuidanceExecutor's spin_fn comment above).
            t = self._tf.get_transform(
                target_frame=self._tf.reference_frame, source_frame=via_waypoint_name
            )
            if t is None:
                self.get_logger().error(
                    "[GuidanceNode] via_waypoints TF frame '%s' not "
                    "available, aborting goal" % via_waypoint_name
                )
                return TERMINATE_ABORTED
            tr = t.transform.translation
            via_waypoints.append([tr.x, tr.y, tr.z])

        status = self._executor_logic.execute(
            p_target, q_target, feedback_cb, is_cancel_requested,
            via_waypoints=via_waypoints, face_travel=(mode == "face_travel"),
            **goal_execute_kwargs(self),
        )
        if status == STATUS_SUCCESS:
            return TERMINATE_SUCCESS
        if status in (STATUS_CANCELED, STATUS_PLANNING_FAILED):
            self._start_braking()
        return TERMINATE_ABORTED

    def _start_braking(self) -> None:
        """Brake after the cancel result is sent, like JAXA (setPreempted, then
        cancelTarget): rclpy only sends the result once execute returns."""
        self._braking = True
        threading.Thread(target=self._run_braking, daemon=True).start()

    def _run_braking(self) -> None:
        try:
            self._executor_logic.brake()
        except Exception as exc:  # noqa: BLE001 -- must clear _braking regardless
            self.get_logger().error("[GuidanceNode] brake failed: %r" % exc)
        finally:
            self._braking = False


def main(args=None) -> None:
    # Refuse to start a second guidance_node in this container: see
    # GUIDANCE_LOCK_PATH's comment above for the incident this prevents.
    import sys

    try:
        lock_file = acquire_singleton_lock(GUIDANCE_LOCK_PATH)  # noqa: F841
    except SingletonLockError as exc:
        print("guidance: %s" % exc, file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=args)
    node = GuidanceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
