#!/usr/bin/env python3
"""Guidance orchestrator node for IntBall2 (the single guidance-system node).

The only ``rclpy`` node in the guidance system (1-file-1-node rule, matching
``control/control.py``). Wires the ROS I/O wrappers (``guidance/ros``,
``common/ros``) to :class:`~sobits_intball2_gnc.guidance.utils.guidance_executor.GuidanceExecutor`
via dependency injection, and serves it as the ``execute_fn`` behind
``CtlCommandActionServer`` (``docs/guidance_node_implementation_plan.md``).

Configuration comes from ``config/gnc_params.yaml``'s ``guidance`` section.
"""
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.singleton_lock import (
    SingletonLockError,
    acquire_singleton_lock,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.ros.checkpoint_publisher import CheckpointPublisher
from sobits_intball2_gnc.guidance.ros.ctl_command_action_server import (
    TERMINATE_ABORTED,
    TERMINATE_SUCCESS,
    CtlCommandActionServer,
)
from sobits_intball2_gnc.guidance.ros.multi_dof_joint_trajectory_publisher import (
    MultiDOFJointTrajectoryPublisher,
)
from sobits_intball2_gnc.guidance.ros.speed_path_publisher import SpeedPathPublisher
from sobits_intball2_gnc.guidance.utils.guidance_executor import (
    STATUS_CANCELED,
    STATUS_SUCCESS,
    GuidanceExecutor,
)
from sobits_intball2_gnc.guidance.utils.velocity_estimator import VelocityEstimator

ACTION_NAME = "/gnc/move_to"
TRAJECTORY_SPEED_PATH_TOPIC = "/gnc/trajectory_path_speed"
TF_STARTUP_TIMEOUT = 5.0
# Separate lock file from control_node's (docs/main_plan.md, "guidance_node
# multi-launch" incident: a leftover process survived kill as a child and
# answered /gnc/move_to alongside the new one, corrupting goal feedback).
GUIDANCE_LOCK_PATH = "/tmp/intball2_guidance_node.lock"

_GUIDANCE_PARAM_DEFAULTS = {
    "guidance.target_speed": 0.5,
    "guidance.attitude_speed_threshold": 0.02,
    "guidance.align_tolerance_deg": 3.0,
    "guidance.align_timeout": 60.0,
    "guidance.align_settle_time": 0.5,
    "guidance.align_pos_tolerance_m": 0.05,
    "guidance.align_pos_settle_time": 0.5,
    "guidance.align_pos_timeout": 10.0,
    "guidance.tf_staleness_timeout": 1.0,
    "guidance.rate": 50.0,
    "guidance.velocity_estimate_rate": 10.0,
    "guidance.velocity_estimate_alpha": 0.3,
    "guidance.camera_forward_axis.main": [1.0, 0.0, 0.0],
    "guidance.camera_forward_axis.stereo": [0.0, 1.0, 0.0],
    "guidance.attitude_reference_mode": "face_travel",
    "guidance.pre_align": True,
    "guidance.align_at_arrival": True,
    "guidance.look_at_target_frame": "",
    # Optional single interior relay point (docs/
    # 2026-08-25_guidance_waypoint_insertion_curve_verification.md): a TF
    # frame name (e.g. a maps/iss_location.yaml entry) resolved via self._tf
    # at goal receipt in _execute_fn, same Category B latching as the other
    # per-goal options here. "" (default) means no via_waypoint -- unchanged
    # prior 2-waypoint behavior.
    "guidance.via_waypoint": "",
    "guidance.face_travel_camera": "main",
    "guidance.align_at_arrival_camera": "main",
    # Real-time re-planning (docs/guidance_realtime_replanning_design.md):
    # "static" (default, unchanged prior behavior) or "replanning". Category
    # B, like attitude_reference_mode -- latched at goal receipt in
    # _execute_fn below, not applied mid-trajectory.
    "guidance.trajectory_tracking_mode": "static",
    # q_des rate limit for both modes (docs/archive/achieved/
    # 2026-08-24_trajectory_state_carryover_design.md 3-4節). First-cut
    # default, not yet tuned against real tracking performance.
    "guidance.max_angular_rate_deg": 90.0,
    # Remaining-distance threshold below which "replanning" mode
    # permanently stops re-planning for the rest of that goal (docs/
    # archive/achieved/2026-08-24_replanning_distance_fallback_decision.md).
    "guidance.distance_fallback_m": 0.3,
    # Re-plan cadence for "replanning" mode, deliberately far below `rate`
    # (docs/archive/achieved/2026-08-24_replan_rate_design.md) -- matches
    # velocity_estimate_rate since v0 (this tracker's re-plan input) only
    # refreshes that often anyway. Read-only: implemented as a tick counter
    # inside _run_trajectory's existing `rate`-paced loop, not a separate
    # timer, so it only ever takes effect at construction.
    "guidance.replan_rate_hz": 10.0,
    # SLERP+trapezoid align ramp (docs/2026-08-27_align_slerp_trapezoid_
    # next_steps.md): _align_to() feeds the checkpoint a moving intermediate
    # target along this profile instead of stepping straight to the goal
    # attitude, removing composite-axis overshoot (docs/
    # 2026-08-27_composite_axis_overshoot_summary_and_plan.md). Read-only
    # (see _STATIC_PARAMS below): only read at GuidanceExecutor construction,
    # and align_angular_accel_deg specifically does not auto-track
    # control_node gain changes -- re-derive by hand if those gains change.
    "guidance.align_angular_speed_deg": 15.0,
    "guidance.align_angular_accel_deg": 2.4,
    "guidance.align_traj_publish_rate_hz": 20.0,
    # static mode only: shrinks wrench_envelope_halfspaces (see that
    # function's docstring and docs/2026-08-28_toppra_static_path_attitude_
    # overshoot_incident.md "追記（2026-08-28 その5/6）"). Only read once at
    # wrench_envelope construction below -- static like the fan geometry it's
    # paired with.
    "guidance.wrench_envelope_safety_margin": 0.7,
}

_ATTITUDE_REFERENCE_MODES = frozenset({"fixed", "face_travel", "look_at"})
_CAMERA_NAMES = frozenset({"main", "stereo"})
_TRAJECTORY_TRACKING_MODES = frozenset({"static", "replanning"})


class GuidanceNode(Node):
    """Single orchestrator node: wire wrappers to logic and serve the action."""

    def __init__(self) -> None:
        # Default use_sim_time=True since this node's whole timing model
        # (clock_seconds_fn/spin_fn below) assumes self.get_clock() is sim
        # time, and unlike control_node it is always run standalone
        # (`ros2 run`, CLAUDE.md), never through a launch file that would
        # otherwise inject this parameter (hover_control.launch.py does for
        # control_node). A parameter_override is a default, not a lock: an
        # explicit `--ros-args -p use_sim_time:=false` still wins over it.
        super().__init__(
            "guidance_node",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        static_descriptor = ParameterDescriptor(read_only=True)
        # Timer periods are only ever read at node construction (below), so
        # changing them at runtime would silently have no effect -- read-only
        # like guidance.rate (docs/guidance_velocity_estimator_design.md 5 節).
        _STATIC_PARAMS = frozenset({
            "guidance.rate", "guidance.velocity_estimate_rate",
            # max_angular_rate_deg/distance_fallback_m: no dynamic-reconfigure
            # design has been done for these yet (docs/archive/achieved/
            # 2026-08-24_replanning_distance_fallback_decision.md /
            # 2026-08-24_trajectory_state_carryover_design.md only decided the
            # values/semantics, not a Category-A wiring) -- read-only until
            # that's explicitly designed. replan_rate_hz is read-only for the
            # same reason as velocity_estimate_rate above (only read at
            # construction, to compute _replan_every_n_ticks).
            "guidance.max_angular_rate_deg", "guidance.distance_fallback_m",
            "guidance.replan_rate_hz",
            # Only read at GuidanceExecutor construction (see the ramp's own
            # comment above); no Category-A wiring exists for these either.
            "guidance.align_angular_speed_deg", "guidance.align_angular_accel_deg",
            "guidance.align_traj_publish_rate_hz",
            "guidance.wrench_envelope_safety_margin",
        })
        for name, default in _GUIDANCE_PARAM_DEFAULTS.items():
            descriptor = static_descriptor if name in _STATIC_PARAMS else None
            self.declare_parameter(name, default, descriptor)
        g = lambda name: self.get_parameter("guidance." + name).value  # noqa: E731

        # Same frame names as control.py's tf_correction section (shared TF
        # tree, iss_body <- body) -- declared here too since this is a
        # separate node/parameter namespace.
        self.declare_parameter("tf_correction.reference_frame", "iss_body", static_descriptor)
        self.declare_parameter("tf_correction.target_frame", "body", static_descriptor)
        reference_frame = str(self.get_parameter("tf_correction.reference_frame").value)

        # Same values as control.py's trajectory_controller section (shared
        # with HoverController's TrajectoryController) -- declared here too,
        # read-only, so segment-time allocation knows the same force/mass
        # budget the control side will actually track with (see
        # HeuristicSegmentTimeAllocator's docstring and
        # docs/guidance_move_to_debug_2026-08-20.md).
        self.declare_parameter("trajectory_controller.max_force", [0.181, 0.0996, 0.122])
        self.declare_parameter("trajectory_controller.mass", 4.5, static_descriptor)
        # Only used by HeuristicSegmentTimeAllocator (the replanning path's
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
            max_accel=trajectory_max_force / trajectory_mass,
            wrench_envelope=wrench_envelope,
            mass=trajectory_mass,
            inertia=trajectory_inertia,
            velocity_fn=self._vel_estimator.get,
            max_angular_rate=np.radians(float(g("max_angular_rate_deg"))),
            distance_fallback_m=float(g("distance_fallback_m")),
            replan_rate_hz=float(g("replan_rate_hz")),
            align_angular_speed_deg=float(g("align_angular_speed_deg")),
            align_angular_accel_deg=float(g("align_angular_accel_deg")),
            align_traj_publish_rate_hz=float(g("align_traj_publish_rate_hz")),
        )

        self._action_server = CtlCommandActionServer(
            self, ACTION_NAME, self._execute_fn,
            expected_frame=reference_frame,
            callback_group=ReentrantCallbackGroup(),
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
                if str(p.value) not in _ATTITUDE_REFERENCE_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason="guidance.attitude_reference_mode must be one "
                        "of %s" % sorted(_ATTITUDE_REFERENCE_MODES),
                    )
            elif p.name in ("guidance.face_travel_camera",
                             "guidance.align_at_arrival_camera"):
                if str(p.value) not in _CAMERA_NAMES:
                    return SetParametersResult(
                        successful=False,
                        reason="%s must be one of %s"
                        % (p.name, sorted(_CAMERA_NAMES)),
                    )
            elif p.name == "guidance.trajectory_tracking_mode":
                if str(p.value) not in _TRAJECTORY_TRACKING_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason="guidance.trajectory_tracking_mode must be "
                        "one of %s" % sorted(_TRAJECTORY_TRACKING_MODES),
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

        via_waypoint_name = str(self.get_parameter("guidance.via_waypoint").value)
        via_waypoint = None
        if via_waypoint_name:
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
                    "[GuidanceNode] via_waypoint TF frame '%s' not "
                    "available, aborting goal" % via_waypoint_name
                )
                return TERMINATE_ABORTED
            tr = t.transform.translation
            via_waypoint = [tr.x, tr.y, tr.z]

        status = self._executor_logic.execute(
            p_target, q_target, feedback_cb, is_cancel_requested,
            via_waypoint=via_waypoint,
            face_travel=(mode == "face_travel"),
            face_travel_camera=str(
                self.get_parameter("guidance.face_travel_camera").value
            ),
            pre_align=bool(self.get_parameter("guidance.pre_align").value),
            align_at_arrival=bool(
                self.get_parameter("guidance.align_at_arrival").value
            ),
            look_at_target_frame=str(
                self.get_parameter("guidance.look_at_target_frame").value
            ),
            align_at_arrival_camera=str(
                self.get_parameter("guidance.align_at_arrival_camera").value
            ),
            trajectory_tracking_mode=str(
                self.get_parameter("guidance.trajectory_tracking_mode").value
            ),
        )
        if status == STATUS_SUCCESS:
            return TERMINATE_SUCCESS
        if status == STATUS_CANCELED:
            return TERMINATE_ABORTED
        return TERMINATE_ABORTED


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
