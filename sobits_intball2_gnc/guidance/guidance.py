#!/usr/bin/env python3
"""Guidance orchestrator node for IntBall2 (the single guidance-system node).

The only ``rclpy`` node in the guidance system (1-file-1-node rule, matching
``control/control.py``). Wires the ROS I/O wrappers (``guidance/ros``,
``common/ros``) to :class:`~sobits_intball2_gnc.guidance.utils.guidance_executor.GuidanceExecutor`
via dependency injection, and serves it as the ``execute_fn`` behind
``CtlCommandActionServer`` (``docs/guidance_node_implementation_plan.md``).

Configuration comes from ``config/gnc_params.yaml``'s ``guidance`` section.
"""
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sobits_intball2_gnc.common.ros.tf_client import TfClient
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

ACTION_NAME = "/gnc/move_to"
TRAJECTORY_SPEED_PATH_TOPIC = "/gnc/trajectory_path_speed"
TF_STARTUP_TIMEOUT = 5.0

_GUIDANCE_PARAM_DEFAULTS = {
    "guidance.target_speed": 0.5,
    "guidance.attitude_speed_threshold": 0.02,
    "guidance.align_tolerance_deg": 3.0,
    "guidance.align_timeout": 60.0,
    "guidance.align_settle_time": 0.5,
    "guidance.rate": 50.0,
    "guidance.camera_forward_axis.main": [1.0, 0.0, 0.0],
    "guidance.camera_forward_axis.stereo": [0.0, 1.0, 0.0],
    "guidance.attitude_reference_mode": "face_travel",
    "guidance.pre_align": True,
    "guidance.align_at_arrival": True,
    "guidance.look_at_target_frame": "",
    "guidance.face_travel_camera": "main",
    "guidance.align_at_arrival_camera": "main",
}

_ATTITUDE_REFERENCE_MODES = frozenset({"fixed", "face_travel", "look_at"})
_CAMERA_NAMES = frozenset({"main", "stereo"})


class GuidanceNode(Node):
    """Single orchestrator node: wire wrappers to logic and serve the action."""

    def __init__(self) -> None:
        super().__init__("guidance_node")

        static_descriptor = ParameterDescriptor(read_only=True)
        for name, default in _GUIDANCE_PARAM_DEFAULTS.items():
            descriptor = static_descriptor if name == "guidance.rate" else None
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
        self.declare_parameter("trajectory_controller.max_force", 0.1)
        self.declare_parameter("trajectory_controller.mass", 4.5, static_descriptor)
        trajectory_max_force = float(
            self.get_parameter("trajectory_controller.max_force").value
        )
        trajectory_mass = float(self.get_parameter("trajectory_controller.mass").value)
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
            rate=float(g("rate")),
            camera_forward_axis={
                "main": g("camera_forward_axis.main"),
                "stereo": g("camera_forward_axis.stereo"),
            },
            speed_path_publisher=self._speed_path_pub,
            max_accel=trajectory_max_force / trajectory_mass,
        )

        self._action_server = CtlCommandActionServer(
            self, ACTION_NAME, self._execute_fn,
            expected_frame=reference_frame,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("GuidanceNode up: serving %s" % ACTION_NAME)

        # Category-A dynamic parameter
        # (docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md):
        # align_tolerance_deg/align_timeout/align_settle_time are plain
        # threshold checks the align-wait loop re-reads every iteration, so
        # they're safe to change
        # at runtime. target_speed/attitude_speed_threshold are category B
        # (only take effect via a fresh trajectory generation) and are
        # intentionally left unhandled here.
        self.add_on_set_parameters_callback(self._on_set_parameters)

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
        status = self._executor_logic.execute(
            p_target, q_target, feedback_cb, is_cancel_requested,
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
        )
        if status == STATUS_SUCCESS:
            return TERMINATE_SUCCESS
        if status == STATUS_CANCELED:
            return TERMINATE_ABORTED
        return TERMINATE_ABORTED


def main(args=None) -> None:
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
