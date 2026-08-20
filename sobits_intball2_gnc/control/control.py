#!/usr/bin/env python3
"""Control orchestrator node for IntBall2 (the single control-system node).

This is the only ``rclpy`` node in the control system (1-file-1-node rule). It
wires the ROS I/O wrappers (``control/ros``) to the ROS-agnostic control logic
(``control/utils``) via dependency injection and drives the control loop:

    IMU (+ TF pose) --> HoverController --> ThrustAllocator --> FanDutyPublisher

Configuration comes from the ROS2 parameter system (``config/gnc_params.yaml``).
This node does not read individual algorithm parameters; each module declares
and reads its own parameters through its ``from_node`` factory. The parameters
this node reads directly are the ones it uses itself: the hover mode, the
control loop rate, the TF frame names and the checkpoint topic name.

Self-position comes from the TF tree (``iss_body`` <- ``body``), which the
simulator publishes with Navigation OFF. Nothing here touches
``/sensor_fusion/navigation`` or the JAXA ``ctl_only`` controller: with
Navigation OFF that controller stays in STAND_BY and never competes for
``/ctl/duty``.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.fan_duty_publisher import (
    DUTY_TOPIC,
    FanDutyPublisher,
)
from sobits_intball2_gnc.control.ros.imu_subscriber import ImuSubscriber, IMU_TOPIC
from sobits_intball2_gnc.control.ros.multi_dof_joint_trajectory_subscriber import (
    MultiDOFJointTrajectorySubscriber,
)
from sobits_intball2_gnc.control.ros.pose_array_subscriber import PoseArraySubscriber
from sobits_intball2_gnc.control.utils.hover_controller import (
    HOVER_MODES,
    HoverController,
)
from sobits_intball2_gnc.control.utils.singleton_lock import (
    SingletonLockError,
    acquire_singleton_lock,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

# Phase 1 (docs/main_plan.md): manually step the hold target to the next
# checkpoint. A service (not a topic) so the caller gets an explicit
# success/failure back -- False means the array was already on its last
# checkpoint (or none was ever received), so the caller can tell "no-op"
# apart from "advanced".
ADVANCE_CHECKPOINT_SERVICE = "/gnc/advance_checkpoint"

# Period of the periodic "who owns the fans" status log [s] (0 disables it).
DEFAULT_STATUS_LOG_PERIOD = 2.0
# How long to wait for the configured TF frames at startup [s].
TF_STARTUP_TIMEOUT = 5.0


class ControlNode(Node):
    """Single orchestrator node: wire wrappers to logic and run the loop."""

    def __init__(self) -> None:
        super().__init__("control_node")

        # --- ROS I/O wrappers (attach to this node; none is itself a Node) ---
        self._fan = FanDutyPublisher(self)
        self._imu = ImuSubscriber(self, IMU_TOPIC)

        # --- control logic (parameters read by each module's from_node) ------
        self._allocator = ThrustAllocator.from_node(self)
        # declare_parameters() must run before we read hover_control.mode and
        # the TF frame names below; from_node() calls it again idempotently.
        HoverController.declare_parameters(self)

        # This node owns the mode decision: it picks which ROS interfaces to
        # create, then injects the result. The logic layer only sees whether it
        # was handed a TF client.
        mode = str(self.get_parameter("hover_control.mode").value)
        if mode not in HOVER_MODES:
            raise ValueError(
                "invalid hover_control.mode %r: expected one of %s"
                % (mode, ", ".join(HOVER_MODES))
            )
        self._mode = mode

        self._tf = None
        if mode == "tf_imu":
            self._tf = TfClient(
                self,
                str(self.get_parameter("tf_correction.reference_frame").value),
                str(self.get_parameter("tf_correction.target_frame").value),
            )
            # Diagnose a bad frame name at startup rather than silently never
            # correcting. A failure is not fatal: the node degrades to pure IMU
            # hover and recovers if TF appears later.
            if not self._tf.wait_for_frame(TF_STARTUP_TIMEOUT):
                self.get_logger().warn(
                    "TF frames unavailable at startup; hovering on IMU alone "
                    "until they appear"
                )

        # Trajectory setpoint interface (Phase 3a, openspec/changes/
        # add-trajectory-following): only meaningful alongside TF, same as
        # the pose corrector.
        self._trajectory_sub = None
        if self._tf is not None:
            self._trajectory_sub = MultiDOFJointTrajectorySubscriber(
                self,
                expected_frame=str(
                    self.get_parameter("tf_correction.reference_frame").value
                ),
            )

        self._hover = HoverController.from_node(
            self, self._imu, self._fan, self._allocator, self._tf,
            self._trajectory_sub,
        )

        # Checkpoint array interface (poses in the TF reference frame).
        self._path = PoseArraySubscriber(
            self,
            str(self.get_parameter("tf_correction.checkpoint_topic").value),
            on_path=self._hover.set_checkpoints,
            expected_frame=str(
                self.get_parameter("tf_correction.reference_frame").value
            ),
        )

        # Manual checkpoint advance (Phase 1, docs/main_plan.md): no prior
        # ROS interface called ControlNode.advance_checkpoint() at all.
        self._advance_srv = self.create_service(
            Trigger, ADVANCE_CHECKPOINT_SERVICE, self._on_advance_checkpoint
        )

        # Loop rate is owned/used here; it was declared by HoverController.
        self._rate = float(self.get_parameter("hover_control.control_rate").value)
        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(
            "ControlNode up: mode=%s, subscribing %s, publishing %s at %.1f Hz"
            % (mode, IMU_TOPIC, DUTY_TOPIC, self._rate)
        )

        # Periodic "do we actually own the fans?" status log.
        self.declare_parameter("control.status_log_period",
                               DEFAULT_STATUS_LOG_PERIOD)
        period = float(self.get_parameter("control.status_log_period").value)
        if period > 0.0:
            # Listen to our own output topic so we can tell how many duty
            # messages are really on the wire vs how many we sent: an excess
            # means another node is actively driving the fans. (A merely
            # *registered* foreign publisher is not proof of competition -- the
            # JAXA controller keeps its publisher open while in STAND_BY.)
            from std_msgs.msg import Float64MultiArray

            self._duty_rx_count = 0
            self._duty_rx_zero_count = 0
            self._tx_zero_count = 0
            self._last_duty_rx = 0
            self._last_duty_rx_zero = 0
            self._last_duty_tx = 0
            self._last_tx_zero = 0
            self._sub_duty = self.create_subscription(
                Float64MultiArray, DUTY_TOPIC, self._on_duty_echo, 10
            )
            self._status_timer = self.create_timer(period, self._on_status_log)

    @staticmethod
    def _is_zero_duty(values, eps: float = 1e-6) -> bool:
        """True when every duty in ``values`` is (near) zero."""
        return all(abs(v) < eps for v in values)

    def _on_duty_echo(self, msg) -> None:
        """Count duty messages seen on the wire (including our own).

        Zero-duty and non-zero-duty messages are counted separately: the
        STAND_BY heartbeat from the bridged JAXA controller publishes an
        all-zero duty array at ~1 Hz even with no competition, so counting it
        as "foreign" alongside genuinely competing (non-zero) duty commands
        produced false CONTESTED warnings (see docs/phase0_findings.md).
        """
        self._duty_rx_count += 1
        if self._is_zero_duty(msg.data):
            self._duty_rx_zero_count += 1

    def advance_checkpoint(self) -> bool:
        """Step the hover hold target to the next checkpoint (free-path hook)."""
        return self._hover.advance_checkpoint()

    def _on_advance_checkpoint(self, request, response):
        """Trigger callback for ADVANCE_CHECKPOINT_SERVICE (Phase 1)."""
        advanced = self.advance_checkpoint()
        response.success = advanced
        response.message = (
            "advanced to next checkpoint" if advanced
            else "no checkpoints received yet, or already on the last one"
        )
        return response

    def _on_status_log(self) -> None:
        """Periodically report whether this node actually drives the fans."""
        # Messages on the wire this period vs messages we sent this period.
        # foreign > 0 means someone else is actively publishing duties.
        rx = self._duty_rx_count
        rx_zero = self._duty_rx_zero_count
        tx = self._fan.publish_count
        tx_zero = self._tx_zero_count
        rx_delta = rx - self._last_duty_rx
        rx_zero_delta = rx_zero - self._last_duty_rx_zero
        tx_delta = tx - self._last_duty_tx
        tx_zero_delta = tx_zero - self._last_tx_zero
        self._last_duty_rx, self._last_duty_rx_zero = rx, rx_zero
        self._last_duty_tx, self._last_tx_zero = tx, tx_zero

        # Split the excess into zero-duty (benign STAND_BY heartbeat) and
        # non-zero (genuinely competing) so a harmless idle heartbeat from the
        # bridged JAXA controller doesn't get reported as CONTESTED.
        foreign_zero = max(0, rx_zero_delta - tx_zero_delta)
        foreign_nonzero = max(
            0, (rx_delta - rx_zero_delta) - (tx_delta - tx_zero_delta)
        )
        foreign = foreign_zero + foreign_nonzero

        # A registered foreign publisher is normal (the JAXA controller keeps
        # its publisher open in STAND_BY); report it as context only.
        other_pubs = max(0, self.count_publishers(DUTY_TOPIC) - 1)
        # Phase 0 diagnosis: force/torque split by source, pre-combination, to
        # see whether the TF correction and the IMU law cancel each other out
        # (see docs/main_plan.md Phase 0).
        f_imu = ", ".join("%.4f" % v for v in self._hover.last_force_imu)
        f_corr = ", ".join("%.4f" % v for v in self._hover.last_force_corr)
        t_imu = ", ".join("%.4f" % v for v in self._hover.last_torque_imu)
        t_corr = ", ".join("%.4f" % v for v in self._hover.last_torque_corr)
        summary = (
            "fan-control: ours=%d msgs, foreign=%d (other publishers=%d), "
            "mode=%s, imu=%s, tf=%s, duty=[%s], "
            "force_imu=[%s], force_corr=[%s], torque_imu=[%s], torque_corr=[%s]"
            % (tx_delta, foreign, other_pubs, self._mode,
               "ok" if self._imu.ready else "WAITING",
               self._hover.tf_status,
               ", ".join("%.2f" % d for d in self._fan.duties),
               f_imu, f_corr, t_imu, t_corr)
        )
        if foreign_nonzero > 0:
            self.get_logger().warn(
                summary + "  <-- FOREIGN duty messages: fan control is CONTESTED"
            )
        elif foreign_zero > 0:
            self.get_logger().info(
                summary + "  <-- benign zero-duty STAND_BY heartbeat only"
            )
        elif tx_delta == 0:
            self.get_logger().warn(
                summary + "  <-- we published nothing this period"
            )
        else:
            self.get_logger().info(summary + "  <-- fan control is OURS")

    def _on_timer(self) -> None:
        # self.get_clock().now() (not time.monotonic()) so this loop's own
        # notion of elapsed time is on the same clock as the TF stamps it
        # compares itself against: with use_sim_time=true and /clock bridged
        # from the simulator, both are sim time, so a Gazebo real-time-factor
        # drop under CPU load no longer desyncs "how much time we think
        # passed" from "how far the vehicle actually got to move" -- see
        # docs/recording_cpu_load_control_degradation.md.
        self._hover.step(self.get_clock().now().nanoseconds * 1e-9)
        if hasattr(self, "_tx_zero_count") and self._is_zero_duty(self._fan.duties):
            self._tx_zero_count += 1


def main(args=None) -> None:
    """Run the control orchestrator node.

    Configuration is supplied through the ROS2 parameter system (not positional
    arguments); ``-h`` documents how to pass the parameter file.
    """
    import argparse
    import sys
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="control",
        description=(
            "IntBall2 control orchestrator (the single control-system node): "
            "TF-corrected IMU hover. Subscribes /imu/imu (and, in tf_imu mode, "
            "the TF tree and /gnc/checkpoints) and publishes 8 fan duties to "
            "/ctl/duty. Runs with Navigation OFF."
        ),
        epilog=(
            "Parameters are provided via the ROS2 parameter system, not CLI "
            "arguments. Examples:\n"
            "  # tuned gains from the installed parameter file:\n"
            "  ros2 run sobits_intball2_gnc control --ros-args --params-file "
            "$(ros2 pkg prefix sobits_intball2_gnc)"
            "/share/sobits_intball2_gnc/config/gnc_params.yaml\n"
            "  # or, more simply:\n"
            "  ros2 launch sobits_intball2_gnc hover_control.launch.py\n"
            "  # IMU-only hover (no TF lookups):\n"
            "  ros2 run sobits_intball2_gnc control --ros-args "
            "-p hover_control.mode:=imu\n"
            "Inspect live values with: ros2 param list /control_node"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No functional flags (configuration is via ROS2 parameters); parsing the
    # non-ROS args still provides `-h` and rejects unknown arguments.
    parser.parse_args(remove_ros_args(args=argv)[1:])

    # Refuse to start a second control_node in this container: a leftover
    # process silently fighting the new one over /ctl/duty caused a real
    # incident (docs/main_plan.md, docs/trajectory_force_duration_investigation.md
    # 6-1). Held for the whole process lifetime; released automatically on exit.
    try:
        lock_file = acquire_singleton_lock()  # noqa: F841
    except SingletonLockError as exc:
        print("control: %s" % exc, file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=argv)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
