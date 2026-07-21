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
import time

import rclpy
from rclpy.node import Node

from sobits_intball2_gnc.control.ros.fan_duty_publisher import (
    DUTY_TOPIC,
    FanDutyPublisher,
)
from sobits_intball2_gnc.control.ros.imu_subscriber import ImuSubscriber, IMU_TOPIC
from sobits_intball2_gnc.control.ros.path_subscriber import PathSubscriber
from sobits_intball2_gnc.control.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.hover_controller import (
    HOVER_MODES,
    HoverController,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

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

        self._hover = HoverController.from_node(
            self, self._imu, self._fan, self._allocator, self._tf
        )

        # Checkpoint array interface (poses in the TF reference frame).
        self._path = PathSubscriber(
            self,
            str(self.get_parameter("tf_correction.checkpoint_topic").value),
            on_path=self._hover.set_checkpoints,
            expected_frame=str(
                self.get_parameter("tf_correction.reference_frame").value
            ),
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
            self._last_duty_rx = 0
            self._last_duty_tx = 0
            self._sub_duty = self.create_subscription(
                Float64MultiArray, DUTY_TOPIC, self._on_duty_echo, 10
            )
            self._status_timer = self.create_timer(period, self._on_status_log)

    def _on_duty_echo(self, msg) -> None:
        """Count duty messages seen on the wire (including our own)."""
        self._duty_rx_count += 1

    def advance_checkpoint(self) -> bool:
        """Step the hover hold target to the next checkpoint (free-path hook)."""
        return self._hover.advance_checkpoint()

    def _on_status_log(self) -> None:
        """Periodically report whether this node actually drives the fans."""
        # Messages on the wire this period vs messages we sent this period.
        # foreign > 0 means someone else is actively publishing duties.
        rx = self._duty_rx_count
        tx = self._fan.publish_count
        rx_delta = rx - self._last_duty_rx
        tx_delta = tx - self._last_duty_tx
        self._last_duty_rx, self._last_duty_tx = rx, tx
        foreign = max(0, rx_delta - tx_delta)

        # A registered foreign publisher is normal (the JAXA controller keeps
        # its publisher open in STAND_BY); report it as context only.
        other_pubs = max(0, self.count_publishers(DUTY_TOPIC) - 1)
        summary = (
            "fan-control: ours=%d msgs, foreign=%d (other publishers=%d), "
            "mode=%s, imu=%s, tf=%s, duty=[%s]"
            % (tx_delta, foreign, other_pubs, self._mode,
               "ok" if self._imu.ready else "WAITING",
               self._hover.tf_status,
               ", ".join("%.2f" % d for d in self._fan.duties))
        )
        if foreign > 0:
            self.get_logger().warn(
                summary + "  <-- FOREIGN duty messages: fan control is CONTESTED"
            )
        elif tx_delta == 0:
            self.get_logger().warn(
                summary + "  <-- we published nothing this period"
            )
        else:
            self.get_logger().info(summary + "  <-- fan control is OURS")

    def _on_timer(self) -> None:
        # time.monotonic() is deliberate: TF stamps are compared only against
        # other TF stamps, so this clock never has to match the TF publisher's.
        self._hover.step(time.monotonic())


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
