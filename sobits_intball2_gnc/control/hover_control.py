#!/usr/bin/env python3
"""IMU-based hover control for IntBall2.

Holds a stable hover using IMU feedback only (``/imu/imu``, ``ib2_msgs/msg/IMU``):
the angular rate (gyro) is damped toward zero and the linear-acceleration
disturbance (accelerometer, with an EMA bias estimate removed) is opposed. The
resulting corrective wrench is allocated to the 8 fans via
:class:`ThrustAllocator` and published through :class:`FanControlNode`.

The control law lives in :class:`HoverController`, which is ROS-agnostic and
exposes a feed-forward translation-force hook so future free-path motion can be
stacked on top of a stable hover without changing the hover logic.

Note: with IMU only there is no absolute attitude/position reference, so this is
station-keeping in the rate/disturbance sense (anti-tumble + disturbance
rejection); attitude and position slowly drift. ``compute`` keeps an optional
``attitude_ref`` argument so a future pose reference can extend it to true
station-keeping.
"""
import numpy as np
import rclpy
from rclpy.node import Node

from sobits_intball2_gnc.control.fan_control import FanControlNode
from sobits_intball2_gnc.control.gnc_params import load_gnc_config
from sobits_intball2_gnc.control.thrust_allocator import ThrustAllocator

IMU_TOPIC = "/imu/imu"
FEEDFORWARD_TOPIC = "/gnc/feedforward_force"


def _deadband(vec, threshold):
    """Zero out per-axis components whose magnitude is below ``threshold``."""
    return np.where(np.abs(vec) < threshold, 0.0, vec)


class HoverController:
    """IMU-only hover control law (ROS-agnostic, testable)."""

    def __init__(self, params: dict) -> None:
        self.kd_w = np.asarray(params["kd_w"], dtype=float)
        self.kp_a = np.asarray(params["kp_a"], dtype=float)
        self.deadband_w = float(params["deadband_w"])
        self.deadband_a = float(params["deadband_a"])
        self.acc_bias_alpha = float(params["acc_bias_alpha"])
        self.max_force = float(params["max_force"])
        self.max_torque = float(params["max_torque"])
        self._acc_bias = np.zeros(3)

    def compute(self, gyro, acc, feedforward_force=None, attitude_ref=None):
        """Return (force, torque) lists for the given IMU sample.

        ``gyro`` / ``acc`` are 3-element body-frame iterables. ``feedforward_force``
        (optional) is summed with the hover corrective force before clamping, so
        free-path motion can command translation on top of hover. ``attitude_ref``
        is reserved for a future pose reference and is currently unused.
        """
        w = np.asarray(gyro, dtype=float)
        a = np.asarray(acc, dtype=float)

        # Torque: damp angular rate toward zero (deadband to ignore noise).
        w_db = _deadband(w, self.deadband_w)
        torque = -self.kd_w * w_db

        # Force: oppose acceleration disturbance after removing EMA bias.
        self._acc_bias = (
            (1.0 - self.acc_bias_alpha) * self._acc_bias
            + self.acc_bias_alpha * a
        )
        a_res = _deadband(a - self._acc_bias, self.deadband_a)
        force = -self.kp_a * a_res

        # Feed-forward translation hook (for future free-path motion).
        if feedforward_force is not None:
            force = force + np.asarray(feedforward_force, dtype=float)

        force = np.clip(force, -self.max_force, self.max_force)
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return force.tolist(), torque.tolist()


class HoverControlNode(Node):
    """Subscribe to IMU, run the hover law, publish fan duties."""

    def __init__(self) -> None:
        super().__init__("hover_control_node")
        cfg = load_gnc_config()
        hc = cfg["hover_control"]
        self._rate = float(hc["control_rate"])
        self._controller = HoverController(hc)
        self._allocator = ThrustAllocator(cfg)
        self._fan = FanControlNode("hover_fan_pub")

        self._gyro = None
        self._acc = None
        self._feedforward = None

        # ib2_msgs is only needed at runtime; import here so the package builds
        # without it present.
        from ib2_msgs.msg import IMU
        from geometry_msgs.msg import Vector3

        self._sub_imu = self.create_subscription(
            IMU, IMU_TOPIC, self._on_imu, 10
        )
        self._sub_ff = self.create_subscription(
            Vector3, FEEDFORWARD_TOPIC, self._on_feedforward, 1
        )
        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(
            "HoverControlNode up: subscribing %s, publishing /ctl/duty "
            "at %.1f Hz" % (IMU_TOPIC, self._rate)
        )

    def _on_imu(self, msg) -> None:
        self._gyro = [msg.gyro_x, msg.gyro_y, msg.gyro_z]
        self._acc = [msg.acc_x, msg.acc_y, msg.acc_z]

    def _on_feedforward(self, msg) -> None:
        self._feedforward = [msg.x, msg.y, msg.z]

    def _on_timer(self) -> None:
        if self._gyro is None or self._acc is None:
            self._fan.set_duty_array([])  # no IMU yet -> idle
            return
        force, torque = self._controller.compute(
            self._gyro, self._acc, feedforward_force=self._feedforward
        )
        duties = self._allocator.allocate(force, torque)
        self._fan.set_duty_array(duties)

    def destroy_node(self) -> bool:
        self._fan.destroy_node()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HoverControlNode()
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
