#!/usr/bin/env python3
"""IMU subscriber for IntBall2 (`/imu/imu`, ``ib2_msgs/IMU``).

ROS I/O wrapper (does not subclass Node): keeps the latest body-frame gyro and
accelerometer readings so the hover controller can read them each control tick.
``ib2_msgs`` is imported lazily so the package builds without it present.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

IMU_TOPIC = "/imu/imu"

# This package's default QoS for streaming sensor topics (best-effort), see
# docs/future_design_notes.md 6-2.
DEFAULT_QOS = qos_profile_sensor_data


class ImuSubscriber:
    """Buffer the latest IMU sample from ``/imu/imu``.

    Args:
        node: The rclpy Node that owns this subscriber.
        topic: IMU topic name.
        qos_profile: QoS for the subscription (default: best-effort, see
            ``DEFAULT_QOS``).
    """

    def __init__(self, node: Node, topic: str = IMU_TOPIC,
                 qos_profile: QoSProfile = DEFAULT_QOS) -> None:
        self._node = node
        self._gyro = None   # [gx, gy, gz]
        self._acc = None    # [ax, ay, az]
        from ib2_msgs.msg import IMU
        self._sub = node.create_subscription(IMU, topic, self._callback, qos_profile)
        node.get_logger().info(f"[ImuSubscriber] subscribing {topic}")

    def _callback(self, msg) -> None:
        self._gyro = [msg.gyro_x, msg.gyro_y, msg.gyro_z]
        self._acc = [msg.acc_x, msg.acc_y, msg.acc_z]

    @property
    def gyro(self):
        """Latest body-frame angular rate [rad/s], or None."""
        return self._gyro

    @property
    def acc(self):
        """Latest body-frame linear acceleration [m/s^2], or None."""
        return self._acc

    @property
    def ready(self) -> bool:
        """True once at least one IMU sample has arrived."""
        return self._gyro is not None and self._acc is not None


def main(args=None) -> None:
    """Standalone manual test: subscribe and print the latest IMU sample.

    Run with ``ros2 run sobits_intball2_gnc imu_subscriber [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="imu_subscriber",
        description="Manual test for ImuSubscriber: print /imu/imu gyro & acc.",
    )
    parser.add_argument("--topic", default=IMU_TOPIC,
                        help="IMU topic to subscribe (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=0.0, metavar="SEC",
                        help="seconds to run; 0 runs until Ctrl-C (default: 0)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("imu_subscriber_test")
    sub = ImuSubscriber(node, ns.topic)
    end = None if ns.duration <= 0 else time.monotonic() + ns.duration
    try:
        while rclpy.ok() and (end is None or time.monotonic() < end):
            rclpy.spin_once(node, timeout_sec=0.5)
            if sub.ready:
                node.get_logger().info(f"[ImuSubscriber] gyro={sub.gyro} acc={sub.acc}")
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
