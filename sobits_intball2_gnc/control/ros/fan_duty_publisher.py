#!/usr/bin/env python3
"""Fan duty publisher for IntBall2 (`/ctl/duty`).

ROS I/O wrapper (does not subclass Node): attaches a ``/ctl/duty`` publisher to
the node passed in and turns fan duties into a ``std_msgs/Float64MultiArray``.
The number of fans and the thrust->duty coefficient ``kj`` are read from the
node parameters (declared here), so the publisher stays in sync with the thrust
allocator. Reverse thrust is physically impossible (the thr plugin maps any
negative duty to zero force), so duties are clamped to ``[0.0, 1.0]``.
"""
import argparse
import math
import sys
import time
from typing import Mapping

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

DUTY_TOPIC = "/ctl/duty"
DEFAULT_KJ = 4.082482905
DEFAULT_FAN_COUNT = 8


class FanDutyPublisher:
    """Publish 8 fan duties to ``/ctl/duty`` on the supplied node.

    Args:
        node: The rclpy Node that owns this publisher.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self.declare_parameters(node)
        self._kj = float(node.get_parameter("thrust_allocator.kj").value)
        self._fan_count = int(node.get_parameter("fan_duty_publisher.fan_count").value)
        self._duties = [0.0] * self._fan_count
        self._publish_count = 0
        self._pub = node.create_publisher(Float64MultiArray, DUTY_TOPIC, 1)
        node.get_logger().info(
            "[FanDutyPublisher] initialized (kj=%.6f, fans=%d), publishing to %s"
            % (self._kj, self._fan_count, DUTY_TOPIC)
        )

    @staticmethod
    def declare_parameters(node: Node) -> None:
        """Declare the parameters this wrapper reads (idempotent)."""
        for name, default in (
            ("thrust_allocator.kj", DEFAULT_KJ),
            ("fan_duty_publisher.fan_count", DEFAULT_FAN_COUNT),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

    @property
    def fan_count(self) -> int:
        return self._fan_count

    @property
    def duties(self) -> list:
        """The duty array most recently published (copy)."""
        return list(self._duties)

    @property
    def publish_count(self) -> int:
        """Number of duty messages this publisher has sent (for diagnostics)."""
        return self._publish_count

    def set_duty_array(self, duties: list) -> None:
        """Set the full duty array (clamped) and publish.

        Extra entries are ignored; missing entries stay 0.
        """
        self._duties = [0.0] * self._fan_count
        for i, duty in enumerate(duties[: self._fan_count]):
            self._duties[i] = self._clamp(duty, i + 1)
        self.publish()

    def set_all_duty(self, duty: float) -> None:
        """Set all fans to the same duty and publish."""
        clamped = self._clamp(duty)
        self._duties = [clamped] * self._fan_count
        self._node.get_logger().info(f"[FanDutyPublisher] all fans duty -> {clamped:.3f}")
        self.publish()

    def set_duties(self, duties: Mapping[int, float]) -> None:
        """Set duty per fan (fan_id 1-based) and publish once."""
        changed = False
        for fan_id, duty in duties.items():
            if 1 <= fan_id <= self._fan_count:
                self._duties[fan_id - 1] = self._clamp(duty, fan_id)
                changed = True
            else:
                self._node.get_logger().warn(
                    f"[FanDutyPublisher] fan_id {fan_id} out of range "
                    f"[1, {self._fan_count}], ignored"
                )
        if changed:
            self.publish()

    def publish(self) -> None:
        """Publish the current duty array to ``/ctl/duty``."""
        self._node.get_logger().debug(
            f'[FanDutyPublisher] publish: {[f"{d:.3f}" for d in self._duties]}'
        )
        self._pub.publish(self._make_msg())
        self._publish_count += 1

    def force_to_duty(self, f: float) -> float:
        """Convert thrust [N] to duty ratio. Negative values are treated as 0."""
        return self._kj * math.sqrt(max(0.0, f))

    def _clamp(self, duty: float, fan_id=None) -> float:
        clamped = max(0.0, min(1.0, duty))
        if clamped != duty:
            target = f" for fan {fan_id}" if fan_id is not None else ""
            self._node.get_logger().warn(
                f"[FanDutyPublisher] duty {duty} clamped to {clamped}{target}"
            )
        return clamped

    def _make_msg(self) -> Float64MultiArray:
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="fan_duty", size=self._fan_count, stride=1)
        ]
        msg.layout.data_offset = 0
        msg.data = list(self._duties)
        return msg


def _fan_duty_pair(text: str):
    """Parse a ``FAN:DUTY`` token into a (fan_id, duty) tuple."""
    try:
        fan_str, duty_str = text.split(":")
        return int(fan_str), float(duty_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid FAN:DUTY pair '{text}' (expected e.g. 1:0.5)"
        )


def main(args=None) -> None:
    """Standalone manual test: publish fan duties directly to ``/ctl/duty``."""
    parser = argparse.ArgumentParser(
        description="IntBall2 direct fan control (publishes to /ctl/duty)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fan", type=int, metavar="N",
                       help="fan number to control (1-8); use with --duty")
    group.add_argument("--all", type=float, metavar="DUTY", dest="all_duty",
                       help="set all fans to DUTY [0.0-1.0]")
    group.add_argument("--set", type=_fan_duty_pair, nargs="+", metavar="FAN:DUTY",
                       dest="fan_duties", help="set per-fan duties, e.g. --set 1:0.5 3:0.2")
    parser.add_argument("--duty", type=float, default=0.0, metavar="DUTY",
                        help="duty ratio [0.0-1.0] (used with --fan)")
    parser.add_argument("--duration", type=float, default=1.0, metavar="SEC",
                        help="publish duration in seconds (default: 1.0)")
    from rclpy.utilities import remove_ros_args
    argv = sys.argv if args is None else args
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("fan_duty_publisher_test")
    pub = FanDutyPublisher(node)
    if ns.all_duty is not None:
        pub.set_all_duty(ns.all_duty)
    elif ns.fan_duties is not None:
        pub.set_duties(dict(ns.fan_duties))
    else:
        pub.set_duties({ns.fan: ns.duty})

    node.get_logger().info(f"[FanDutyPublisher] publishing for {ns.duration:.1f} s at 50 Hz ...")
    end = time.time() + ns.duration
    try:
        while rclpy.ok() and time.time() < end:
            pub.publish()
            rclpy.spin_once(node, timeout_sec=1.0 / 50.0)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
