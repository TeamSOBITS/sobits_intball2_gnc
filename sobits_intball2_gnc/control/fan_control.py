#!/usr/bin/env python3
import argparse
import math
import sys
import time
from typing import Mapping

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

_KJ: float = 4.082482905  # 推力から duty への換算係数 (N^0.5 -> duty)
_FAN_COUNT: int = 8


class FanControlNode(Node):
    """Direct fan duty control for IntBall2 via /ctl/duty.

    Inherits from rclpy.node.Node. Instantiate directly to use as a
    standalone node, or share the instance with other logic running in
    the same process.
    """

    def __init__(self, node_name: str = "fan_control_node") -> None:
        super().__init__(node_name)
        self._duties: list[float] = [0.0] * _FAN_COUNT
        self._pub = self.create_publisher(Float64MultiArray, "/ctl/duty", 1)
        self.get_logger().info(
            "FanControlNode initialized, publishing to /ctl/duty"
        )

    def set_duty(self, fan_id: int, duty: float) -> None:
        """Set duty for a single fan (fan_id: 1-8) and publish."""
        if self._apply_duty(fan_id, duty):
            self.publish()

    def set_duties(self, duties: Mapping[int, float]) -> None:
        """Set duty per fan and publish once.

        ``duties`` maps fan_id (1-8) to its duty ratio [0.0-1.0].
        Only the listed fans are updated; the rest keep their values.
        """
        changed = False
        for fan_id, duty in duties.items():
            changed |= self._apply_duty(fan_id, duty)
        if changed:
            self.publish()

    def set_all_duty(self, duty: float) -> None:
        """Set all 8 fans to the same duty and publish."""
        clamped = self._clamp(duty)
        self._duties = [clamped] * _FAN_COUNT
        self.get_logger().info(f"all fans duty -> {clamped:.3f}")
        self.publish()

    def _apply_duty(self, fan_id: int, duty: float) -> bool:
        """Update one fan in the internal array. Returns True if applied."""
        if not 1 <= fan_id <= _FAN_COUNT:
            self.get_logger().warn(
                f"fan_id {fan_id} out of range [1, {_FAN_COUNT}], ignored"
            )
            return False
        clamped = self._clamp(duty, fan_id)
        self._duties[fan_id - 1] = clamped
        self.get_logger().info(f"fan{fan_id} duty -> {clamped:.3f}")
        return True

    def _clamp(self, duty: float, fan_id: int | None = None) -> float:
        """Clamp duty into [0.0, 1.0], warning if out of range.

        Reverse thrust is physically impossible: the thr plugin maps any
        negative duty to zero force (verified empirically), so duty is
        clamped to the valid [0.0, 1.0] range.
        """
        clamped = max(0.0, min(1.0, duty))
        if clamped != duty:
            target = f" for fan {fan_id}" if fan_id is not None else ""
            self.get_logger().warn(f"duty {duty} clamped to {clamped}{target}")
        return clamped

    def publish(self) -> None:
        """Publish the current duty array to /ctl/duty."""
        self.get_logger().debug(
            f'publish duties: {[f"{d:.3f}" for d in self._duties]}'
        )
        self._pub.publish(self._make_msg())

    def force_to_duty(self, f: float) -> float:
        """Convert thrust [N] to duty ratio. Negative values are treated as 0."""
        return _KJ * math.sqrt(max(0.0, f))

    def duty_to_force(self, duty: float) -> float:
        """Convert duty ratio to thrust [N]."""
        return (duty / _KJ) ** 2

    def _make_msg(self) -> Float64MultiArray:
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="fan_duty", size=_FAN_COUNT, stride=1)
        ]
        msg.layout.data_offset = 0
        msg.data = list(self._duties)
        return msg


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1:
        _build_parser().print_help()
        sys.exit(1)
    return _build_parser().parse_args()


def _fan_duty_pair(text: str) -> tuple[int, float]:
    """Parse a ``FAN:DUTY`` token into a (fan_id, duty) tuple."""
    try:
        fan_str, duty_str = text.split(":")
        return int(fan_str), float(duty_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid FAN:DUTY pair '{text}' (expected e.g. 1:0.5)"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IntBall2 direct fan control (publishes to /ctl/duty)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fan", type=int, metavar="N", help="fan number to control (1-8)"
    )
    group.add_argument(
        "--all",
        type=float,
        metavar="DUTY",
        dest="all_duty",
        help="set all fans to DUTY [0.0-1.0]",
    )
    group.add_argument(
        "--set",
        type=_fan_duty_pair,
        nargs="+",
        metavar="FAN:DUTY",
        dest="fan_duties",
        help="set per-fan duties, e.g. --set 1:0.5 3:0.2",
    )
    parser.add_argument(
        "--duty",
        type=float,
        default=0.0,
        metavar="DUTY",
        help="duty ratio [0.0-1.0] (used with --fan)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        metavar="SEC",
        help="publish duration in seconds (default: 1.0)",
    )
    return parser


def main() -> None:
    args = parse_args()
    rclpy.init()
    fan = FanControlNode()

    if args.all_duty is not None:
        fan.set_all_duty(args.all_duty)
    elif args.fan_duties is not None:
        fan.set_duties(dict(args.fan_duties))
    else:
        fan.set_duty(args.fan, args.duty)

    fan.get_logger().info(f"Publishing for {args.duration:.1f} s at 50 Hz ...")
    end = time.time() + args.duration
    try:
        while rclpy.ok() and time.time() < end:
            fan.publish()
            rclpy.spin_once(fan, timeout_sec=1.0 / 50.0)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        fan.get_logger().info("Shutting down fan_control_node")
        fan.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
