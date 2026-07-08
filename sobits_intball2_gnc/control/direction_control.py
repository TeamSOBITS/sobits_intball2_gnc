#!/usr/bin/env python3
"""Direction-vector fan control for IntBall2.

Subscribes to a ``geometry_msgs/Vector3`` topic carrying a desired travel
direction in the body frame, turns it into a pure-translation wrench, allocates
it to the 8 fans via :class:`ThrustAllocator`, and publishes the duties through
:class:`FanControlNode` at a fixed rate. A standalone CLI one-shot mode is also
provided for quick testing.

This is one of the building blocks for future free-path motion (the other being
the IMU hover controller).
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

from sobits_intball2_gnc.control.fan_control import FanControlNode
from sobits_intball2_gnc.control.gnc_params import load_gnc_config
from sobits_intball2_gnc.control.thrust_allocator import ThrustAllocator

DIRECTION_TOPIC = "/gnc/direction"


def direction_to_force(direction, force_magnitude, max_force):
    """Normalize ``direction`` and scale to a clamped force vector.

    Returns a 3-element list. A zero-length direction yields a zero force.
    """
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    mag = min(force_magnitude, max_force)
    scale = mag / norm
    return [x * scale, y * scale, z * scale]


class DirectionControlNode(Node):
    """Drive IntBall2 along a commanded body-frame direction vector."""

    def __init__(self) -> None:
        super().__init__("direction_control_node")
        cfg = load_gnc_config()
        dc = cfg["direction_control"]
        self._force_magnitude = float(dc["force_magnitude"])
        self._max_force = float(dc["max_force"])
        self._rate = float(dc["control_rate"])

        self._allocator = ThrustAllocator(cfg)
        self._fan = FanControlNode("direction_fan_pub")
        self._direction = None  # latest received direction (body frame)

        self._sub = self.create_subscription(
            Vector3, DIRECTION_TOPIC, self._on_direction, 1
        )
        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(
            "DirectionControlNode up: subscribing %s, publishing /ctl/duty "
            "at %.1f Hz" % (DIRECTION_TOPIC, self._rate)
        )

    def _on_direction(self, msg: Vector3) -> None:
        self._direction = [msg.x, msg.y, msg.z]

    def _on_timer(self) -> None:
        if self._direction is None:
            self._fan.set_duty_array([])  # publish all-zero duties (quiet)
            return
        force = direction_to_force(
            self._direction, self._force_magnitude, self._max_force
        )
        duties = self._allocator.allocate(force, [0.0, 0.0, 0.0])
        self._fan.set_duty_array(duties)

    def destroy_node(self) -> bool:
        self._fan.destroy_node()
        return super().destroy_node()


def _run_node() -> None:
    rclpy.init()
    node = DirectionControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _run_oneshot(vec, duration) -> None:
    """Publish duties for a fixed direction over ``duration`` seconds."""
    rclpy.init()
    cfg = load_gnc_config()
    dc = cfg["direction_control"]
    allocator = ThrustAllocator(cfg)
    fan = FanControlNode("direction_fan_pub")
    force = direction_to_force(
        vec, float(dc["force_magnitude"]), float(dc["max_force"])
    )
    duties = allocator.allocate(force, [0.0, 0.0, 0.0])
    rate = float(dc["control_rate"])
    fan.get_logger().info(
        "one-shot: dir=%s force=%s for %.1fs" % (vec, force, duration)
    )
    end = time.time() + duration
    try:
        while rclpy.ok() and time.time() < end:
            fan.set_duty_array(duties)
            rclpy.spin_once(fan, timeout_sec=1.0 / rate)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        fan.set_all_duty(0.0)
        fan.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntBall2 direction-vector fan control"
    )
    parser.add_argument(
        "--vec",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="one-shot body-frame direction vector (testing). "
        "If omitted, runs as a node subscribing to %s." % DIRECTION_TOPIC,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        metavar="SEC",
        help="one-shot publish duration in seconds (default: 2.0)",
    )
    args = parser.parse_args(sys.argv[1:])
    if args.vec is not None:
        _run_oneshot(args.vec, args.duration)
    else:
        _run_node()


if __name__ == "__main__":
    main()
