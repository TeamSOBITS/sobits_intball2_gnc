#!/usr/bin/env python3
"""Speed-colored path publisher for IntBall2 (RViz visualization only).

ROS I/O wrapper (does not subclass Node): publishes a one-shot
``visualization_msgs/Marker`` (``LINE_STRIP``) with a per-vertex color
gradient by speed, for a human to see in RViz. ``nav_msgs/Path``
(``guidance/ros/path_publisher.py``) has no per-point color support, so this
is a separate marker rather than an extension of that class.

Color is blue (0 m/s) -> red (>= ``max_speed``), linearly interpolated and
clamped -- an absolute scale, not normalized to the min/max speed of a given
call, so the same color always means the same speed across different paths.
RViz interpolates ``marker.colors`` between adjacent vertices, so the result
reads as one smooth gradient along the line, not discrete segments.

Purely a visualization aid, with no bearing on control behavior (same role
as ``PathPublisher``, see that module's docstring).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

DEFAULT_REFERENCE_FRAME = "iss_body"
DEFAULT_MAX_SPEED = 0.5  # [m/s], matches guidance.target_speed's default
DEFAULT_LINE_WIDTH = 0.01  # [m]

# Transient-local + reliable, matching path_publisher.py's DEFAULT_QOS (see
# that module for the rationale).
DEFAULT_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


def _speed_to_color(speed: float, max_speed: float) -> ColorRGBA:
    """Blue (0 m/s) -> red (>= max_speed), clamped to [0, max_speed]."""
    frac = 0.0 if max_speed <= 0.0 else min(1.0, max(0.0, speed / max_speed))
    color = ColorRGBA()
    color.r = frac
    color.g = 0.0
    color.b = 1.0 - frac
    color.a = 1.0
    return color


class SpeedPathPublisher:
    """Publish a one-shot speed-colored ``Marker`` for RViz visualization only.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: Marker topic name (no default, since the caller names the
            topic, matching ``PathPublisher``'s convention).
        reference_frame: ``frame_id`` stamped on the marker (must match
            ``tf_correction.reference_frame`` on the Control side, i.e.
            ``iss_body``, so the path lines up with the TF tree).
        max_speed: speed [m/s] at which the color scale saturates to red
            (default: ``DEFAULT_MAX_SPEED``; pass the same value as
            ``guidance.target_speed`` so "red" means "at the planned cruise
            speed").
        line_width: marker line width [m] (default: ``DEFAULT_LINE_WIDTH``).
        qos_profile: QoS for the publisher (default: transient-local +
            reliable, see ``DEFAULT_QOS``).
    """

    def __init__(self, node: Node, topic: str,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 max_speed: float = DEFAULT_MAX_SPEED,
                 line_width: float = DEFAULT_LINE_WIDTH,
                 qos_profile: QoSProfile = DEFAULT_QOS) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._max_speed = float(max_speed)
        self._line_width = float(line_width)
        self._pub = node.create_publisher(Marker, topic, qos_profile)
        node.get_logger().info(
            "[SpeedPathPublisher] publishing to %s (frame: %s, max_speed: %.3f)"
            % (topic, reference_frame, self._max_speed)
        )

    def publish(self, samples) -> None:
        """Publish the full speed-colored path.

        ``samples`` is an iterable of ``(pos, speed)`` pairs -- ``pos`` a
        3-element iterable expressed in the reference frame, ``speed`` a
        scalar [m/s] -- in path order (e.g. the planned trajectory sampled
        at a fixed step for display, not one point per control tick).
        """
        marker = Marker()
        marker.header.frame_id = self._reference_frame
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = "speed_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self._line_width
        marker.pose.orientation.w = 1.0
        for pos, speed in samples:
            point = Point()
            point.x, point.y, point.z = pos
            marker.points.append(point)
            marker.colors.append(_speed_to_color(speed, self._max_speed))
        self._pub.publish(marker)
        self._node.get_logger().info(
            "[SpeedPathPublisher] published %d-point speed path" % len(marker.points)
        )


def main(args=None) -> None:
    """Standalone manual test: publish a short straight-line speed path once.

    Run with ``ros2 run sobits_intball2_gnc speed_path_publisher -- --topic <topic> [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="speed_path_publisher",
        description="Manual test for SpeedPathPublisher: publish a "
                    "short straight-line speed-colored path once for RViz.",
    )
    parser.add_argument("--topic", required=True,
                        help="marker topic to publish (e.g. /gnc/trajectory_path_speed)")
    parser.add_argument("--frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    parser.add_argument("--length", type=float, default=1.0, metavar="M",
                        help="straight-line length along +x [m] (default: 1.0)")
    parser.add_argument("--points", type=int, default=10, metavar="N",
                        help="number of path points (default: 10)")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        metavar="MPS",
                        help="speed [m/s] at which color saturates to red "
                             "(default: %(default)s)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("speed_path_publisher_test")
    pub = SpeedPathPublisher(node, ns.topic, ns.frame, max_speed=ns.max_speed)
    samples = [
        ([ns.length * i / (ns.points - 1), 0.0, 0.0],
         ns.max_speed * i / (ns.points - 1))
        for i in range(ns.points)
    ]
    # Give the publisher time to match with any already-open subscriber.
    end = time.monotonic() + 1.0
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    pub.publish(samples)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
