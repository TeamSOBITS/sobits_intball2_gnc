#!/usr/bin/env python3
"""Trajectory visualization path publisher for IntBall2 (`/gnc/trajectory_path`).

ROS I/O wrapper (does not subclass Node): publishes the full planned path as a
one-shot ``nav_msgs/Path``, purely for a human to see in RViz.

Control's real, control-facing interface for a Guidance trajectory is
``/gnc/trajectory_setpoint`` (``trajectory_msgs/MultiDOFJointTrajectory``,
see ``openspec/specs/trajectory-following/spec.md`` and
``docs/phase3.md``), continuously republished at the control rate with the
setpoint sampled at the current time. That type has no built-in RViz
display, so this publishes the same planned path as a standard
``nav_msgs/Path`` on a *separate* topic that Control never subscribes to
(docs/phase3.md section 8) -- purely a visualization aid, with no bearing on
control behavior.

Published once per new trajectory (not at the control rate) with a
transient-local QoS, so a viewer (e.g. RViz) still receives the latest path
even if it starts subscribing after the publish call.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

PATH_TOPIC = "/gnc/trajectory_path"
DEFAULT_REFERENCE_FRAME = "iss_body"

# Transient-local (latched-like): a late subscriber (RViz opened after the
# path was published) still gets the last message instead of nothing.
# Best-effort: matches RViz's default display QoS for most topics, which
# otherwise fails to connect to a RELIABLE-only publisher depending on the
# display's own QoS override.
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
)


class TrajectoryPathPublisher:
    """Publish a one-shot ``nav_msgs/Path`` for RViz visualization only.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: Path topic name.
        reference_frame: ``frame_id`` stamped on the path and each pose
            (must match ``tf_correction.reference_frame`` on the Control
            side, i.e. ``iss_body``, so the path lines up with the TF tree).
    """

    def __init__(self, node: Node, topic: str = PATH_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._pub = node.create_publisher(Path, topic, _LATCHED_QOS)
        node.get_logger().info(
            "[TrajectoryPathPublisher] publishing to %s (frame: %s)"
            % (topic, reference_frame)
        )

    def publish(self, samples) -> None:
        """Publish the full planned path.

        ``samples`` is an iterable of ``(pos, quat)`` pairs -- ``pos`` a
        3-element iterable, ``quat`` a 4-element ``[x, y, z, w]`` iterable --
        expressed in the reference frame, in path order (e.g. the planned
        trajectory sampled at a fixed time step for display, not one point
        per control tick).
        """
        msg = Path()
        msg.header.frame_id = self._reference_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        for pos, quat in samples:
            pose = PoseStamped()
            pose.header.frame_id = self._reference_frame
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
            (pose.pose.orientation.x, pose.pose.orientation.y,
             pose.pose.orientation.z, pose.pose.orientation.w) = quat
            msg.poses.append(pose)
        self._pub.publish(msg)
        self._node.get_logger().info(
            "[TrajectoryPathPublisher] published %d-point path" % len(msg.poses)
        )


def main(args=None) -> None:
    """Standalone manual test: publish a short straight-line path once.

    Run with ``ros2 run sobits_intball2_gnc trajectory_path_publisher [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="trajectory_path_publisher",
        description="Manual test for TrajectoryPathPublisher: publish a "
                    "short straight-line path once for RViz.",
    )
    parser.add_argument("--topic", default=PATH_TOPIC,
                        help="path topic to publish (default: %(default)s)")
    parser.add_argument("--frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    parser.add_argument("--length", type=float, default=1.0, metavar="M",
                        help="straight-line length along +x [m] (default: 1.0)")
    parser.add_argument("--points", type=int, default=10, metavar="N",
                        help="number of path points (default: 10)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("trajectory_path_publisher_test")
    pub = TrajectoryPathPublisher(node, ns.topic, ns.frame)
    samples = [
        ([ns.length * i / (ns.points - 1), 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
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
