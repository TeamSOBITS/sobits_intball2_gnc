#!/usr/bin/env python3
"""Path publisher for IntBall2 (generic ``nav_msgs/Path`` visualization).

ROS I/O wrapper (does not subclass Node): publishes a one-shot
``nav_msgs/Path`` for a human to see in RViz. Generic over topic and role: the
same class is used both for the fine trajectory preview (formerly
``/gnc/trajectory_path``, published by the now-removed
``TrajectoryPathPublisher``) and for the large-scale global path
(``/gnc/global_path``, ``docs/future_design_notes.md`` 3-1). The role lives in
the topic name passed at construction, not in the class name -- naming the
class after its message type (``PathPublisher`` for ``nav_msgs/Path``) avoids
baking a role like "trajectory" into a name that also means "path", which was
a source of confusion (see ``docs/future_design_notes.md`` 6-1).

Control's real, control-facing interface for a Guidance trajectory is
``/gnc/trajectory_setpoint`` (``trajectory_msgs/MultiDOFJointTrajectory``, see
``openspec/specs/trajectory-following/spec.md`` and
``docs/archive/achieved/phase3a_interface_contract.md``), continuously
republished at the control rate with the setpoint sampled at the current time.
That type has no built-in RViz display, so this class publishes the same
planned path as a standard ``nav_msgs/Path`` on a *separate* topic that
Control never subscribes to -- purely a visualization aid, with no bearing on
control behavior.

Published once per new path (not at the control rate) with a transient-local
QoS, so a viewer (e.g. RViz) still receives the latest path even if it starts
subscribing after the publish call.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

DEFAULT_REFERENCE_FRAME = "iss_body"

# Transient-local (latched-like): a late subscriber (RViz opened after the
# path was published) still gets the last message instead of nothing.
# Reliable: this package's publishers default to reliable (subscribers
# default to best-effort/sensor_data instead, see
# docs/future_design_notes.md 6-4) -- confirmed live (2026-08-19) after a
# best-effort /ctl/duty publisher silently dropped every command against a
# RELIABLE-only subscriber.
DEFAULT_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


class PathPublisher:
    """Publish a one-shot ``nav_msgs/Path`` for RViz visualization only.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: Path topic name (e.g. ``/gnc/global_path`` or a trajectory
            preview topic -- no default, since this class is reused across
            roles; the caller names the topic).
        reference_frame: ``frame_id`` stamped on the path and each pose
            (must match ``tf_correction.reference_frame`` on the Control
            side, i.e. ``iss_body``, so the path lines up with the TF tree).
        qos_profile: QoS for the publisher (default: transient-local +
            best-effort, see ``DEFAULT_QOS``).
    """

    def __init__(self, node: Node, topic: str,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 qos_profile: QoSProfile = DEFAULT_QOS) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._pub = node.create_publisher(Path, topic, qos_profile)
        node.get_logger().info(
            "[PathPublisher] publishing to %s (frame: %s)"
            % (topic, reference_frame)
        )

    def publish(self, samples) -> None:
        """Publish the full path.

        ``samples`` is an iterable of ``(pos, quat)`` pairs -- ``pos`` a
        3-element iterable, ``quat`` a 4-element ``[x, y, z, w]`` iterable --
        expressed in the reference frame, in path order (e.g. the planned
        trajectory or global path sampled at a fixed step for display, not
        one point per control tick).
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
            "[PathPublisher] published %d-point path" % len(msg.poses)
        )


def main(args=None) -> None:
    """Standalone manual test: publish a short straight-line path once.

    Run with ``ros2 run sobits_intball2_gnc path_publisher -- --topic <topic> [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="path_publisher",
        description="Manual test for PathPublisher: publish a "
                    "short straight-line path once for RViz.",
    )
    parser.add_argument("--topic", required=True,
                        help="path topic to publish (e.g. /gnc/global_path)")
    parser.add_argument("--frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    parser.add_argument("--length", type=float, default=1.0, metavar="M",
                        help="straight-line length along +x [m] (default: 1.0)")
    parser.add_argument("--points", type=int, default=10, metavar="N",
                        help="number of path points (default: 10)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("path_publisher_test")
    pub = PathPublisher(node, ns.topic, ns.frame)
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
