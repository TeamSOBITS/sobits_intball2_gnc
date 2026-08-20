#!/usr/bin/env python3
"""Checkpoint publisher for IntBall2 (`/gnc/checkpoints`, producer side).

ROS I/O wrapper (does not subclass Node): publishes a single-pose
``geometry_msgs/PoseArray`` for Control's ``PoseArraySubscriber``
(``control/ros/pose_array_subscriber.py``) to consume as a static hold
target -- the same mechanism ``test/manual/send_curve_via_naventry_to_*``
scripts have been using ad hoc (inline ``PoseArray`` construction) for
pre-/post-alignment. This wrapper is the reusable form Guidance publishes
through (``docs/guidance_node_implementation_plan.md`` decision 3), instead of
adding a new alignment-specific service/topic.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Pose, PoseArray

CHECKPOINT_TOPIC = "/gnc/checkpoints"
DEFAULT_REFERENCE_FRAME = "iss_body"

# Matches this package's other publishers (reliable; see
# docs/future_design_notes.md 6-4).
DEFAULT_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


class CheckpointPublisher:
    """Publish a single ``(pos, quat)`` checkpoint to ``/gnc/checkpoints``.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: Checkpoint topic name.
        reference_frame: ``frame_id`` stamped on the message (must match
            ``PoseArraySubscriber``'s ``expected_frame`` on the Control side).
        qos_profile: QoS for the publisher (default: reliable, see
            ``DEFAULT_QOS``).
    """

    def __init__(self, node: Node, topic: str = CHECKPOINT_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 qos_profile: QoSProfile = DEFAULT_QOS) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._pub = node.create_publisher(PoseArray, topic, qos_profile)

    def publish(self, pos, quat) -> None:
        """Publish a single-pose checkpoint array ``[pos]``/``[quat]``."""
        msg = PoseArray()
        msg.header.frame_id = self._reference_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = pos
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = quat
        msg.poses.append(pose)
        self._pub.publish(msg)

    def wait_for_subscriber(self, timeout_sec: float = 5.0, spin_fn=None) -> bool:
        """Poll until a subscriber has matched, or timeout.

        A fixed sleep before the first ``publish()`` isn't guaranteed to win
        the discovery race (dropped a checkpoint silently on one run, see
        docs/trajectory_force_duration_investigation.md 6-7) -- poll the
        actual match count instead.

        Args:
            spin_fn: callable ``spin_fn(seconds)`` used to pace the poll
                (e.g. the caller's own sim-time-based ``spin_fn``, see
                ``GuidanceExecutor``). Must NOT call ``rclpy.spin_once`` on
                a node that's already being spun by another executor --
                this method itself never spins the node directly, since it
                is invoked from within an action server's execute_callback,
                which is already running on a MultiThreadedExecutor thread
                (see docs/guidance_move_to_debug_2026-08-20.md). Defaults to
                a plain 0.05s wall-clock poll interval for the standalone
                ``main()`` CLI test below, where no executor is spinning
                concurrently.
        """
        spin_fn = spin_fn or (lambda seconds: time.sleep(seconds))
        deadline = time.monotonic() + timeout_sec
        while (rclpy.ok() and self._pub.get_subscription_count() < 1
               and time.monotonic() < deadline):
            spin_fn(0.05)
        return self._pub.get_subscription_count() >= 1


def main(args=None) -> None:
    """Standalone manual test: publish a fixed checkpoint once.

    Run with ``ros2 run sobits_intball2_gnc checkpoint_publisher [options]``.
    """
    import argparse
    import sys
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="checkpoint_publisher",
        description="Manual test for CheckpointPublisher: publish a single "
                    "fixed (pos, quat) checkpoint to /gnc/checkpoints.",
    )
    parser.add_argument("--topic", default=CHECKPOINT_TOPIC,
                        help="checkpoint topic (default: %(default)s)")
    parser.add_argument("--frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="checkpoint position")
    parser.add_argument("--quat", type=float, nargs=4,
                        default=[0.0, 0.0, 0.0, 1.0],
                        metavar=("X", "Y", "Z", "W"),
                        help="checkpoint orientation")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("checkpoint_publisher_test")
    pub = CheckpointPublisher(node, ns.topic, ns.frame)
    if not pub.wait_for_subscriber(timeout_sec=5.0):
        node.get_logger().warn("no subscriber matched after 5s, publishing anyway")
    pub.publish(ns.pos, ns.quat)
    node.get_logger().info("published checkpoint pos=%s quat=%s" % (ns.pos, ns.quat))
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
