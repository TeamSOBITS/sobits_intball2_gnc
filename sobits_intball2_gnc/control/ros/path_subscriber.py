#!/usr/bin/env python3
"""Checkpoint path subscriber for IntBall2 (`/gnc/checkpoints`).

ROS I/O wrapper (does not subclass Node): receives a checkpoint array
(``geometry_msgs/PoseArray``, expressed in the TF reference frame) as the
interface for the future free-path flight program, and forwards the parsed list
of ``(pos, quat)`` poses to a caller-supplied callback (typically the hover
controller's pose corrector).

The array's ``frame_id`` is validated against the expected reference frame. The
checkpoint frame changed from the DS frame to the TF reference frame, and a
path published in the old frame would otherwise be followed silently to the
wrong place.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

CHECKPOINT_TOPIC = "/gnc/checkpoints"


class PathSubscriber:
    """Subscribe to a checkpoint ``PoseArray`` and expose the latest path.

    Args:
        node: The rclpy Node that owns this subscriber.
        topic: Checkpoint topic name.
        on_path: Optional callback ``(list_of_(pos, quat))`` invoked per message.
        expected_frame: Reference frame the poses must be expressed in. An
            empty ``frame_id`` is accepted as "unspecified"; any other
            mismatch rejects the whole array.
    """

    def __init__(self, node: Node, topic: str = CHECKPOINT_TOPIC, on_path=None,
                 expected_frame: str = "") -> None:
        self._node = node
        self._on_path = on_path
        self._expected_frame = expected_frame
        self._checkpoints = []
        self._sub = node.create_subscription(PoseArray, topic, self._callback, 1)
        node.get_logger().info(
            "[PathSubscriber] subscribing %s (frame: %s)"
            % (topic, expected_frame or "<any>")
        )

    def _callback(self, msg: PoseArray) -> None:
        frame = msg.header.frame_id
        if self._expected_frame and frame and frame != self._expected_frame:
            # Reject wholesale: a partially applied path is worse than none.
            self._node.get_logger().warn(
                "[PathSubscriber] discarding %d checkpoints in frame '%s': "
                "expected '%s'"
                % (len(msg.poses), frame, self._expected_frame)
            )
            return
        poses = [
            ([p.position.x, p.position.y, p.position.z],
             [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            for p in msg.poses
        ]
        self._checkpoints = poses
        self._node.get_logger().info(
            "[PathSubscriber] checkpoints: %d received%s"
            % (len(poses), "" if poses else " (cleared)")
        )
        if self._on_path is not None:
            self._on_path(poses)

    @property
    def checkpoints(self):
        """Latest list of ``(pos, quat)`` checkpoints (empty if none)."""
        return self._checkpoints


def main(args=None) -> None:
    """Standalone manual test: report received checkpoint arrays.

    Run with ``ros2 run sobits_intball2_gnc path_subscriber [options]``.
    """
    import argparse
    import sys
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="path_subscriber",
        description="Manual test for PathSubscriber: log checkpoint PoseArrays "
                    "received on the checkpoint topic.",
    )
    parser.add_argument("--topic", default=CHECKPOINT_TOPIC,
                        help="checkpoint PoseArray topic (default: %(default)s)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("path_subscriber_test")
    PathSubscriber(node, ns.topic)
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
