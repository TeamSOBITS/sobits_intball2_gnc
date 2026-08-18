#!/usr/bin/env python3
"""Phase 1 manual test: publish a 3-point checkpoint array to /gnc/checkpoints.

Points are small offsets (a few cm) from the current TF hold position, in the
iss_body frame, so the vehicle stays well clear of any structure while still
producing a visible move between checkpoints.

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose

from sobits_intball2_gnc.control.ros.tf_client import TfClient

CHECKPOINT_TOPIC = "/gnc/checkpoints"
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
# Cumulative per-axis offsets [m] from the current hold position, applied one
# axis at a time (checkpoint N adds STEP[N-1] on top of checkpoint N-1).
STEP = [0.25, 0.30, 0.20]  # x, y, z


def main():
    rclpy.init(args=sys.argv)
    node = Node("send_checkpoints")
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 1)

    node.get_logger().info("[send_checkpoints] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=5.0):
        node.get_logger().error("[send_checkpoints] could not get current TF pose")
        return
    pos0, quat0, _ = tf_client.get_pose()
    node.get_logger().info(f"[send_checkpoints] current pose: pos={pos0} quat={quat0}")

    # 4 checkpoints: current position, then +STEP[0] in x, +STEP[1] in y,
    # +STEP[2] in z, cumulatively (one axis added per checkpoint).
    checkpoints = [(pos0, quat0)]
    cur = list(pos0)
    for axis, step in enumerate(STEP):
        cur = list(cur)
        cur[axis] += step
        checkpoints.append((list(cur), quat0))

    msg = PoseArray()
    msg.header.frame_id = REFERENCE_FRAME
    msg.header.stamp = node.get_clock().now().to_msg()
    for p, q in checkpoints:
        pose_msg = Pose()
        pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = p
        (pose_msg.orientation.x, pose_msg.orientation.y,
         pose_msg.orientation.z, pose_msg.orientation.w) = q
        msg.poses.append(pose_msg)

    # Give the publisher time to match with the subscriber before sending.
    time.sleep(1.0)
    pub.publish(msg)
    node.get_logger().info(f"[send_checkpoints] published {len(checkpoints)} checkpoints:")
    for i, (p, _) in enumerate(checkpoints):
        node.get_logger().info(f"  [{i}] {p}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
