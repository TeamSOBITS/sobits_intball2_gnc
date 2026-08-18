#!/usr/bin/env python3
"""Move the vehicle to the 'nav_entry' named location via /gnc/checkpoints.

Coordinates from maps/iss_location.yaml (iss_body frame).

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose

CHECKPOINT_TOPIC = "/gnc/checkpoints"
REFERENCE_FRAME = "iss_body"

NAV_ENTRY_POS = [11.0, -4.3, 5.0]
NAV_ENTRY_QUAT = [-0.707106, 0.707106, 0.0, 0.0]  # x,y,z,w


def main():
    rclpy.init(args=sys.argv)
    node = Node("send_to_nav_entry")
    pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 1)

    time.sleep(1.0)  # let the publisher match with the subscriber

    msg = PoseArray()
    msg.header.frame_id = REFERENCE_FRAME
    msg.header.stamp = node.get_clock().now().to_msg()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = NAV_ENTRY_POS
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = NAV_ENTRY_QUAT
    msg.poses.append(pose)
    pub.publish(msg)
    node.get_logger().info(f"[send_to_nav_entry] published checkpoint at nav_entry: {NAV_ENTRY_POS}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
