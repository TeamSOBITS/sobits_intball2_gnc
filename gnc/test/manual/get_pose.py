#!/usr/bin/env python3
"""Print the current body pose from TF once, then exit.

A quick substitute for repeatedly typing `ros2 run tf2_ros tf2_echo`. Uses
`TfClient` (not a raw `TransformListener`) so it doesn't hit the
`/tf_static` base->body identity race -- see the note in
`common/ros/tf_client.py`.

Usage:
    python3 test/manual/get_pose.py [--reference-frame iss_body] [--target-frame body] [--timeout 5.0]
"""
import argparse
import math

import rclpy
from rclpy.node import Node

from sobits_intball2_gnc.common.ros.tf_client import TfClient


def quat_to_rpy_deg(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [math.degrees(a) for a in (roll, pitch, yaw)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-frame", default="iss_body")
    parser.add_argument("--target-frame", default="body")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("get_pose_once")
    node.set_parameters(
        [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)]
    )
    tf_client = TfClient(node, args.reference_frame, args.target_frame)

    try:
        if not tf_client.wait_for_frame(timeout_sec=args.timeout):
            print(
                f"TF unavailable: {args.reference_frame} <- {args.target_frame} "
                f"(waited {args.timeout:.1f}s)"
            )
            return

        pos, quat, stamp = tf_client.get_pose()
        rpy = quat_to_rpy_deg(quat)
        print(f"frame: {args.reference_frame} <- {args.target_frame}")
        print(f"stamp: {stamp:.3f}s (sim time)")
        print(f"pos:   x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}  [m]")
        print(f"quat:  x={quat[0]:.4f} y={quat[1]:.4f} z={quat[2]:.4f} w={quat[3]:.4f}")
        print(f"rpy:   roll={rpy[0]:.2f} pitch={rpy[1]:.2f} yaw={rpy[2]:.2f}  [deg]")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
