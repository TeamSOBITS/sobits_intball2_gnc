#!/usr/bin/env python3
"""Send one checkpoint to /gnc/checkpoints and wait (sim time) until the vehicle holds it.

Changes only the given coordinates of the current TF pose, or only the attitude
(``--stereo-toward`` turns body +y, the stereo camera axis, at a point without
moving). Republishes until the vehicle starts moving and exits non-zero if it
never does, since a single publish right after discovery can be dropped.

Usage:
    python3 test/manual/hold_checkpoint.py --x 10.686 [--y ...] [--z ...] [--timeout 60]
    python3 test/manual/hold_checkpoint.py --stereo-toward 10.936 -9.0 5.0
"""
import argparse
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from compare_stereo_depth_to_octomap import quat_to_matrix
from virtual_obstacle import spin_for_sim

CHECKPOINT_TOPIC = "/gnc/checkpoints"
REPUBLISH_TRIES = 10
ARRIVED_M = 0.01
ARRIVED_DEG = 1.0
SETTLE_SIM_S = 15.0


def matrix_to_quat(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2 * np.sign(r[2, 1] - r[1, 2] or 1)
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2 * np.sign(r[0, 2] - r[2, 0] or 1)
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2 * np.sign(r[1, 0] - r[0, 1] or 1)
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def stereo_toward(pos, quat, target):
    y_axis = np.asarray(target, dtype=float) - pos
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, quat_to_matrix(quat)[:, 2])
    x_axis /= np.linalg.norm(x_axis)
    return matrix_to_quat(np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)]))


def angle_deg(q0, q1):
    return float(2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--stereo-toward", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--timeout", type=float, default=60.0, help="sim seconds to reach the goal")
    args = parser.parse_args()

    rclpy.init()
    node = Node("hold_checkpoint")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    tf_client = TfClient(node, "iss_body", "body")
    try:
        if not tf_client.wait_for_frame(timeout_sec=5.0):
            print("TF unavailable: iss_body <- body")
            return 1
        pos, quat, _ = tf_client.get_pose()
        pos, quat = np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
        goal = pos.copy()
        for axis, value in enumerate((args.x, args.y, args.z)):
            if value is not None:
                goal[axis] = value
        goal_quat = stereo_toward(pos, quat, args.stereo_toward) if args.stereo_toward else quat

        def errors():
            p, q, _ = tf_client.get_pose()
            return float(np.linalg.norm(np.asarray(p) - goal)), angle_deg(q, goal_quat)

        start_m, start_deg = errors()
        print(f"goal pos {np.round(goal, 4)} quat {np.round(goal_quat, 4)}, "
              f"start error {start_m * 1e3:.1f}mm {start_deg:.1f}deg", flush=True)

        def arrived():
            e_m, e_deg = errors()
            return e_m < ARRIVED_M and e_deg < ARRIVED_DEG

        def started():
            e_m, e_deg = errors()
            return arrived() or e_m < start_m - 0.005 or e_deg < start_deg - 0.5

        pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 1)
        spin_for_sim(node, 5.0, lambda: pub.get_subscription_count() > 0)
        msg = PoseArray()
        msg.header.frame_id = "iss_body"
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, goal)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, goal_quat)
        msg.poses.append(pose)
        for attempt in range(REPUBLISH_TRIES):
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            if spin_for_sim(node, 1.0, started):
                break
            print(f"no motion yet, republish {attempt + 1}", flush=True)
        else:
            print(f"vehicle did not move after {REPUBLISH_TRIES} publishes; is control_node running?")
            return 1
        print("moving", flush=True)

        reached = spin_for_sim(node, args.timeout, arrived)
        spin_for_sim(node, SETTLE_SIM_S)
        e_m, e_deg = errors()
        print(f"{'arrived' if reached else 'NOT arrived'}: error {e_m * 1e3:.1f}mm {e_deg:.2f}deg after settling", flush=True)
        return 0 if reached else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
