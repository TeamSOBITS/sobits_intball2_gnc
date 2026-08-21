#!/usr/bin/env python3
"""Trigger move_to <location> while continuously logging TF pose (position +
rpy) at high rate for the ENTIRE action lifetime (pre_align, translation,
align_at_arrival), so pre_align's attitude AND translation can both be
inspected -- in particular whether translation starts before pre_align has
converged. Written for the pre_align/align_at_arrival investigation, see
docs/archive/achieved/2026-08-21_tf_correction_align_optimization.md and
docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md.

Runs the TF-logging timer and the MoveToClient's blocking send_goal on the
SAME node/executor, so spin_until_future_complete (inside send_goal) still
services the logging timer while it blocks waiting for the goal result.

Usage:
    python3 test/manual/move_to_full_trace.py capture_point_2 --out-csv /tmp/trace.csv
"""
import argparse
import csv
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

OUT_CSV_DEFAULT = "/tmp/move_to_full_trace.csv"


def rpy(q):
    x, y, z, w = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sinp))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("location_name")
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--rate-hz", type=float, default=20.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("move_to_full_trace")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    tf_client = TfClient(node, "iss_body", "body")
    if not tf_client.wait_for_frame(timeout_sec=10.0):
        print("TF unavailable, aborting")
        return 1

    rows = []

    def tick():
        pose = tf_client.get_pose()
        if pose is None:
            return
        pos, quat, stamp = pose
        r, p, y = rpy(quat)
        rows.append((stamp, pos[0], pos[1], pos[2], r, p, y,
                     quat[0], quat[1], quat[2], quat[3]))

    timer = node.create_timer(1.0 / args.rate_hz, tick)

    client = MoveToClient(node)
    resolved = client.resolve_location(args.location_name)
    if resolved is None:
        print(f"could not resolve '{args.location_name}' via TF")
        return 1
    pos, quat = resolved
    print(f"resolved {args.location_name} -> pos={pos} quat={quat}, sending goal...")

    def on_feedback(fb):
        pass  # position/attitude already captured by the tick() timer

    result = client.send_goal(pos, quat, feedback_cb=on_feedback, timeout_sec=90.0)
    timer.cancel()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw"])
        w.writerows(rows)

    if result is None:
        print(f"goal did not complete, logged {len(rows)} ticks -> {args.out_csv}")
    else:
        print(f"finished result.type={result.type}, logged {len(rows)} ticks -> {args.out_csv}")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
