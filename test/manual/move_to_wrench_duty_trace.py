#!/usr/bin/env python3
"""Trigger move_to <location> while logging /ctl/duty, /ctl/wrench (desired),
and /ctl/wrench_achieved (event-driven, one row per message) for the whole
action lifetime -- for diagnosing whether a large attitude tracking error is
caused by fan-duty/torque saturation (thrust_allocator's shared force/torque
budget being crushed by an oversized torque request) rather than the
attitude reference itself, see
docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md.

Usage:
    python3 test/manual/move_to_wrench_duty_trace.py near_dock --out-csv /tmp/trace.csv
"""
import argparse
import csv
import sys

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

OUT_CSV_DEFAULT = "/tmp/move_to_wrench_duty_trace.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("location_name")
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--timeout-sec", type=float, default=90.0)
    ap.add_argument("--pose-rate-hz", type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("move_to_wrench_duty_trace")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    rows = []

    def stamp_now():
        return node.get_clock().now().nanoseconds * 1e-9

    def on_duty(msg):
        rows.append((stamp_now(), "duty", list(msg.data), None, None, None))

    def on_wrench(msg):
        f, t = msg.wrench.force, msg.wrench.torque
        rows.append((stamp_now(), "wrench", None, [f.x, f.y, f.z, t.x, t.y, t.z], None, None))

    def on_wrench_achieved(msg):
        f, t = msg.wrench.force, msg.wrench.torque
        rows.append((stamp_now(), "wrench_achieved", None, None, [f.x, f.y, f.z, t.x, t.y, t.z], None))

    node.create_subscription(Float64MultiArray, "/ctl/duty", on_duty, 10)
    node.create_subscription(WrenchStamped, "/ctl/wrench", on_wrench, 10)
    node.create_subscription(WrenchStamped, "/ctl/wrench_achieved", on_wrench_achieved, 10)

    tf_client = TfClient(node, "iss_body", "body")
    tf_client.wait_for_frame(timeout_sec=10.0)

    def on_pose_tick():
        pose = tf_client.get_pose()
        if pose is None:
            return
        pos, quat, _stamp = pose
        rows.append((stamp_now(), "pose", None, None, None, list(pos) + list(quat)))

    node.create_timer(1.0 / args.pose_rate_hz, on_pose_tick)

    client = MoveToClient(node)
    resolved = client.resolve_location(args.location_name)
    if resolved is None:
        print(f"could not resolve '{args.location_name}' via TF")
        return 1
    pos, quat = resolved
    print(f"resolved {args.location_name} -> pos={pos} quat={quat}, sending goal...")

    def on_feedback(fb):
        pass

    result = None
    try:
        result = client.send_goal(pos, quat, feedback_cb=on_feedback,
                                   timeout_sec=args.timeout_sec)
    except Exception as exc:  # noqa: BLE001 -- still want the CSV written below
        print(f"send_goal raised {exc!r}, writing whatever was logged so far")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "topic", "duty", "wrench", "wrench_achieved", "pose"])
        w.writerows(rows)

    print(f"finished result={'None' if result is None else result.type}, "
          f"logged {len(rows)} rows -> {args.out_csv}")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
