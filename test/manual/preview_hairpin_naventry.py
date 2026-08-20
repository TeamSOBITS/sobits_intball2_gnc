#!/usr/bin/env python3
"""RViz-only preview of a sharpened near_dock <-> above_dock turn.

Endpoints are the existing named locations near_dock/above_dock (already
flown safely in this session's other tests). The via-point is nav_entry's
"bulge" from the near_dock/above_dock midpoint, scaled by --bulge-scale to
sharpen the turn angle beyond the ~127 deg the raw triangle already gives
(docs/trajectory_force_duration_investigation.md 6 section: exploring a
harsher attitude-tracking stress test than the current ~89 deg
near_dock<->above_dock_2 cobra maneuver, without traveling all the way out to
the far inspection_entry/capture_point locations).

Publishes the planned path to /gnc/trajectory_path for RViz only -- does NOT
send /gnc/trajectory_setpoint, so the vehicle does not move. Ctrl-C to exit
after previewing.

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.path_publisher import PathPublisher

TRAJECTORY_PATH_TOPIC = "/gnc/trajectory_path"
PATH_SAMPLE_DT = 0.1
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
DURATION_SEC = 40.0

# near_dock / above_dock / nav_entry, from maps/iss_location.yaml (iss_body
# frame). Using above_dock (not above_dock_2) here -- combined with
# near_dock and nav_entry it forms a naturally sharper (~127 deg) triangle
# than the near_dock<->above_dock_2 pair (~89 deg) uses, with comparable leg
# lengths (no need for a brand-new, unverified endpoint).
NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])


def quintic(tau):
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return s


def bezier(p0, c, p1, s):
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s ** 2 * p1


def turn_angle_deg(p0, w, p1):
    leg1 = w - p0
    leg2 = p1 - w
    cos_a = np.dot(leg1, leg2) / (np.linalg.norm(leg1) * np.linalg.norm(leg2))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def main():
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bulge-scale", type=float, default=1.5,
        help="Scale factor on nav_entry's offset from the near_dock/above_dock "
             "midpoint (1.0 = actual nav_entry, sharper turn as this grows).",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Preview above_dock -> near_dock instead of near_dock -> above_dock.",
    )
    ns = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = Node("preview_hairpin_naventry")
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    path_pub = PathPublisher(node, TRAJECTORY_PATH_TOPIC, reference_frame=REFERENCE_FRAME)

    node.get_logger().info("[preview_hairpin_naventry] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error("[preview_hairpin_naventry] could not get a TF pose")
        return
    _pos, quat0, _ = tf_client.get_pose()

    p0, p1 = (ABOVE_DOCK, NEAR_DOCK) if ns.reverse else (NEAR_DOCK, ABOVE_DOCK)
    midpoint = 0.5 * (p0 + p1)
    bulge = NAV_ENTRY - midpoint
    w = midpoint + ns.bulge_scale * bulge
    c = 2.0 * w - 0.5 * p0 - 0.5 * p1

    angle = turn_angle_deg(p0, w, p1)
    leg1_len = np.linalg.norm(w - p0)
    leg2_len = np.linalg.norm(p1 - w)
    node.get_logger().info(
        f"[preview_hairpin_naventry] bulge_scale={ns.bulge_scale} "
        f"p0={p0.tolist()} -> waypoint={w.tolist()} -> p1={p1.tolist()} "
        f"turn_angle={angle:.1f}deg leg_lengths=[{leg1_len:.3f}, {leg2_len:.3f}]m "
        f"total_path_len~={leg1_len + leg2_len:.3f}m"
    )

    n_samples = int(DURATION_SEC / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s = quintic(tau)
        p = bezier(p0, c, p1, s)
        samples.append((p.tolist(), quat0))
    path_pub.publish(samples)
    node.get_logger().info(
        f"[preview_hairpin_naventry] published {len(samples)}-point preview path to "
        f"{TRAJECTORY_PATH_TOPIC} for RViz. --path-only preview: not sending "
        "/gnc/trajectory_setpoint, vehicle will not move. Ctrl-C to exit."
    )
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
