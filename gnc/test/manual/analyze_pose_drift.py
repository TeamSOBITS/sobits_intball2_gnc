#!/usr/bin/env python3
"""Compute net position/attitude drift between two timestamps in a
log_pose_drift.py CSV.

Position drift is the straight-line distance between the two samples nearest
--start-t and --end-t. Attitude drift is the quaternion geodesic angle
between them (sign/double-cover agnostic, matching docs/archive/achieved/
phase0_5_findings.md's more robust metric vs. RPY under a large attitude
offset). Does not touch ROS -- pure CSV post-processing.

Usage:
    python3 test/manual/analyze_pose_drift.py docs/results/imu_drift_60s.csv \\
        --start-t 5.0 --end-t 60.0
"""
import argparse
import csv
import math


def nearest_row(rows, t_target):
    return min(rows, key=lambda r: abs(float(r["t_sim"]) - t_target))


def geodesic_angle_deg(q0, q1):
    dot = abs(sum(a * b for a, b in zip(q0, q1)))
    dot = min(1.0, dot)
    return math.degrees(2.0 * math.acos(dot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV produced by log_pose_drift.py")
    parser.add_argument("--start-t", type=float, required=True,
                         help="t_sim of the drift's start reference (e.g. "
                              "past any warm-up transient)")
    parser.add_argument("--end-t", type=float, required=True,
                         help="t_sim of the drift's end reference")
    args = parser.parse_args()

    with open(args.csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.csv_path}: no rows")

    r0 = nearest_row(rows, args.start_t)
    r1 = nearest_row(rows, args.end_t)

    p0 = tuple(float(r0[k]) for k in ("px_mm", "py_mm", "pz_mm"))
    p1 = tuple(float(r1[k]) for k in ("px_mm", "py_mm", "pz_mm"))
    q0 = tuple(float(r0[k]) for k in ("qx", "qy", "qz", "qw"))
    q1 = tuple(float(r1[k]) for k in ("qx", "qy", "qz", "qw"))

    dx, dy, dz = (b - a for a, b in zip(p0, p1))
    dist_mm = math.sqrt(dx * dx + dy * dy + dz * dz)
    angle_deg = geodesic_angle_deg(q0, q1)

    print(f"start: t_sim={r0['t_sim']} pos_mm={p0} quat={q0}")
    print(f"end:   t_sim={r1['t_sim']} pos_mm={p1} quat={q1}")
    print(f"dx={dx:.1f} dy={dy:.1f} dz={dz:.1f} mm")
    print(f"position drift distance: {dist_mm:.1f} mm")
    print(f"attitude drift (geodesic angle): {angle_deg:.2f} deg")


if __name__ == "__main__":
    main()
