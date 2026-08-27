#!/usr/bin/env python3
"""Continuously log TF pose (position + quaternion) to CSV.

Written for docs/imu_mode_drift_measurement_plan.md: measuring how far
`hover_control.mode:=imu` (no TF feedback) drifts from the TF ground truth
(`iss_body <- body`) compared to `hover_control.mode:=tf_imu`. Unlike
get_pose.py (one-shot), this samples on a sim-clock timer and appends one row
per tick until Ctrl-C or --duration (sim seconds) elapses.

Position/attitude only, matching the plan's "keep it simple" decision --
no control-internal quantities (force_imu/torque_imu etc.) are recorded.
Rotation is logged as the raw TF quaternion (not RPY): docs/archive/achieved/
phase0_presentation_data.md found RPY distorted by this vehicle's ~176 deg
attitude offset, whereas quaternion lets the presentation step choose RPY or
geodesic angle later.

Run once per experiment phase (tf_imu pre-position, imu, tf_imu recovery) --
each invocation is independent and does not know about mode; note the mode
being tested via --label so rows can be filtered after the fact. This script
never touches hover_control.mode itself: switching modes is a separate
control_node restart (see the plan doc; the parameter is read only once at
node startup and is not covered by the dynamic-parameter callback).

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.

Usage:
    python3 test/manual/log_pose_drift.py --output /tmp/imu_drift.csv \\
        --duration 60 --rate 50 --label imu
"""
import argparse
import csv
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-frame", default=REFERENCE_FRAME)
    parser.add_argument("--target-frame", default=TARGET_FRAME)
    parser.add_argument("--timeout", type=float, default=8.0,
                         help="seconds to wait for the TF frame at startup")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--duration", type=float, default=0.0,
                         help="sim seconds to log; 0 = until Ctrl-C (default: 0)")
    parser.add_argument("--rate", type=float, default=5.0, help="sample rate [Hz]")
    parser.add_argument("--label", default="",
                         help="free-text tag written to every row, e.g. the "
                              "hover_control.mode under test")
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = Node(
        "log_pose_drift",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    pose_source = TfClient(node, args.reference_frame, args.target_frame)

    if not pose_source.wait_for_frame(timeout_sec=args.timeout):
        node.get_logger().error(
            "[log_pose_drift] TF unavailable: %s <- %s (waited %.1fs); aborting"
            % (args.reference_frame, args.target_frame, args.timeout)
        )
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info(
        "[log_pose_drift] logging to %s at %.1f Hz (label=%r). Ctrl-C to stop."
        % (args.output, args.rate, args.label)
    )

    rows = []
    t_start = node.get_clock().now()
    stop_requested = False

    def on_tick():
        nonlocal stop_requested
        elapsed = (node.get_clock().now() - t_start).nanoseconds * 1e-9
        pose = pose_source.get_pose()
        if pose is None:
            node.get_logger().warn(
                "[log_pose_drift] t=%.2fs no transform available; skipping sample"
                % elapsed
            )
        else:
            pos, quat, stamp = pose
            rows.append((
                elapsed, stamp,
                pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0,
                quat[0], quat[1], quat[2], quat[3],
                args.label,
            ))
        if args.duration > 0.0 and elapsed >= args.duration:
            node.get_logger().info(
                "[log_pose_drift] reached duration %.1fs (%d samples); stopping"
                % (args.duration, len(rows))
            )
            stop_requested = True

    node.create_timer(1.0 / args.rate, on_tick)

    try:
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sim", "tf_stamp",
                "px_mm", "py_mm", "pz_mm",
                "qx", "qy", "qz", "qw",
                "label",
            ])
            writer.writerows(rows)
        node.get_logger().info(
            "[log_pose_drift] %d samples written to %s" % (len(rows), args.output)
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
