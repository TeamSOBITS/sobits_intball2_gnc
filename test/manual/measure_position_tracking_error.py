#!/usr/bin/env python3
"""Measure the actual POSITION tracking error (TF p_now vs /gnc/trajectory_setpoint's
p_des), same rationale as measure_attitude_tracking_error.py but for translation.

Written because the attitude-only measurement had no visibility into whether the
vehicle was also drifting off the planned path in position -- a gap noticed when
recording-time CPU load spikes (docs/trajectory_force_duration_investigation.md
6-19/6-20-ish) were suspected of degrading control, but only attitude error had
actually been checked.

Manual verification script (test/manual/): requires a running sim + gnc launch,
not collected by pytest. See test/manual/README.md.
"""
import csv
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.multi_dof_joint_trajectory_subscriber import (
    MultiDOFJointTrajectorySubscriber,
)

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 5.0
OUT_CSV = "/tmp/position_tracking_error.csv"


def main():
    rclpy.init(args=sys.argv)
    node_name = f"measure_position_tracking_error_{int(time.monotonic() * 1000) % 1000000}"
    node = Node(
        node_name,
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    traj_sub = MultiDOFJointTrajectorySubscriber(node, expected_frame=REFERENCE_FRAME)

    node.get_logger().info(f"[{node_name}] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"[{node_name}] could not get a TF pose")
        return

    node.get_logger().info(
        f"[{node_name}] logging position tracking error to {OUT_CSV} at {RATE_HZ} Hz. "
        "Ctrl-C to stop."
    )

    rows = []
    dt = 1.0 / RATE_HZ
    # sim clock (not time.monotonic()) so logged elapsed time matches sim
    # time, comparable to the TF stamps / trajectory setpoints it's measured
    # against.
    t_start = node.get_clock().now()
    max_err = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            pose = tf_client.get_pose()
            if pose is not None and traj_sub.ready:
                pos_now, _quat_now, _stamp = pose
                p_des = np.asarray(traj_sub.p_des, dtype=float)
                err_m = float(np.linalg.norm(np.asarray(pos_now) - p_des))
                elapsed = (node.get_clock().now() - t_start).nanoseconds * 1e-9
                rows.append((elapsed, err_m))
                max_err = max(max_err, err_m)
                if len(rows) % int(RATE_HZ * 2) == 0:  # every ~2s
                    node.get_logger().info(
                        f"[{node_name}] t={elapsed:.1f}s err={err_m * 1000:.1f}mm "
                        f"(max so far={max_err * 1000:.1f}mm)"
                    )
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "position_error_m"])
            writer.writerows(rows)
        if rows:
            errs = [r[1] for r in rows]
            node.get_logger().info(
                f"[{node_name}] {len(rows)} samples written to {OUT_CSV}. "
                f"max={max(errs) * 1000:.1f}mm mean={sum(errs) / len(errs) * 1000:.1f}mm "
                f"final={errs[-1] * 1000:.1f}mm"
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
