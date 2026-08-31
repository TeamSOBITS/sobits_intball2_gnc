#!/usr/bin/env python3
"""Phase 3b manual check: measure the actual attitude-tracking error (NOT just
whether torque_corr hits its clamp) while a facing-direction-of-travel
trajectory setpoint is being followed.

Subscribes to the live TF pose (iss_body <- body, actual attitude) and the
Guidance-side setpoint (/gnc/trajectory_setpoint's q_des), and logs the
quaternion geodesic angle between them at a fixed rate -- same metric used
for the JAXA-baseline comparison (docs/archive/achieved/phase0_5_findings.md),
so results are directly comparable.

Run this WHILE a trajectory-setpoint script (e.g.
send_curve_via_naventry_to_near_dock_facing_direction.py) is sending
/gnc/trajectory_setpoint, in a separate terminal/process.

Also derives a theory-based kp_att recommendation (docs/
trajectory_force_duration_investigation.md 6-8): a P+D attitude controller
has a nonzero steady-state tracking error against a reference with nonzero
angular ACCELERATION (not just angular velocity) -- e_ss ~= I * |theta_ref''|
/ kp_att, the "type-0 response to a parabolic input" result. Peak curvature
(e.g. the nav_entry crossing on the cobra-maneuver curve) is exactly where
q_des(t)'s angular acceleration peaks, which is why the residual tracking
error concentrates there regardless of q_des(t) generation strategy (6-5,
6-7). This script numerically differentiates the logged q_des(t) twice to
estimate that peak angular acceleration, then inverts the formula for a
target tracking error to recommend kp_att directly instead of guessing
multipliers.

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
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
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 5.0
OUT_CSV = "/tmp/attitude_tracking_error.csv"

# Per-axis moment of inertia [kg*m^2], x/y/z (docs/archive/achieved/
# phase0_5_findings.md, despin + known-torque step test). Using the largest
# axis is the conservative (worst-case) choice for the kp_att recommendation
# below: whichever axis the peak rotation happens to be about, this ensures
# kp_att is high enough.
I_MAX = 0.0470  # kg*m^2
TARGET_ERROR_DEG = 5.0  # desired steady-state tracking error at peak curvature


def geodesic_angle_rad(q_a, q_b):
    """Quaternion geodesic angle [rad] between two orientations."""
    qe = quat_mul(quat_conj(q_a), q_b)
    w = float(np.clip(abs(qe[3]), 0.0, 1.0))
    return 2.0 * np.arccos(w)


def geodesic_angle_deg(q_des, q_now):
    """Quaternion geodesic angle [deg] between q_des and q_now.

    Same definition as docs/archive/achieved/phase0_5_findings.md's JAXA
    comparison (NOT the RPY-based proxy, which is distorted by this
    vehicle's own attitude offset -- see that doc's observation B).
    """
    return np.degrees(geodesic_angle_rad(q_des, q_now))


def estimate_peak_angular_acceleration(t_list, q_des_list, exclude_last_sec=3.0):
    """Numerically differentiate q_des(t) twice to estimate peak |theta''|.

    Returns (peak_accel_rad_s2, t_at_peak). Angular speed is estimated as the
    geodesic angle between consecutive q_des samples divided by dt (a scalar
    magnitude, not a signed per-axis rate); acceleration is the same finite
    difference applied to that speed profile. This is a magnitude estimate
    for the kp_att formula below, not a full 3D angular-acceleration vector.

    ``exclude_last_sec``: drops samples within this many seconds of when
    q_des(t) actually stops moving (not of when the log stops recording --
    the sender keeps publishing the final point long after the trajectory
    ends, so cutting from the log's own end wouldn't reach the transition at
    all if the log runs on for a while, as 6-9 found). Near a quintic
    trajectory's tail, v_des's magnitude approaches zero and
    compute_q_des's direction estimate (v_des/|v_des|) becomes
    ill-conditioned there, plus the "hold below speed_threshold" branch
    introduces a genuine discontinuity in q_des's derivative -- neither is
    the smooth, physically-driven acceleration this estimate is meant to
    capture (confirmed: an early run's "peak" landed right at the
    trajectory's nominal end, not at the curve's actual peak-curvature
    crossing -- docs/trajectory_force_duration_investigation.md 6-8).
    """
    if not t_list:
        return 0.0, 0.0
    # Find when q_des(t) last differed meaningfully from its final (held)
    # value -- that's the trajectory's actual end, independent of how much
    # longer the log kept recording the held setpoint afterward.
    still_rad = np.radians(0.5)
    q_final = q_des_list[-1]
    t_motion_end = t_list[-1]
    for i in range(len(t_list) - 1, -1, -1):
        if geodesic_angle_rad(q_des_list[i], q_final) > still_rad:
            t_motion_end = t_list[i]
            break
    cutoff = t_motion_end - exclude_last_sec
    speeds = []
    speed_times = []
    for i in range(1, len(t_list)):
        if t_list[i] > cutoff:
            break
        dt = t_list[i] - t_list[i - 1]
        if dt <= 1e-6:
            continue
        speed = geodesic_angle_rad(q_des_list[i - 1], q_des_list[i]) / dt
        speeds.append(speed)
        speed_times.append(0.5 * (t_list[i] + t_list[i - 1]))

    peak_accel = 0.0
    peak_t = 0.0
    for i in range(1, len(speeds)):
        dt = speed_times[i] - speed_times[i - 1]
        if dt <= 1e-6:
            continue
        accel = abs(speeds[i] - speeds[i - 1]) / dt
        if accel > peak_accel:
            peak_accel = accel
            peak_t = speed_times[i]
    return peak_accel, peak_t


def main():
    rclpy.init(args=sys.argv)
    node_name = f"measure_attitude_tracking_error_{int(time.monotonic() * 1000) % 1000000}"
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
        f"[{node_name}] logging geodesic tracking error to {OUT_CSV} at {RATE_HZ} Hz. "
        "Ctrl-C to stop."
    )

    rows = []
    q_des_history = []  # (t, q_des) for the angular-acceleration estimate
    dt = 1.0 / RATE_HZ
    # sim clock (not time.monotonic()) so logged elapsed time, and the
    # angular-acceleration estimate derived from it, matches sim time.
    t_start = node.get_clock().now()
    max_err = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            pose = tf_client.get_pose()
            if pose is not None and traj_sub.ready:
                _pos, quat_now, _stamp = pose
                q_des = np.asarray(traj_sub.q_des, dtype=float)
                err_deg = geodesic_angle_deg(q_des, np.asarray(quat_now))
                elapsed = (node.get_clock().now() - t_start).nanoseconds * 1e-9
                rows.append((elapsed, err_deg))
                q_des_history.append((elapsed, q_des))
                max_err = max(max_err, err_deg)
                if len(rows) % int(RATE_HZ * 2) == 0:  # every ~2s
                    node.get_logger().info(
                        f"[{node_name}] t={elapsed:.1f}s err={err_deg:.3f}deg "
                        f"(max so far={max_err:.3f}deg)"
                    )
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "geodesic_angle_deg"])
            writer.writerows(rows)
        if rows:
            errs = [r[1] for r in rows]
            node.get_logger().info(
                f"[{node_name}] {len(rows)} samples written to {OUT_CSV}. "
                f"max={max(errs):.3f}deg mean={sum(errs) / len(errs):.3f}deg "
                f"final={errs[-1]:.3f}deg"
            )
        if len(q_des_history) > 2:
            t_list = [r[0] for r in q_des_history]
            q_list = [r[1] for r in q_des_history]
            peak_accel, peak_t = estimate_peak_angular_acceleration(t_list, q_list)
            target_error_rad = np.radians(TARGET_ERROR_DEG)
            recommended_kp_att = (
                I_MAX * peak_accel / target_error_rad if target_error_rad > 0 else float("inf")
            )
            node.get_logger().info(
                f"[{node_name}] peak |q_des angular accel| = {peak_accel:.4f} rad/s^2 "
                f"at t={peak_t:.1f}s. Theory (e_ss ~= I*theta_ref''/kp_att, "
                f"I_max={I_MAX} kg*m^2, target error={TARGET_ERROR_DEG} deg): "
                f"recommended kp_att >= {recommended_kp_att:.4f} Nm"
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
