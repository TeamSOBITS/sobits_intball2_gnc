#!/usr/bin/env python3
"""Offline (no ROS, no sim) rigid-body simulation of the attitude PD law,
to test whether composite-axis overshoot (docs/2026-08-27_composite_axis_
overshoot_next_steps.md) is inherent to attitude_error_to_torque()'s
geometry, independent of the thrust allocator/actuator/bridge.

Integrates q_dot = 0.5 * q (x) [omega, 0] and omega_dot = torque / I (the
gyroscopic coupling term omega x (I omega) is exactly zero here because the
sim's inertia is isotropic, see docs/2026-08-27_sim_ground_truth_params.md
-- so it's omitted rather than computed-and-discarded). No thrust allocator,
no duty saturation, no actuator/bridge lag: this is the control law's own
response to an idealized, infinitely-fast actuator.

If this idealized sim still overshoots/oscillates for a composite-axis
offset, the control law's geometry (moving instantaneous torque axis) is
sufficient to explain the observed behavior. If it converges monotonically
while the real sim overshoots, some other factor (actuator lag, EMA filter
phase delay, duty saturation nonlinearity, ...) is responsible instead.

Usage:
    python3 test/manual/simulate_ideal_attitude_response.py \\
        --axis xyz --offset-deg 120 \\
        --kp-override 0.20 0.20 0.20 --kd-override 0.4635 0.4077 0.264 \\
        --max-torque 0.3 --out-csv /tmp/ideal_xyz_120.csv
"""
import argparse
import csv
import sys

import numpy as np

from sobits_intball2_gnc.control.utils.pose_control_law import (
    attitude_error_to_torque,
    clamp_torque,
)
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul

INERTIA_ISO = 0.0136  # kg*m^2, ground truth (docs/2026-08-27_sim_ground_truth_params.md)
DT_DEFAULT = 0.005
DURATION_S_DEFAULT = 25.0
OUT_CSV_DEFAULT = "/tmp/simulate_ideal_attitude_response_log.csv"

# Same mid-angle gains diagnose_align_gains.py defaults to, for apples-to-apples
# comparison against a real-sim CSV run with the same flags.
DEFAULT_KP_ATT = [0.20, 0.20, 0.20]
DEFAULT_KD_ATT = [0.4635, 0.4077, 0.264]
DEFAULT_MAX_TORQUE = 0.3


def qe_and_sign(target_quat, quat):
    qe = quat_mul(quat_conj(target_quat), quat)
    raw_w = qe[3]
    sign = np.sign(raw_w if raw_w != 0.0 else 1.0)
    angle_deg = float(np.degrees(2.0 * np.arccos(min(1.0, abs(raw_w)))))
    return qe, raw_w, sign, angle_deg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", default="z", choices=["x", "y", "z", "xyz"],
                     help="body-local rotation axis for the offset (default: z)")
    ap.add_argument("--offset-deg", type=float, default=180.0)
    ap.add_argument("--kp-override", type=float, nargs=3, default=DEFAULT_KP_ATT)
    ap.add_argument("--kd-override", type=float, nargs=3, default=DEFAULT_KD_ATT)
    ap.add_argument("--max-torque", type=float, default=DEFAULT_MAX_TORQUE)
    ap.add_argument("--alpha", type=float, default=0.3,
                     help="EMA filter weight on the finite-difference omega_err "
                          "(default: 0.3, matching tf_correction.att_filter_alpha)")
    ap.add_argument("--preserve-direction", action="store_true")
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--duration", type=float, default=DURATION_S_DEFAULT)
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    args = ap.parse_args()

    kp_att = np.array(args.kp_override, dtype=float)
    kd_att = np.array(args.kd_override, dtype=float)
    axis_vec = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "xyz": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }[args.axis]

    quat = np.array([0.0, 0.0, 0.0, 1.0])
    half = np.radians(args.offset_deg) / 2.0
    offset_q = np.array([*(np.sin(half) * axis_vec), np.cos(half)])
    target_quat = quat_mul(quat, offset_q)

    omega = np.zeros(3)
    omega_filtered = np.zeros(3)
    last_qe_vec = None

    rows = []
    last_sign = None
    flip_count = 0
    n_steps = int(args.duration / args.dt)

    for step in range(n_steps):
        t = step * args.dt
        qe, raw_w, sign, angle_deg = qe_and_sign(target_quat, quat)
        flipped = last_sign is not None and sign != last_sign
        if flipped:
            flip_count += 1
        last_sign = sign

        qe_vec = sign * qe[:3]
        omega_err_raw = np.zeros(3)
        if last_qe_vec is not None:
            omega_err_raw = (qe_vec - last_qe_vec) / args.dt
        last_qe_vec = qe_vec
        omega_filtered = (
            args.alpha * omega_err_raw + (1.0 - args.alpha) * omega_filtered
        )

        torque_raw = attitude_error_to_torque(
            kp_att, kd_att, target_quat, quat, omega_filtered, np.inf,
        )
        torque = clamp_torque(
            torque_raw, args.max_torque, preserve_direction=args.preserve_direction,
        )

        rows.append({
            "t": t,
            "angle_deg": angle_deg,
            "qe_w": raw_w,
            "sign": sign,
            "sign_flip": flipped,
            "omega_x": omega[0], "omega_y": omega[1], "omega_z": omega[2],
            "torque_x": torque[0], "torque_y": torque[1], "torque_z": torque[2],
        })

        # omega x (I omega) == 0 identically for isotropic I -- see module docstring.
        omega_dot = torque / INERTIA_ISO
        omega = omega + omega_dot * args.dt
        omega_quat = np.array([omega[0], omega[1], omega[2], 0.0])
        qdot = 0.5 * quat_mul(quat, omega_quat)
        quat = quat + qdot * args.dt
        quat = quat / np.linalg.norm(quat)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"logged {len(rows)} steps, {flip_count} sign flips -> {args.out_csv}")

    angles = np.array([r["angle_deg"] for r in rows])
    print(f"angle_deg: start={angles[0]:.2f} min={angles.min():.2f} "
          f"max_after_t0={angles[1:].max():.2f} final={angles[-1]:.2f}")

    print("\n--- sign-flip events (angle before -> after) ---")
    for idx, r in enumerate(rows):
        if r["sign_flip"] and idx > 0:
            prev = rows[idx - 1]
            print(f"t={r['t']:.2f}s  angle {prev['angle_deg']:.1f}->{r['angle_deg']:.1f} deg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
