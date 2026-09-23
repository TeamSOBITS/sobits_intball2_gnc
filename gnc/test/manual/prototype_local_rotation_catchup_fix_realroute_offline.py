#!/usr/bin/env python3
"""Offline (no ROS, no sim) follow-up to
``prototype_local_rotation_catchup_fix_offline.py``: that script confirmed
the receding-horizon rotation-catchup lag mechanism against a *synthetic,
fixed* 90 degree corner target. This script re-checks the same latch-based
fix idea against the *real* `nav_entry -> inspection_entry_1 ->
capture_point_2` route's actual `MincoTrajectory` global trajectory, whose
lookahead rotation target keeps moving throughout (``lookahead_t =
global_elapsed + t_local`` samples a continuously advancing point on the
route) -- the fixed-target synthetic scenario could not tell whether a naive
"latch the target and coast" fix would introduce new tracking error while
the real target itself is still turning under the catch-up window.

Two latch variants are compared against the current per-tick full-horizon
baseline:

- ``latch_frozen_target``: on error trigger, freeze both the deadline *and*
  the target rotvec/rate at the moment of the trigger (simplest version of
  the fix prototyped in the previous script).
- ``latch_shrinking_horizon``: on error trigger, freeze only the deadline
  (absolute time), but keep re-sampling the *current* lookahead target off
  the moving global trajectory every tick, solving a fresh Hermite blend
  each tick against a shrinking (not full-``t_local``) remaining duration.

No production code changes -- everything here calls the real
``MincoTrajectory``/``solve_quintic_hermite_coeffs``/``evaluate_vector``
production primitives directly.

Usage:
    python3 test/manual/prototype_local_rotation_catchup_fix_realroute_offline.py
"""
import math
import sys

import numpy as np
import yaml

from sobits_intball2_gnc.control.utils.quat_math import quat_exp, quat_mul, quat_rotate
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector
from sobits_intball2_gnc.guidance.utils.quintic_hermite import solve_quintic_hermite_coeffs

ISS_LOCATION_YAML = "/root/colcon_ws/src/sobits_intball2_gnc/gnc/maps/iss_location.yaml"
MASS_KG = 3.216
MAX_FORCE_XYZ = (0.181, 0.0996, 0.122)
MAX_ACCEL = min(MAX_FORCE_XYZ) / MASS_KG
TARGET_SPEED = 0.5
WRENCH_SAFETY_MARGIN = 0.7
VIA_HALF_WIDTH = 0.0
ATTITUDE_RESAMPLE_SPACING_M = 0.3
FORWARD_AXIS = np.array([1.0, 0.0, 0.0])

DT = 0.02
T_LOCAL = 1.0
ERROR_TRIGGER_DEG = 15.0
SETTLE_DEG = 5.0


def load_locations(names):
    with open(ISS_LOCATION_YAML) as f:
        doc = yaml.safe_load(f)
    poses = doc["location_pose"]
    out = {}
    for name in names:
        entry = poses[name]
        t = entry["translation"]
        r = entry["rotation"]
        out[name] = (
            np.array([t["x"], t["y"], t["z"]]),
            np.array([r["x"], r["y"], r["z"], r["w"]]),
        )
    return out


def fwd_v_angle_deg(rotvec, v_dir, q0):
    speed = np.linalg.norm(v_dir)
    if speed < 1e-6:
        return 0.0
    q = quat_mul(q0, quat_exp(rotvec))
    fwd = quat_rotate(q, FORWARD_AXIS)
    cos_a = np.dot(fwd, v_dir) / (np.linalg.norm(fwd) * speed)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def build_global_trajectory():
    names = ["nav_entry", "inspection_entry_1", "capture_point_2"]
    locs = load_locations(names)
    q0 = locs["nav_entry"][1]
    waypoints = np.array([locs[n][0] for n in names])
    traj = MincoTrajectory(
        waypoints, q0,
        forward_axis=tuple(FORWARD_AXIS), face_travel=True,
        via_half_width=VIA_HALF_WIDTH,
        attitude_resample_spacing_m=ATTITUDE_RESAMPLE_SPACING_M,
        wrench_safety_margin=WRENCH_SAFETY_MARGIN,
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
    )
    return traj, q0


def run(mode, traj, q0):
    """Simulate the local rotation-blend layer perfectly tracking the global
    trajectory's own position/velocity (isolating rotation catch-up
    dynamics, same simplification as the synthetic script), for ``mode`` in
    {"baseline", "latch_frozen_target", "latch_shrinking_horizon"}."""
    duration = traj.global_total_duration
    n_steps = int(duration / DT)

    rotvec, angvel, _ = traj.sample_rotvec_derivatives(0.0)
    rotvec = np.array(rotvec, dtype=float)
    angvel = np.array(angvel, dtype=float)

    latch_deadline = None
    latch_target = None  # (tgt_rv, tgt_rv_vel, tgt_rv_accel), frozen mode only

    ts = np.empty(n_steps + 1)
    angles = np.empty(n_steps + 1)
    _, v0, _, _ = traj.sample(0.0)
    ts[0] = 0.0
    angles[0] = fwd_v_angle_deg(rotvec, v0, q0)

    for i in range(1, n_steps + 1):
        t = i * DT
        lookahead_t = min(t + T_LOCAL, duration)
        tgt_rv, tgt_rv_vel, tgt_rv_accel = traj.sample_rotvec_derivatives(lookahead_t)
        _, v_now, _, _ = traj.sample(t)
        current_error = fwd_v_angle_deg(rotvec, v_now, q0)

        if mode == "baseline":
            coeffs = solve_quintic_hermite_coeffs(
                rotvec, angvel, np.zeros(3), tgt_rv, tgt_rv_vel, tgt_rv_accel, T_LOCAL
            )
            rotvec = evaluate_vector(coeffs, DT, order=0)
            angvel = evaluate_vector(coeffs, DT, order=1)

        elif mode == "latch_frozen_target":
            if latch_deadline is None and current_error > ERROR_TRIGGER_DEG:
                latch_deadline = t + T_LOCAL
                latch_target = (tgt_rv, tgt_rv_vel, tgt_rv_accel)
            if latch_deadline is not None:
                remaining = max(latch_deadline - t, DT)
                tr, trv, tra = latch_target
                coeffs = solve_quintic_hermite_coeffs(
                    rotvec, angvel, np.zeros(3), tr, trv, tra, remaining
                )
                rotvec = evaluate_vector(coeffs, DT, order=0)
                angvel = evaluate_vector(coeffs, DT, order=1)
                if t >= latch_deadline or current_error <= SETTLE_DEG:
                    latch_deadline = None
                    latch_target = None
            else:
                coeffs = solve_quintic_hermite_coeffs(
                    rotvec, angvel, np.zeros(3), tgt_rv, tgt_rv_vel, tgt_rv_accel, T_LOCAL
                )
                rotvec = evaluate_vector(coeffs, DT, order=0)
                angvel = evaluate_vector(coeffs, DT, order=1)

        elif mode == "latch_shrinking_horizon":
            if latch_deadline is None and current_error > ERROR_TRIGGER_DEG:
                latch_deadline = t + T_LOCAL
            if latch_deadline is not None:
                remaining = max(latch_deadline - t, DT)
                lookahead_shrink = min(t + remaining, duration)
                tgt_rv_s, tgt_rv_vel_s, tgt_rv_accel_s = traj.sample_rotvec_derivatives(
                    lookahead_shrink
                )
                coeffs = solve_quintic_hermite_coeffs(
                    rotvec, angvel, np.zeros(3), tgt_rv_s, tgt_rv_vel_s, tgt_rv_accel_s, remaining
                )
                rotvec = evaluate_vector(coeffs, DT, order=0)
                angvel = evaluate_vector(coeffs, DT, order=1)
                if t >= latch_deadline or current_error <= SETTLE_DEG:
                    latch_deadline = None
            else:
                coeffs = solve_quintic_hermite_coeffs(
                    rotvec, angvel, np.zeros(3), tgt_rv, tgt_rv_vel, tgt_rv_accel, T_LOCAL
                )
                rotvec = evaluate_vector(coeffs, DT, order=0)
                angvel = evaluate_vector(coeffs, DT, order=1)
        else:
            raise ValueError(mode)

        ts[i] = t
        _, v_after, _, _ = traj.sample(t)
        angles[i] = fwd_v_angle_deg(rotvec, v_after, q0)

    return ts, angles


def summarize(label, ts, angles):
    max_ang = float(np.max(angles))
    max_t = float(ts[int(np.argmax(angles))])
    above = angles > ERROR_TRIGGER_DEG
    total_time_above = float(np.sum(above)) * DT
    print(f"{label:26s} max_angle={max_ang:7.2f} deg (at t={max_t:6.2f}s)  "
          f"time_above_{ERROR_TRIGGER_DEG:.0f}deg={total_time_above:6.2f}s")


def main():
    traj, q0 = build_global_trajectory()
    duration = traj.global_total_duration
    print(f"real route: duration={duration:.2f}s\n")

    for mode, label in [
        ("baseline", "baseline (current prod)"),
        ("latch_frozen_target", "latch_frozen_target"),
        ("latch_shrinking_horizon", "latch_shrinking_horizon"),
    ]:
        ts, angles = run(mode, traj, q0)
        summarize(label, ts, angles)

    return 0


if __name__ == "__main__":
    sys.exit(main())
