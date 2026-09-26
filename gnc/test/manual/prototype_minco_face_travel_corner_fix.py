#!/usr/bin/env python3
"""Offline (no ROS, no sim, no production-code changes) prototype/comparison of two
candidate fixes for the corner face-travel divergence found and confirmed in
`analyze_minco_face_travel_corner_offline.py` (up to 38 deg between MincoTrajectory's
own commanded attitude and its own commanded velocity direction, for ~10s around a via
waypoint -- reproduced identically in both `static_minco` live-sim data and the pure
offline global trajectory, so the root cause is `MincoTrajectory._waypoint_rotvecs`
seeding attitude from the *pre-solve* piecewise-straight-line path rather than the
*solved* corner-rounded position polynomial's actual local tangent -- see that
script's docstring and `docs/archive/achieved/
2026-08-28_attitude_waypoint_premature_rotation_root_cause.md`, the same anti-pattern
already found and fixed once for the static/TOPP-RA path).

Candidates:
  A) two-pass iterative: solve once with the current (buggy) straight-line rotvecs,
     sample the resulting solved position polynomial's actual velocity at each
     densified waypoint's arrival time, rebuild rotvecs from THAT tangent, re-solve.
  B) Hermite pre-seed: build a `HermiteSplineTrajectoryGenerator` path (the same class
     `ToppraTrajectory` uses) through the sparse waypoints first, sample ITS tangent
     at each densified waypoint location, seed rotvecs from that (no MINCO re-solve
     needed -- single solve, using a smoothed pre-estimate of the corner shape
     instead of the actual MINCO-solved one).

Both candidates reuse the project's own canonical helpers (`compute_q_des`,
`control/utils/quat_math`, `MincoTrajectory._densify`/`_call_minco`,
`HermiteSplineTrajectoryGenerator`) rather than hand-rolled math, and do not modify
`minco_trajectory.py` itself -- this script duplicates only the minimal
solve-and-sample bookkeeping needed to test a corrected `waypoints_flat` input.

Usage:
    python3 test/manual/prototype_minco_face_travel_corner_fix.py
"""
import math
import sys

import numpy as np
import yaml

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj, quat_exp, quat_log, quat_mul, unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoTrajectory, _DEGENERATE_TANGENT_THRESHOLD, _N_COEFFS, _N_DIMS,
)
from sobits_intball2_gnc.guidance.trajectory.generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

ISS_LOCATION_YAML = "/root/colcon_ws/src/sobits_intball2_gnc/gnc/maps/iss_location.yaml"
MASS_KG = 3.216
MAX_FORCE_XYZ = (0.181, 0.0996, 0.122)
MAX_ACCEL = min(MAX_FORCE_XYZ) / MASS_KG
TARGET_SPEED = 0.5
WRENCH_SAFETY_MARGIN = 0.7
VIA_HALF_WIDTH = 0.0
ATTITUDE_RESAMPLE_SPACING_M = 0.3
FORWARD_AXIS = np.array([1.0, 0.0, 0.0])


def load_locations(names):
    with open(ISS_LOCATION_YAML) as f:
        doc = yaml.safe_load(f)
    poses = doc["location_pose"]
    out = {}
    for name in names:
        entry = poses[name]
        t, r = entry["translation"], entry["rotation"]
        out[name] = (np.array([t["x"], t["y"], t["z"]]),
                     np.array([r["x"], r["y"], r["z"], r["w"]]))
    return out


def quat_rotate(q, v):
    x, y, z, w = q
    qv = np.array([x, y, z])
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2 * (w * uv + uuv)


class SolvedTrajectory:
    """Minimal re-implementation of MincoTrajectory's post-solve bookkeeping
    (segment_times/coeffs -> sample()), so candidate fixes can be solved with a
    custom waypoints_flat without going through MincoTrajectory.__init__'s own
    (buggy) rotvec derivation."""

    def __init__(self, waypoints_flat, q0, v0, w0):
        success, error_code, segment_times, coeffs_flat, duration = (
            MincoTrajectory._call_minco(
                waypoints_flat, v0.tolist(), w0.tolist(), VIA_HALF_WIDTH,
                WRENCH_SAFETY_MARGIN, TARGET_SPEED, MAX_ACCEL,
            )
        )
        if not success:
            raise RuntimeError(f"solve failed, error_code={error_code}")
        self._q0 = q0
        self._segment_times = np.asarray(segment_times, dtype=float)
        self._cum_times = np.concatenate([[0.0], np.cumsum(self._segment_times)])
        n_segments = len(self._segment_times)
        coeffs = np.asarray(coeffs_flat, dtype=float).reshape(n_segments, _N_DIMS, _N_COEFFS)
        self._pos_coeffs = coeffs[:, 0:3, :]
        self._rot_coeffs = coeffs[:, 3:6, :]
        self.duration = float(duration)

    def _segment_index(self, t):
        idx = int(np.searchsorted(self._cum_times, t, side="right")) - 1
        return min(max(idx, 0), len(self._segment_times) - 1)

    def sample(self, t):
        t = min(max(float(t), 0.0), self.duration)
        seg = self._segment_index(t)
        tau = t - self._cum_times[seg]
        p = evaluate_vector(self._pos_coeffs[seg], tau, order=0)
        v = evaluate_vector(self._pos_coeffs[seg], tau, order=1)
        rv = evaluate_vector(self._rot_coeffs[seg], tau, order=0)
        q = quat_mul(self._q0, quat_exp(rv))
        return p, v, q


def waypoints_flat_from(positions, rotvecs):
    flat = []
    for pos, rv in zip(positions, rotvecs):
        flat.extend(float(c) for c in pos)
        flat.extend(float(c) for c in rv)
    return flat


def rotvecs_from_tangents(tangents, q0):
    """Same face_travel policy as MincoTrajectory._waypoint_rotvecs/
    ToppraTrajectory._dense_travel_rotvecs, but driven by externally-supplied
    per-sample tangent vectors instead of straight-line waypoint differences."""
    n = len(tangents)
    rotvecs = np.zeros((n, 3))
    q_prev = q0.copy()
    for i in range(1, n):
        q_prev = compute_q_des(tangents[i], q_prev, _DEGENERATE_TANGENT_THRESHOLD, FORWARD_AXIS)
        raw = quat_log(quat_mul(quat_conj(q0), q_prev))
        rotvecs[i] = unwrap_rotvec(raw, rotvecs[i - 1])
    return rotvecs


def max_angle_near_via(traj, via_pos, window_s=15.0, dt=0.02):
    n = int(traj.duration / dt) + 1
    ts = np.linspace(0.0, traj.duration, n)
    dists, angles = [], []
    for t in ts:
        p, v, q = traj.sample(t)
        dists.append(np.linalg.norm(p - via_pos))
        speed = np.linalg.norm(v)
        if speed < 1e-4:
            angles.append(float("nan"))
            continue
        fwd = quat_rotate(q, FORWARD_AXIS)
        cos_a = max(-1.0, min(1.0, np.dot(fwd, v) / (np.linalg.norm(fwd) * speed)))
        angles.append(math.degrees(math.acos(cos_a)))
    via_idx = int(np.argmin(dists))
    lo = via_idx
    hi = min(len(angles), via_idx + int(window_s / dt))
    return float(np.nanmax(angles[lo:hi])), ts, angles, via_idx


def main():
    names = ["nav_entry", "inspection_entry_1", "capture_point_2"]
    locs = load_locations(names)
    q0 = locs["nav_entry"][1]
    waypoints = np.array([locs[n][0] for n in names])
    via_pos = locs["inspection_entry_1"][0]
    v0 = np.zeros(3)
    w0 = np.zeros(3)

    densified = MincoTrajectory._densify(waypoints, ATTITUDE_RESAMPLE_SPACING_M)
    print(f"waypoints={len(waypoints)} -> densified={len(densified)} points "
          f"(spacing={ATTITUDE_RESAMPLE_SPACING_M}m)")

    # --- Baseline (current production behavior) ---
    baseline_rotvecs = MincoTrajectory._waypoint_rotvecs(
        densified, q0, FORWARD_AXIS, face_travel=True
    )
    baseline_traj = SolvedTrajectory(
        waypoints_flat_from(densified, baseline_rotvecs), q0, v0, w0
    )
    base_max, _, _, _ = max_angle_near_via(baseline_traj, via_pos)
    print(f"\n[Baseline, straight-line rotvec seeding] "
          f"duration={baseline_traj.duration:.2f}s max_angle_near_via={base_max:.2f} deg")

    # --- Candidate A: two-pass iterative (reseed from pass-1's own solved v(t)) ---
    pass1 = baseline_traj  # pass 1 == baseline
    actual_tangents = [pass1.sample(t)[1] for t in pass1._cum_times]
    candA_rotvecs = rotvecs_from_tangents(actual_tangents, q0)
    candA_traj = SolvedTrajectory(
        waypoints_flat_from(densified, candA_rotvecs), q0, v0, w0
    )
    a_max, _, _, _ = max_angle_near_via(candA_traj, via_pos)
    print(f"[Candidate A, two-pass reseed from pass-1 solved v(t)] "
          f"duration={candA_traj.duration:.2f}s max_angle_near_via={a_max:.2f} deg")

    # --- Candidate B: Hermite pre-seed (tangent from a Hermite spline through the
    #     sparse waypoints, sampled at the same densified locations) ---
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    seg_times = np.where(distances < 1e-9, 1.0, distances)
    hermite_coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, seg_times)
    # map each densified point to its (segment, tau) on the ORIGINAL sparse waypoints'
    # arc-length parametrization (densify() only splits within a segment, proportionally)
    hermite_tangents = [np.zeros(3)]
    seg_cum = np.concatenate([[0.0], np.cumsum(seg_times)])
    for i in range(1, len(densified)):
        # find which original segment this densified point falls in by nearest match
        # on cumulative straight-line arc-length fraction
        d_from_start = np.linalg.norm(densified[i] - waypoints[0])
        # simpler robust approach: locate via nearest original segment bracketing it
        best_seg, best_tau = 0, 0.0
        best_err = None
        for s in range(len(seg_times)):
            p_a, p_b = waypoints[s], waypoints[s + 1]
            seg_len = np.linalg.norm(p_b - p_a)
            if seg_len < 1e-9:
                continue
            frac = np.dot(densified[i] - p_a, (p_b - p_a) / seg_len) / seg_len
            frac = max(0.0, min(1.0, frac))
            proj = p_a + frac * (p_b - p_a)
            err = np.linalg.norm(proj - densified[i])
            if best_err is None or err < best_err:
                best_err, best_seg, best_tau = err, s, frac * seg_times[s]
        v = evaluate_vector(hermite_coeffs[best_seg], best_tau, order=1)
        hermite_tangents.append(v)
    candB_rotvecs = rotvecs_from_tangents(hermite_tangents, q0)
    candB_traj = SolvedTrajectory(
        waypoints_flat_from(densified, candB_rotvecs), q0, v0, w0
    )
    b_max, _, _, _ = max_angle_near_via(candB_traj, via_pos)
    print(f"[Candidate B, Hermite pre-seed tangent] "
          f"duration={candB_traj.duration:.2f}s max_angle_near_via={b_max:.2f} deg")

    print("\nsummary (max |fwd_des<->v_des| angle within 15s of via passage):")
    print(f"  baseline (current production) : {base_max:6.2f} deg")
    print(f"  candidate A (two-pass reseed)  : {a_max:6.2f} deg")
    print(f"  candidate B (Hermite pre-seed) : {b_max:6.2f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
