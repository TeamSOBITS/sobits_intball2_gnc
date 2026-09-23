#!/usr/bin/env python3
"""Offline (no ROS, no sim) reproduction of the "does the *global* MincoTrajectory's
own face_travel attitude reference actually point along its own commanded velocity
through a corner?" question -- no live sim run, no production code changes.

Background: a live-sim `capture_point_2` move (`docs/archive/achieved/
2026-09-20_replanning_minco_v2_goal_arrival_oscillation_root_cause_and_fix.md`'s
`fix_verify` trace) showed the *setpoint itself* (not the TF tracking error) pointing
42-60 degrees away from its own commanded velocity direction for ~12s right after
passing the via waypoint `inspection_entry_1`, matching the unresolved task
`docs/issue.md` "[G] replanning_minco_v2でカーブ後に進行方向を向かずに進む".

That live trace can't tell whether the divergence originates in the *global*
MincoTrajectory (the same class `replanning_minco_v2_tracker.py`'s local layer draws
its lookahead target rotvec/velocity from every replan) or only appears once the
local receding-horizon quintic-Hermite layer is involved. This script builds the
same `MincoTrajectory` production class directly, with the real waypoints/params, and
checks the global trajectory alone -- isolating the two candidate root causes without
touching production code or running a sim.

Mechanism under test (`minco_trajectory.py::MincoTrajectory._waypoint_rotvecs` +
`_densify`): face_travel attitude waypoints are seeded from the *piecewise-linear*
path through the original sparse waypoints (each straight sub-segment locally
densified by `attitude_resample_spacing_m`), each pointing along that straight-line
direction -- computed *before* MINCO solves the actual (smooth, corner-rounding)
position polynomial. If the solved position path rounds the corner over some
distance/time (rather than a sharp kink exactly at the via waypoint), the rotation
polynomial's seed directions (based on the sharp-angle piecewise-linear path) and the
solved position polynomial's actual local tangent disagree throughout that
corner-rounding zone -- this script checks whether that is the case.

Usage:
    python3 test/manual/analyze_minco_face_travel_corner_offline.py
"""
import math
import sys

import numpy as np
import yaml

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

ISS_LOCATION_YAML = "/root/colcon_ws/src/sobits_intball2_gnc/gnc/maps/iss_location.yaml"

# Matches guidance.py's live computation (trajectory_controller.max_force/mass,
# gnc_params.yaml) and guidance.wrench_envelope_safety_margin used in the live
# fix_verify run.
MASS_KG = 3.216
MAX_FORCE_XYZ = (0.181, 0.0996, 0.122)
MAX_ACCEL = min(MAX_FORCE_XYZ) / MASS_KG
TARGET_SPEED = 0.5
WRENCH_SAFETY_MARGIN = 0.7
VIA_HALF_WIDTH = 0.0  # guidance.minco_via_half_width default
ATTITUDE_RESAMPLE_SPACING_M = 0.3  # guidance.minco_attitude_resample_spacing_m default
FORWARD_AXIS = (1.0, 0.0, 0.0)


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


def quat_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    # standard quat-vector rotation, same convention as control/utils/quat_math.py
    qv = np.array([x, y, z])
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2 * (w * uv + uuv)


def main():
    names = ["nav_entry", "inspection_entry_1", "capture_point_2"]
    locs = load_locations(names)
    q0 = locs["nav_entry"][1]
    waypoints = np.array([locs[n][0] for n in names])

    print("waypoints:")
    for n in names:
        print(f"  {n}: {locs[n][0]}")
    print(f"q0 (nav_entry orientation): {q0}")
    print(f"max_accel = min({MAX_FORCE_XYZ})/{MASS_KG} = {MAX_ACCEL:.5f} m/s^2")

    traj = MincoTrajectory(
        waypoints, q0,
        forward_axis=FORWARD_AXIS, face_travel=True,
        via_half_width=VIA_HALF_WIDTH,
        attitude_resample_spacing_m=ATTITUDE_RESAMPLE_SPACING_M,
        wrench_safety_margin=WRENCH_SAFETY_MARGIN,
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
    )
    duration = traj.global_total_duration
    print(f"solved duration={duration:.3f}s, num_waypoints={traj.num_waypoints}, "
          f"solve_wall_seconds={traj.solve_wall_seconds:.3f}")

    # Straight-line distance from inspection_entry_1 to bracket where along the
    # solved trajectory the via point is actually passed (closest approach).
    via_pos = locs["inspection_entry_1"][0]
    n_samples = int(duration / 0.02) + 1
    ts = np.linspace(0.0, duration, n_samples)
    dists = []
    angles = []
    speeds = []
    for t in ts:
        p, v, a, q = traj.sample(t)
        dists.append(np.linalg.norm(p - via_pos))
        speed = np.linalg.norm(v)
        speeds.append(speed)
        if speed < 1e-4:
            angles.append(float("nan"))
            continue
        fwd = quat_rotate(q, np.array(FORWARD_AXIS))
        cos_a = np.dot(fwd, v) / (np.linalg.norm(fwd) * speed)
        cos_a = max(-1.0, min(1.0, cos_a))
        angles.append(math.degrees(math.acos(cos_a)))

    via_idx = int(np.argmin(dists))
    via_t = ts[via_idx]
    print(f"\nclosest approach to inspection_entry_1: t={via_t:.3f}s, "
          f"dist={dists[via_idx]:.4f}m")

    print("\nt_offset_from_via[s]  fwd_des<->v_des angle[deg]  |v_des|[m/s]")
    dt = duration / (n_samples - 1)
    step = max(1, round(0.2 / dt))
    for i in range(0, len(ts), step):
        off = ts[i] - via_t
        if -3.0 <= off <= 15.0:
            print(f"  {off:+6.2f}  {angles[i]:8.2f}  {speeds[i]:6.4f}")

    max_ang = np.nanmax(angles[via_idx:via_idx + int(15.0 / (duration / n_samples))])
    print(f"\nmax fwd_des<->v_des angle within 15s after via passage: {max_ang:.2f} deg")
    if max_ang > 20.0:
        print("=> CONFIRMED: divergence already present in the GLOBAL MincoTrajectory "
              "alone (no local replanning layer involved) -- root cause is in "
              "_waypoint_rotvecs/_densify's straight-line seeding vs the solved "
              "corner-rounded position polynomial, not the local receding-horizon layer.")
    else:
        print("=> NOT reproduced in the global trajectory alone -- root cause is likely "
              "specific to replanning_minco_v2_tracker's local receding-horizon "
              "quintic-Hermite layer, not the global MincoTrajectory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
