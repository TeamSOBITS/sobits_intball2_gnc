"""Obstacle avoidance stage 0 (docs/2026-09-24_obstacle_avoidance_local_cost_plan.md):
multi-piece local shape solve with unboxed interior points, no obstacles.

Runs the production ReplanMincoTracker (face travel, production gains,
ideal tracking, synchronous replans) with local_piece_length_m=None (current
single-piece) vs EGO-Planner v2's 1.5 m pieces, and compares against it.
"""
import sys

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

MARGIN = 0.7
SPACING = 0.3
TS = 0.5
MA = 0.0996 / 3.216
FWD = np.array([1.0, 0.0, 0.0])
HORIZON = 4.0
LOCAL_MAX_VEL = 0.2
DT = 0.05
SPEED_MIN = 0.05


def yaw_quat(deg):
    h = np.radians(deg) / 2.0
    return np.array([0.0, 0.0, np.sin(h), np.cos(h)])


D = 6.0 / np.sqrt(3.0)
ROUTES = [
    ("straight 6m +x", [[0, 0, 0], [6, 0, 0]], yaw_quat(0.0)),
    ("straight 6m diagonal", [[0, 0, 0], [D, D, D]], None),
    ("S3 turn 90 (via)", [[0, 0, 0], [3, 0, 0], [3, 3, 0]], yaw_quat(0.0)),
]


def facing_quat(direction):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    axis = np.cross(FWD, d)
    s, c = np.linalg.norm(axis), float(FWD @ d)
    if s < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ang = np.arctan2(s, c)
    return np.concatenate([axis / s * np.sin(ang / 2), [np.cos(ang / 2)]])


def dist_to_polyline(p, wps):
    best = np.inf
    for a, b in zip(wps[:-1], wps[1:]):
        ab = b - a
        u = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + u * ab)))
    return best


def run(wps, q0, piece_length):
    wps = np.asarray(wps, float)
    state = {"p": wps[0].copy(), "s": 0.0}
    tr = ReplanMincoTracker(
        wps[0], wps[-1], lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
        TS, MA, route_waypoints=wps[1:-1], via_half_width=0.0,
        wrench_safety_margin=MARGIN, attitude_resample_spacing_m=SPACING,
        planning_horizon_m=HORIZON, face_travel=True, forward_axis=FWD,
        local_max_vel=LOCAL_MAX_VEL, local_piece_length_m=piece_length)
    solves = [tr.last_replan_solve_seconds]
    fails, dev, face, vmax, t = 0, 0.0, 0.0, 0.0, 0.0
    while t <= tr.total_duration and t < 600.0:
        t += DT
        p, v, _a, q = tr.sample(t)
        state["p"], state["s"] = p, t
        if tr.last_replan_occurred:
            solves.append(tr.last_replan_solve_seconds)
        fails += int(tr.last_local_fallback)
        dev = max(dev, dist_to_polyline(p, wps))
        speed = float(np.linalg.norm(v))
        vmax = max(vmax, speed)
        if speed > SPEED_MIN:
            fwd = quat_rotate(q, FWD)
            face = max(face, np.degrees(np.arccos(np.clip(fwd @ v / speed, -1, 1))))
    return {"t": t, "fails": fails, "dev_mm": dev * 1e3, "face_deg": face, "vmax": vmax,
            "goal_mm": np.linalg.norm(state["p"] - wps[-1]) * 1e3,
            "solve_med": np.median(solves), "solve_max": np.max(solves),
            "n_replan": len(solves)}


def main():
    piece_lengths = [None, 1.5] + [float(a) for a in sys.argv[1:]]
    print("%-22s %-5s %7s %5s %7s %7s %6s %7s %8s %8s %6s" % (
        "route", "piece", "t[s]", "fail", "dev[mm]", "face[d]", "vmax", "goal[mm]",
        "solveMed", "solveMax", "nRepl"))
    for name, wps, q0 in ROUTES:
        if q0 is None:
            q0 = facing_quat(np.asarray(wps[1]) - np.asarray(wps[0]))
        for pl in piece_lengths:
            r = run(wps, q0, pl)
            print("%-22s %-5s %7.1f %5d %7.1f %7.1f %6.3f %7.1f %8.3f %8.3f %6d" % (
                name, "K=1" if pl is None else pl, r["t"], r["fails"], r["dev_mm"],
                r["face_deg"], r["vmax"], r["goal_mm"], r["solve_med"], r["solve_max"],
                r["n_replan"]), flush=True)


if __name__ == "__main__":
    main()
