"""Obstacle avoidance stage 4 (docs/2026-09-24_obstacle_avoidance_local_cost_plan.md):
the production ReplanMincoTracker with an obstacle grid, replanning every 1 s
under ideal tracking along a straight 6 m route with virtual person boxes.
"""

import numpy as np

import minco_native_py
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

MARGIN = 0.7
SPACING = 0.3
TS = 0.5
MA = 0.0996 / 3.216
FWD = np.array([1.0, 0.0, 0.0])
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
HORIZON = 4.0
CRUISE = 0.15
PIECE_LENGTH = 1.5
CLEARANCE_SOFT = 0.2
ROBOT_RADIUS = 0.1
GRID_RES = 0.1
GRID_INFLATION = ROBOT_RADIUS
DT = 0.05
PERSON = np.array([0.15, 0.25, 0.85])
GOAL = np.array([6.0, 0.0, 0.0])

SCENARIOS = [
    ("no obstacle", []),
    ("head-on 3.0m", [([3.0, 0.0, 0.0], PERSON)]),
    ("offset y+0.15 3.0m", [([3.0, 0.15, 0.0], PERSON)]),
    ("two boxes gap 1.0m", [([3.0, 0.75, 0.0], PERSON), ([3.0, -0.75, 0.0], PERSON)]),
    ("wall 1.6m wide, y+0.3", [([3.0, 0.3, 0.0], np.array([0.15, 0.8, 0.85]))]),
    ("just before goal 5.3m", [([5.3, 0.0, 0.0], PERSON)]),
    ("on the goal 6.0m", [([6.0, 0.0, 0.0], PERSON)]),
]


def box_distance(p, center, half):
    q = np.abs(p - center) - half
    return np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)


def run(boxes):
    grid = minco_native_py.OccupancyGrid(GRID_RES, GRID_INFLATION)
    for c, h in boxes:
        grid.add_box(list(c), list(h))
    state = {"p": np.zeros(3), "s": 0.0}
    tr = ReplanMincoTracker(
        np.zeros(3), GOAL, lambda: (state["p"], list(Q0), state["s"]), lambda s: True, Q0,
        TS, MA, via_half_width=0.0, wrench_safety_margin=MARGIN,
        attitude_resample_spacing_m=SPACING, planning_horizon_m=HORIZON, face_travel=True,
        forward_axis=FWD, local_max_vel=CRUISE, local_piece_length_m=PIECE_LENGTH,
        obstacle_grid=grid, obstacle_clearance_soft=CLEARANCE_SOFT)
    solves = [tr.last_replan_solve_seconds]
    fails, clr, lat, face, moved, t = 0, np.inf, 0.0, 0.0, False, 0.0
    while t <= tr.total_duration and t < 400.0:
        t += DT
        p, v, _a, q = tr.sample(t)
        state["p"], state["s"] = p, t
        solves += [tr.last_replan_solve_seconds] if tr.last_replan_occurred else []
        fails += int(tr.last_local_fallback)
        moved |= tr.last_goal_moved_out_of_obstacle
        for c, h in boxes:
            clr = min(clr, box_distance(p, np.asarray(c), h) - ROBOT_RADIUS)
        lat = max(lat, abs(p[1]))
        speed = float(np.linalg.norm(v))
        if speed > 0.05:
            f = quat_rotate(q, FWD)
            face = max(face, np.degrees(np.arccos(np.clip(f @ v / speed, -1, 1))))
    return dict(t=t, fails=fails, clr=clr, lat=lat, face=face, moved=moved, end=state["p"],
                solve_max=max(solves), n=len(solves))


def main():
    print("cruise %.2f, soft %.1f; clearance = robot surface to box [m], setpoint path" % (
        CRUISE, CLEARANCE_SOFT))
    for name, boxes in SCENARIOS:
        r = run(boxes)
        print("%-24s t=%6.1fs replans=%3d fails=%3d clearance=%7.3f lateral=%.3f face=%5.1f "
              "solve_max=%.3fs goal_moved=%s end=%s" % (
                  name, r["t"], r["n"], r["fails"], r["clr"], r["lat"], r["face"],
                  r["solve_max"], r["moved"], np.round(r["end"], 3)), flush=True)


if __name__ == "__main__":
    main()
