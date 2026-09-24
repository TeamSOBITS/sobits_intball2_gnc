"""JEM route with the real walls (maps/jem_octomap.bt) plus a person box: the production
ReplanningMincoV3Tracker, ideal tracking, synchronous replans
(docs/2026-09-24_obstacle_avoidance_jem_map_check.md).
"""
import importlib.util
import os
import sys

import numpy as np
import yaml
from scipy.spatial import cKDTree

import minco_native_py
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import (
    ReplanningMincoV3Tracker,
)

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("stage4", os.path.join(_here, "experiment_obstacle_stage4_tracker.py"))
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)

MAP = os.path.join(_here, "..", "maps", "jem_octomap.bt")
LOCATIONS = yaml.safe_load(open(os.path.join(_here, "..", "maps", "iss_location.yaml")))["location_pose"]
PERSON_ACROSS_Y = np.array([0.25, 0.15, 0.85])  # width along x, depth along the +y travel
FLOOR_Z = 4.05

SCENARIOS = [
    ("walls only", []),
    ("person centered", [([10.95, -6.6, FLOOR_Z + 0.85], PERSON_ACROSS_Y)]),
    ("person x+0.15", [([11.10, -6.6, FLOOR_Z + 0.85], PERSON_ACROSS_Y)]),
]


def location(name):
    t = LOCATIONS[name]["translation"]
    return np.array([t["x"], t["y"], t["z"]])


def facing_quat(direction):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    axis = np.cross(stage4.FWD, d)
    s, c = np.linalg.norm(axis), float(stage4.FWD @ d)
    ang = np.arctan2(s, c)
    return np.concatenate([axis / s * np.sin(ang / 2), [np.cos(ang / 2)]])


def run(start, goal, boxes, wall_points, wall_tree):
    grid = minco_native_py.OccupancyGrid(stage4.GRID_RES, stage4.GRID_INFLATION)
    grid.add_points(wall_points)
    for c, h in boxes:
        grid.add_box(list(c), list(h))
    q0 = facing_quat(goal - start)
    state = {"p": start.copy(), "s": 0.0}
    tr = ReplanningMincoV3Tracker(
        start, goal, lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
        stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
        attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON,
        face_travel=True, forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE,
        local_piece_length_m=stage4.PIECE_LENGTH, obstacle_grid=grid,
        obstacle_clearance_soft=stage4.CLEARANCE_SOFT)
    solves = [tr.last_replan_solve_seconds]
    fails, box_clr, wall_clr, off, face, t = 0, np.inf, np.inf, 0.0, 0.0, 0.0
    while t <= tr.total_duration and t < 400.0:
        t += stage4.DT
        p, v, _a, q = tr.sample(t)
        state["p"], state["s"] = p, t
        solves += [tr.last_replan_solve_seconds] if tr.last_replan_occurred else []
        fails += int(tr.last_local_fallback)
        for c, h in boxes:
            box_clr = min(box_clr, stage4.box_distance(p, np.asarray(c), h) - stage4.ROBOT_RADIUS)
        # Voxel centers: subtract half a voxel for the surface.
        wall_clr = min(wall_clr, wall_tree.query(p)[0] - 0.025 - stage4.ROBOT_RADIUS)
        off = max(off, np.linalg.norm(np.cross(p - start, (goal - start) / np.linalg.norm(goal - start))))
        speed = float(np.linalg.norm(v))
        if speed > 0.05:
            f = quat_rotate(q, stage4.FWD)
            face = max(face, np.degrees(np.arccos(np.clip(f @ v / speed, -1, 1))))
    return dict(t=t, fails=fails, box=box_clr, wall=wall_clr, off=off, face=face,
                end=np.linalg.norm(state["p"] - goal), solve_max=max(solves))


def main():
    for a in sys.argv[1:]:
        if a.startswith("--inflation="):
            stage4.GRID_INFLATION = float(a.split("=", 1)[1])
    print("grid inflation %.2f m" % stage4.GRID_INFLATION)
    _res, flat = minco_native_py.load_octomap_points(MAP)
    wall_tree = cKDTree(np.asarray(flat).reshape(-1, 3))
    start, goal = location("inspection_entry_1"), location("nav_entry")
    print("inspection_entry_1 -> nav_entry, cruise %.2f, soft %.1f; clearances = robot surface [m]"
          % (stage4.CRUISE, stage4.CLEARANCE_SOFT))
    for name, boxes in SCENARIOS:
        r = run(start, goal, boxes, flat, wall_tree)
        print("%-16s t=%6.1fs fails=%2d box=%7.3f wall=%6.3f off_line=%.3f face=%5.1f "
              "goal_err=%.3f solve_max=%.3fs" % (name, r["t"], r["fails"], r["box"], r["wall"],
                                                r["off"], r["face"], r["end"], r["solve_max"]),
              flush=True)


if __name__ == "__main__":
    main()
