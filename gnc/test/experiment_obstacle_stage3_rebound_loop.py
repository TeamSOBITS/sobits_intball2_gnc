"""Obstacle avoidance stage 3 (docs/2026-09-24_obstacle_avoidance_local_cost_plan.md):
EGO-Planner v2's whole rebound loop inside plan_minco (grid argument): initial check,
in-optimization rebound, fine-check restarts. One call replaces stage 2's Python loop.
"""
import importlib.util
import math
import os
import time

import numpy as np

import minco_native_py

_here = os.path.dirname(__file__)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage2 = _load("stage2", "experiment_obstacle_stage2_astar_rebound.py")
stage1 = stage2.stage1

SCENARIOS = stage1.SCENARIOS + [
    ("cruise, head-on 1.25m", 0.2, [1.25, 0.0, 0.0]),
    ("cruise, two boxes gap 1.0m", 0.2, None),
]


def build_grid(center):
    grid = minco_native_py.OccupancyGrid(stage1.GRID_RES, stage1.ROBOT_RADIUS)
    boxes = [np.asarray(center, float)] if center is not None else [
        np.array([2.0, 0.75, 0.0]), np.array([2.0, -0.75, 0.0])]
    for c in boxes:
        grid.add_box(list(c), list(stage1.PERSON_HALF))
    return grid, boxes


def run_loop(v_start, center):
    grid, boxes = build_grid(center)
    p0, target = np.zeros(3), np.array([stage1.HORIZON, 0.0, 0.0])
    v0, v_tail = np.array([v_start, 0.0, 0.0]), np.array([stage1.LOCAL_MAX_VEL, 0.0, 0.0])
    n = max(2, math.ceil(stage1.HORIZON / stage1.PIECE_LENGTH))
    seed = np.array([p0 + (target - p0) * i / n for i in range(n + 1)])
    dirs = np.vstack([np.zeros(3), np.tile(target - p0, (n, 1))])
    rotvecs = stage1.face_rotvecs(dirs, np.zeros(3))
    wf = [float(c) for p, rv in zip(seed, rotvecs) for c in (*p, *rv)]
    t0 = time.perf_counter()
    ok, ec, T, C, _d = minco_native_py.plan_minco(
        wf, list(v0), [0.0, 0.0, 0.0], math.inf, stage1.MARGIN,
        warm_start_T=[stage1.HORIZON / stage1.LOCAL_MAX_VEL / n] * n, a0=[0.0, 0.0, 0.0],
        v_tail=list(v_tail), max_vel=stage1.LOCAL_MAX_VEL, q0=list(stage1.Q0),
        obstacle_clearance_soft=stage1.CLEARANCE_SOFT, grid=grid)
    solve_s = time.perf_counter() - t0
    shape = stage1.Piecewise(T, C)
    traj, resolve_s = stage1.attitude_resolve(shape, v0, v_tail)
    clr = min(stage1.clearance(lambda t: traj.sample(t)[0], traj.global_total_duration, b)
              for b in boxes)
    lat = stage1.lateral_max(lambda t: traj.sample(t)[0], traj.global_total_duration)
    return ok, ec, clr, lat, solve_s, resolve_s, traj.global_total_duration


def main():
    stage1.LOCAL_MAX_VEL = stage2.CRUISE
    stage1.CLEARANCE_SOFT = stage2.CLEARANCE_SOFT
    print("cruise %.2f m/s, clearance_soft %.1f m; error_code 0 ok / 1 wrench / 2 collision"
          % (stage2.CRUISE, stage2.CLEARANCE_SOFT))
    for name, v_start, center in SCENARIOS:
        v_start = stage2.CRUISE if v_start > 0.0 else 0.0
        ok, ec, clr, lat, s, rs, dur = run_loop(v_start, center)
        line = "%-28s ok=%s ec=%d clearance=%.3f lateral_max=%.3f shape %.3fs attitude %.3fs dur %.1fs" % (
            name, ok, ec, clr, lat, s, rs, dur)
        if center is not None:
            _side, _n, _rounds, (s2clr, s2lat, _rs, _d) = stage2.run_astar(v_start, center)
            line += " | stage2 clearance=%.3f lateral=%.3f" % (s2clr, s2lat)
        print(line, flush=True)


if __name__ == "__main__":
    main()
