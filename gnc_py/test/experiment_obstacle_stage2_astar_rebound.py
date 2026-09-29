"""Obstacle avoidance stage 2 (docs/archive/achieved/2026-09-24_obstacle_avoidance_local_cost_plan.md):
the rebound pairs come from the C++ grid map, A* and fine collision check
(sobits_intball2_gnc_cpp.rebound_pairs, after Zhou et al., RA-L 2021), applied in a
Python loop around plan_minco on the shared local replica (experiment_obstacle_common.py).
"""
import importlib.util
import math
import os
import time

import numpy as np

import sobits_intball2_gnc_cpp

_spec = importlib.util.spec_from_file_location(
    "common", os.path.join(os.path.dirname(__file__), "experiment_obstacle_common.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

CRUISE = 0.15
CLEARANCE_SOFT = 0.2
REBOUND_ROUNDS = 3


def person_grid(center):
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(common.GRID_RES, common.ROBOT_RADIUS)
    grid.add_box(list(center), list(common.PERSON_HALF))
    return grid


def run_astar(v_start, center):
    center = np.asarray(center, float)
    grid = person_grid(center)
    p0, target = np.zeros(3), np.array([common.HORIZON, 0.0, 0.0])
    v0, v_tail = np.array([v_start, 0.0, 0.0]), np.array([common.LOCAL_MAX_VEL, 0.0, 0.0])
    n = max(2, math.ceil(common.HORIZON / common.PIECE_LENGTH))
    seed = np.array([p0 + (target - p0) * i / n for i in range(n + 1)])
    dirs = np.vstack([np.zeros(3), np.tile(target - p0, (n, 1))])
    T0 = [common.HORIZON / common.LOCAL_MAX_VEL / n] * n

    _ok, shape, _s = common.plan(seed, dirs, v0, v_tail, T0, math.inf)
    pairs_flat, rounds = [], []
    for _ in range(REBOUND_ROUNDS):
        t0 = time.perf_counter()
        status, pairs_flat = sobits_intball2_gnc_cpp.rebound_pairs(
            grid, list(shape.T), list(shape.C.ravel()), common.LOCAL_MAX_VEL,
            obstacle_pairs=pairs_flat)
        rebound_s = time.perf_counter() - t0
        if status != 1:
            rounds.append(("free" if status == 0 else "ERROR", None, rebound_s, None))
            break
        pairs = {}
        for k in range(0, len(pairs_flat), 7):
            pid = int(pairs_flat[k])
            pairs.setdefault(pid, []).append((np.array(pairs_flat[k + 1:k + 4]),
                                              np.array(pairs_flat[k + 4:k + 7])))
        ok, shape, solve_s = common.plan(seed, dirs, v0, v_tail, shape.T, math.inf,
                                         warm_qvia=shape.vias(), pairs=pairs)
        rounds.append((ok, common.clearance(lambda t: shape.pos_vel(t)[0], shape.cum[-1], center),
                       rebound_s, solve_s))
    side = float(np.sign(min((shape.pos_vel(t)[0][1] for t in np.linspace(0, shape.cum[-1], 200)),
                             key=lambda y: -abs(y))))
    traj, resolve_s = common.attitude_resolve(shape, v0, v_tail)
    final_clear = common.clearance(lambda t: traj.sample(t)[0], traj.global_total_duration, center)
    lat = common.lateral_max(lambda t: traj.sample(t)[0], traj.global_total_duration)
    return side, len(pairs_flat) // 7, rounds, (final_clear, lat, resolve_s, traj.global_total_duration)


def main():
    common.LOCAL_MAX_VEL = CRUISE
    common.CLEARANCE_SOFT = CLEARANCE_SOFT
    print("cruise %.2f m/s, clearance_soft %.1f m; clearance = robot surface to box [m]"
          % (CRUISE, CLEARANCE_SOFT))
    for name, v_start, center in common.SCENARIOS:
        v_start = CRUISE if v_start > 0.0 else 0.0
        side, n_pairs, rounds, (clr, lat, resolve_s, dur) = run_astar(v_start, center)
        print("== %s" % name)
        for k, (ok, rclr, rs, ss) in enumerate(rounds):
            if rclr is None:
                print("   A* round %d: %s (check %.3fs)" % (k + 1, ok, rs))
            else:
                print("   A* round %d: ok=%s clearance=%.3f check+A* %.3fs solve %.3fs"
                      % (k + 1, ok, rclr, rs, ss))
        print("   A* final: detour %s pairs %d clearance=%.3f lateral_max=%.3f attitude %.3fs dur %.1fs"
              % ("+y" if side > 0 else "-y", n_pairs, clr, lat, resolve_s, dur), flush=True)


if __name__ == "__main__":
    main()
