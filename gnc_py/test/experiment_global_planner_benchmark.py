#!/usr/bin/env python3
"""Offline synthetic comparison of 6/26-connected A* and occupancy-grid RRT.

No ROS node, action, simulator, or production guidance code is used. Run from
the source tree after building the Python extension:

  PYTHONPATH=gnc_py python3 gnc_py/test/experiment_global_planner_benchmark.py
"""
import statistics
import time

import numpy as np

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.rrt_planner import RRTPlanner
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import shortcut_path


RES = 0.1
BOUNDS = ([0.0, -1.0, -0.5], [3.0, 1.0, 0.5])
START, GOAL = np.array([0.05, 0.05, 0.05]), np.array([2.95, 0.05, 0.05])


def make_grid():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.1)
    grid.add_box([1.5, 0.0, 0.0], [0.15, 0.35, 0.45])
    return grid


def path_length(path):
    return sum(float(np.linalg.norm(b - a)) for a, b in zip(path[:-1], path[1:]))


def measure(name, factory, trials):
    samples = []
    for seed in range(trials):
        planner = factory(seed)
        t0 = time.perf_counter()
        try:
            path = planner.plan(START, GOAL)
            count = getattr(planner, 'last_expansions', getattr(planner, 'last_iterations', 0))
            short = shortcut_path(path, planner._grid, BOUNDS)
            samples.append((True, (time.perf_counter() - t0) * 1000, path_length(path),
                            path_length(short), len(path), len(short), count))
        except RuntimeError:
            samples.append((False, (time.perf_counter() - t0) * 1000, float('nan'), 0))
    successes = [row for row in samples if row[0]]
    if not successes:
        return name, 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    return (name, len(successes), statistics.median(row[1] for row in successes),
            statistics.median(row[2] for row in successes),
            statistics.median(row[3] for row in successes),
            statistics.median(row[4] for row in successes),
            statistics.median(row[5] for row in successes),
            statistics.median(row[6] for row in successes))


def main():
    trials = 10
    rows = [
        measure('A* 6', lambda _seed: AStarPlanner(RES, grid=make_grid().snapshot(),
                search_bounds=BOUNDS, connectivity=6), trials),
        measure('A* 26 supercover', lambda _seed: AStarPlanner(RES, grid=make_grid().snapshot(),
                search_bounds=BOUNDS, connectivity=26), trials),
        measure('RRT', lambda seed: RRTPlanner(grid=make_grid().snapshot(), search_bounds=BOUNDS,
                step_size=0.2, goal_tolerance=0.2, goal_bias=0.3, max_iterations=5000, seed=seed), trials),
    ]
    print('planner                 success/%d  median ms  raw/short m  raw/short points  median expansions/iterations' % trials)
    for name, success, elapsed, raw_length, short_length, raw_points, short_points, count in rows:
        print('%-23s %3d/%-3d    %9.3f  %5.3f/%-5.3f  %5.0f/%-5.0f  %9.0f' %
              (name, success, trials, elapsed, raw_length, short_length,
               raw_points, short_points, count))


if __name__ == '__main__':
    main()
