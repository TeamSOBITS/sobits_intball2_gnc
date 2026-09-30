#!/usr/bin/env python3
"""Synthetic depth-layer snapshot check for test-only global A*6."""
import numpy as np

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner


RES = 0.1
BOUNDS = ([0.0, -0.5, -0.5], [3.5, 0.5, 0.5])
START, GOAL = [0.05, 0.05, 0.05], [2.95, 0.05, 0.05]
# Optical z points along grid +x.
LOOK_X = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
ORIGIN = [0.05, 0.05, 0.05]


def plan(grid):
    return AStarPlanner(RES, grid=grid.snapshot(), search_bounds=BOUNDS, connectivity=6).plan(START, GOAL)


def main():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.1)
    grid.enable_depth_layer(*BOUNDS)
    clear = plan(grid)
    for _ in range(6):
        grid.integrate_depth(np.array([[1.5]], np.float32), 1.0, 1.0, 0.0, 0.0, LOOK_X, ORIGIN)
    blocked = plan(grid)
    for _ in range(6):
        grid.integrate_depth(np.array([[np.inf]], np.float32), 1.0, 1.0, 0.0, 0.0, LOOK_X, ORIGIN)
    restored = plan(grid)
    print("clear_points=%d blocked_points=%d restored_points=%d depth_cells=%d" %
          (len(clear), len(blocked), len(restored), len(grid.depth_occupied_cells())))
    assert len(blocked) > len(clear) and len(restored) == len(clear)


if __name__ == "__main__":
    main()
