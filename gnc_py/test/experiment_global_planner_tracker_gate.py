#!/usr/bin/env python3
"""Offline gate comparison: global MINCO safety plus initial local solve."""
import argparse
import os, sys
import numpy as np
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import densify_polyline, curve_is_free
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

Q0 = np.array([0., 0., 0., 1.]); FWD = np.array([1., 0., 0.])

def gate(path, grid, start, goal, horizon_m):
    failures = []
    for spacing in (0.50, 0.25, 0.10):
        points = densify_polyline(path, spacing)[1:-1]
        state = {"p": start.copy(), "t": 0.0}
        try:
            tracker = ReplanMincoTracker(
                start, goal, lambda: (state["p"], list(Q0), state["t"]),
                lambda _s: True, Q0, base.TARGET_SPEED, base.MAX_ACCEL,
                route_waypoints=points, via_half_width=0., face_travel=True,
                forward_axis=FWD, local_max_vel=.15, local_piece_length_m=1.5,
                obstacle_grid=grid, obstacle_clearance_soft=.2,
                local_replan_period=1.0, planning_horizon_m=horizon_m)
            check = curve_is_free(tracker.trajectory, grid, base.BOUNDS)
            local_check = curve_is_free(tracker.local_trajectory, grid, base.BOUNDS)
            if check["ok"] and local_check["ok"] and not tracker.initial_local_collides:
                return spacing, True, failures
            failures.append("global_or_initial_local_collision")
        except Exception as error:
            failures.append(type(error).__name__)
    return None, False, failures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--planner", choices=("astar6", "all"), default="all")
    args = parser.parse_args()
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,planner,seed,search,initial_local_gate,spacing,horizon_m")
    for x in (10.65, 10.95, 11.25):
      for y in (-7.20, -6.60, -6.00):
        grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
        planners = [("astar6", [0], lambda s: base.AStarPlanner(base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS, connectivity=6))]
        if args.planner == "all":
            planners += [("rrt", [0, 1, 2], lambda s: base.RRTPlanner(grid=grid, search_bounds=base.BOUNDS, step_size=.20, goal_tolerance=.20, goal_bias=.30, max_iterations=5000, seed=s))]
        for name, seeds, factory in planners:
          for seed in seeds:
            try:
                path = factory(seed).plan(start, goal)
                path = base.shortcut_path(path, grid, base.BOUNDS)
                spacing, ok, failures = gate(path, grid, start, goal, args.horizon)
                print("x=%.2f y=%.2f,%s,%d,True,%s,%s,%s,%.1f" % (x, y, name, seed, ok, spacing, ";".join(failures), args.horizon), flush=True)
            except Exception as error:
                print("x=%.2f y=%.2f,%s,%d,False,False,None,%s,%.1f" % (x, y, name, seed, type(error).__name__, args.horizon), flush=True)

if __name__ == "__main__": main()
