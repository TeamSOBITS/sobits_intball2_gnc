#!/usr/bin/env python3
"""Separate global and first-local safety for current no-A* and A* routes."""
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free, densify_polyline
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

Q0, FWD = np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])


def trial(grid, start, goal, route_waypoints):
    state = {"p": start.copy(), "t": 0.0}
    try:
        tracker = ReplanMincoTracker(
            start, goal, lambda: (state["p"], list(Q0), state["t"]),
            lambda _stamp: True, Q0, base.TARGET_SPEED, base.MAX_ACCEL,
            route_waypoints=route_waypoints, via_half_width=0., face_travel=True,
            forward_axis=FWD, local_max_vel=base.LOCAL_MAX_VEL, local_piece_length_m=base.LOCAL_PIECE_LENGTH,
            obstacle_grid=grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
            local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
            async_replan=False)
        return (curve_is_free(tracker.trajectory, grid, base.BOUNDS)["ok"],
                curve_is_free(tracker.local_trajectory, grid, base.BOUNDS)["ok"],
                tracker.initial_local_collides, "-")
    except Exception as error:
        return (False, False, False, type(error).__name__)


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    print("# profile=%s target_speed=%.2f local=%.1fm/%.1fs/%.2fmps grid=%.2f/%.2f" %
          (base.OFFLINE_PROFILE, base.TARGET_SPEED, base.LOCAL_HORIZON,
           base.LOCAL_REPLAN_PERIOD, base.LOCAL_MAX_VEL, base.RESOLUTION, base.INFLATION))
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,mode,global_free,initial_local_free,initial_local_collides,error")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
            label = "x=%.2f y=%.2f" % (x, y)
            print("%s,current_no_astar,%s,%s,%s,%s" %
                  ((label,) + trial(grid, start, goal, None)), flush=True)
            route = base.shortcut_path(base.AStarPlanner(
                base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS,
                connectivity=6).plan(start, goal), grid, base.BOUNDS)
            results = []
            for spacing in (.50, .25, .10):
                results.append((spacing,) + trial(
                    grid, start, goal, densify_polyline(route, spacing)[1:-1]))
            good = next((row for row in results if row[1] and row[2]), results[-1])
            print("%s,astar6_%.2f,%s,%s,%s,%s" % ((label,) + good), flush=True)


if __name__ == "__main__":
    main()
