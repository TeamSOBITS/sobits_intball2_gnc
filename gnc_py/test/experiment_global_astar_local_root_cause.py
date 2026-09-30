#!/usr/bin/env python3
"""Offline root-cause probe for initial local MINCO failures.

The same A* route and person placement are used with/without the local
obstacle grid.  The planner's first look-ahead target and shape seed are
recorded, together with occupancy of the supplied local waypoints.
"""
import os
import sys
import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import densify_polyline, select_first_safe
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

Q0 = np.array([0., 0., 0., 1.])
FWD = np.array([1., 0., 0.])


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,spacing,face,grid,horizon,route_points,result,error")
    for name, boxes in (
            ("person_center", [([10.95, -6.60, 4.90], base.PERSON_HALF)]),
            ("person_x_plus_015", [([11.10, -6.60, 4.90], base.PERSON_HALF)])):
        grid = base.make_grid(static, boxes)
        route = base.shortcut_path(base.AStarPlanner(base.RESOLUTION, grid=grid,
                                      search_bounds=base.BOUNDS, connectivity=6).plan(start, goal), grid, base.BOUNDS)
        def build(points):
            return MincoTrajectory(points, Q0, face_travel=True, forward_axis=FWD,
                                   body_frame_wrench=True, via_half_width=0.,
                                   target_speed=base.TARGET_SPEED, max_accel=base.MAX_ACCEL)
        selected = select_first_safe(route, grid, base.BOUNDS, build)
        for spacing in (None, 0.5, 0.25, 0.1):
            route_points = densify_polyline(route, spacing)[1:-1]
            for face in (False, True):
              for use_grid in (False, True):
               for horizon in (2.0, 3.0, 4.0):
                state = {"p": start.copy(), "t": 0.0}
                captured = {}
                tracker = None
                try:
                    tracker = ReplanMincoTracker(
                        start, goal, lambda: (state["p"], list(Q0), state["t"]),
                        lambda _s: True, Q0, base.TARGET_SPEED, base.MAX_ACCEL,
                        route_waypoints=route_points, via_half_width=0., face_travel=face,
                        forward_axis=FWD, local_max_vel=.15, local_piece_length_m=1.5,
                        obstacle_grid=grid if use_grid else None, obstacle_clearance_soft=.2,
                        local_replan_period=1.0, planning_horizon_m=horizon)
                    captured["result"] = "OK"
                    captured["error"] = "-"
                except Exception as error:
                    captured["result"] = "FAIL"
                    captured["error"] = type(error).__name__
                    tracker = locals().get("tracker")
                if tracker is not None:
                    planner = tracker._planner
                    print("%s,%s,%s,%s,%.1f,%d,%s,%s" %
                          (name, spacing, face, "grid" if use_grid else "none", horizon,
                           len(route_points),
                           captured["result"], captured["error"]), flush=True)
                else:
                    print("%s,%s,%s,%s,%.1f,%d,%s,%s" %
                          (name, spacing, face, "grid" if use_grid else "none", horizon, len(route_points),
                           captured["result"], captured["error"]), flush=True)


if __name__ == "__main__":
    main()
