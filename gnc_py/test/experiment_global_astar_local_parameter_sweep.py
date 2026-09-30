#!/usr/bin/env python3
"""Offline sweep of ReplanMincoTracker local settings after A* global planning."""
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import densify_polyline, select_first_safe
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


Q0, FWD = np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,horizon,piece,soft,selected,construct,initial_collides")
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
        points = densify_polyline(route, selected["selected"]["spacing_m"])[1:-1]
        for horizon in (2.0, 3.0, 4.0):
            for piece in (1.0, 1.5, 2.0):
                for soft in (0.1, 0.2, 0.4):
                    try:
                        state = {"p": start.copy(), "t": 0.0}
                        tracker = ReplanMincoTracker(
                            start, goal, lambda: (state["p"], list(Q0), state["t"]),
                            lambda _s: True, Q0, base.TARGET_SPEED, base.MAX_ACCEL,
                            route_waypoints=points, via_half_width=0., face_travel=True,
                            forward_axis=FWD, local_max_vel=.15,
                            local_piece_length_m=piece, obstacle_grid=grid,
                            obstacle_clearance_soft=soft, local_replan_period=1.0,
                            planning_horizon_m=horizon)
                        print("%s,%.1f,%.1f,%.1f,%s,OK,%s" %
                              (name, horizon, piece, soft,
                               selected["selected"]["spacing_m"], tracker.initial_local_collides), flush=True)
                    except Exception as error:
                        print("%s,%.1f,%.1f,%.1f,%s,%s,-" %
                              (name, horizon, piece, soft, selected["selected"]["spacing_m"],
                               type(error).__name__), flush=True)


if __name__ == "__main__":
    main()
