#!/usr/bin/env python3
"""Offline A*6 -> selected global MINCO -> ReplanMincoTracker integration check."""
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import densify_polyline, select_first_safe
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


CASES = [
    ("person_center", [([10.95, -6.60, 4.90], base.PERSON_HALF)]),
    ("person_x_plus_015", [([11.10, -6.60, 4.90], base.PERSON_HALF)]),
    ("two_people", [([10.70, -6.60, 4.90], base.PERSON_HALF),
                    ([11.25, -7.25, 4.90], base.PERSON_HALF)]),
]
Q0, FWD = np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])


def curve_free(trajectory, grid):
    return all(not grid.inflated_occupied(list(trajectory.sample(t)[0]))
               for t in np.arange(0., trajectory.global_total_duration + .01, .01))


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,selected,global_free,local_free,replans,goal_error")
    for name, boxes in CASES:
        grid = base.make_grid(static, boxes)
        route = base.shortcut_path(base.AStarPlanner(base.RESOLUTION, grid=grid,
                                      search_bounds=base.BOUNDS, connectivity=6).plan(start, goal), grid, base.BOUNDS)
        def build(points):
            return MincoTrajectory(points, Q0, face_travel=True, forward_axis=FWD,
                                   body_frame_wrench=True, via_half_width=0.,
                                   target_speed=base.TARGET_SPEED, max_accel=base.MAX_ACCEL)
        selected = select_first_safe(route, grid, base.BOUNDS, build)
        assert selected["ok"], name
        points = selected["selected"]["trajectory"].num_waypoints
        state = {"p": start.copy(), "t": 0.0}
        try:
            tracker = ReplanMincoTracker(
                start, goal, lambda: (state["p"], list(Q0), state["t"]), lambda _s: True, Q0,
                base.TARGET_SPEED, base.MAX_ACCEL,
                route_waypoints=densify_polyline(route, selected["selected"]["spacing_m"])[1:-1],
                via_half_width=0., face_travel=True, forward_axis=FWD, local_max_vel=.15,
                local_piece_length_m=1.5, obstacle_grid=grid, obstacle_clearance_soft=.2,
                local_replan_period=1.0, planning_horizon_m=4.0)
        except Exception as error:
            print("%s,%d,True,False,0,CONSTRUCT:%s" % (name, points, type(error).__name__), flush=True)
            continue
        local_free, replans, t = True, 0, 0.0
        while t <= tracker.trajectory.global_total_duration and t < 260.0:
            t += .05
            p, _v, _a, _q = tracker.sample(t)
            state["p"], state["t"] = p, t
            local_free &= not grid.inflated_occupied(list(p))
            replans += int(tracker.last_replan_occurred)
        print("%s,%d,%s,%s,%d,%.3f" % (name, points, curve_free(tracker.trajectory, grid),
              local_free, replans, np.linalg.norm(state["p"] - goal)), flush=True)


if __name__ == "__main__":
    main()
