#!/usr/bin/env python3
"""A*6 snapshot-at-goal comparison using the production pre-align seed."""
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free, densify_polyline
from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import shortcut_path
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

FWD = np.array([1., 0., 0.])
Q_RAW = np.array([0., 0., 0., 1.])


def construct(path, grid, start, goal, spacing):
    route = densify_polyline(path, spacing)
    # Same q passed to the tracker after GuidanceExecutor's first-leg pre-align.
    q0 = compute_q_des(route[1] - route[0], Q_RAW, 1e-9, FWD)
    state = {"p": start.copy(), "t": 0.0}
    tracker = ReplanMincoTracker(
        start, goal, lambda: (state["p"], list(q0), state["t"]),
        lambda _stamp: True, q0, base.TARGET_SPEED, base.MAX_ACCEL,
        route_waypoints=route[1:-1], via_half_width=0., face_travel=True,
        forward_axis=FWD, local_max_vel=base.LOCAL_MAX_VEL, local_piece_length_m=base.LOCAL_PIECE_LENGTH,
        obstacle_grid=grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
        local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
        async_replan=False)
    return tracker, state


def run_case(static, start, goal, person):
    grid = base.make_grid(static, [(person, base.PERSON_HALF)])
    path = shortcut_path(AStarPlanner(base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS,
                                      connectivity=6).plan(start, goal), grid, base.BOUNDS)
    errors = []
    for spacing in (.50, .25, .10):
        try:
            tracker, state = construct(path, grid, start, goal, spacing)
            if (not curve_is_free(tracker.trajectory, grid, base.BOUNDS)["ok"]
                    or not curve_is_free(tracker.local_trajectory, grid, base.BOUNDS)["ok"]
                    or tracker.initial_local_collides):
                errors.append("initial_collision")
                continue
            t, replans, fallback, output_free = 0., 0, 0, True
            while t < tracker.total_duration and t < 300.0:
                t += .05
                p, _v, _a, _q = tracker.sample(t)
                state["p"], state["t"] = p, t
                output_free &= not grid.inflated_occupied(list(p))
                replans += int(tracker.last_replan_occurred)
                fallback += int(tracker.last_local_fallback)
                if not output_free or tracker.replanning_stopped:
                    break
            error = float(np.linalg.norm(state["p"] - goal))
            complete = output_free and not tracker.replanning_stopped and fallback == 0 and error <= .10
            return spacing, complete, replans, fallback, error, "-"
        except Exception as exc:
            errors.append(type(exc).__name__)
    return None, False, 0, 0, np.nan, ";".join(errors)


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    print("# profile=%s target_speed=%.2f local=%.1fm/%.1fs/%.2fmps grid=%.2f/%.2f" %
          (base.OFFLINE_PROFILE, base.TARGET_SPEED, base.LOCAL_HORIZON,
           base.LOCAL_REPLAN_PERIOD, base.LOCAL_MAX_VEL, base.RESOLUTION, base.INFLATION))
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,spacing,complete,replans,fallbacks,goal_error_m,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            result = run_case(static, start, goal, np.array([x, y, 4.90]))
            print("x=%.2f y=%.2f,%s,%s,%d,%d,%.3f,%s" %
                  (x, y, *result), flush=True)


if __name__ == "__main__":
    main()
