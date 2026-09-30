#!/usr/bin/env python3
"""Offline end-to-end gate for A*6, global MINCO, and local replanning."""
import argparse
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free, densify_polyline
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

Q0, FWD = np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])


def make_tracker(route, grid, start, goal, horizon_m):
    for spacing in (.50, .25, .10):
        state = {"p": start.copy(), "t": 0.0}
        try:
            tracker = ReplanMincoTracker(
                start, goal, lambda: (state["p"], list(Q0), state["t"]),
                lambda _stamp: True, Q0, base.TARGET_SPEED, base.MAX_ACCEL,
                route_waypoints=densify_polyline(route, spacing)[1:-1],
                via_half_width=0., face_travel=True, forward_axis=FWD,
                local_max_vel=.15, local_piece_length_m=1.5,
                obstacle_grid=grid, obstacle_clearance_soft=.2,
                local_replan_period=1.0, planning_horizon_m=horizon_m,
                async_replan=False)
            if (curve_is_free(tracker.trajectory, grid, base.BOUNDS)["ok"]
                    and curve_is_free(tracker.local_trajectory, grid, base.BOUNDS)["ok"]
                    and not tracker.initial_local_collides):
                return tracker, state, spacing
        except Exception:
            pass
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=float, required=True)
    args = parser.parse_args()
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,initial_gate,spacing,output_free,replans,fallback,goal_error_m,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
            route = base.shortcut_path(base.AStarPlanner(
                base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS,
                connectivity=6).plan(start, goal), grid, base.BOUNDS)
            tracker, state, spacing = make_tracker(route, grid, start, goal, args.horizon)
            label = "x=%.2f y=%.2f" % (x, y)
            if tracker is None:
                print("%s,False,None,-,0,-,-,initial_gate_failed" % label, flush=True)
                continue
            output_free, replans, fallback, t = True, 0, False, 0.0
            while t < tracker.total_duration and t < 300.0:
                t += .05
                p, _v, _a, _q = tracker.sample(t)
                state["p"], state["t"] = p, t
                output_free &= not grid.inflated_occupied(list(p))
                replans += int(tracker.last_replan_occurred)
                fallback |= tracker.last_local_fallback or tracker.replanning_stopped
                if fallback or not output_free:
                    break
            error = float(np.linalg.norm(state["p"] - goal))
            complete = output_free and not fallback and error <= .10
            print("%s,True,%.2f,%s,%d,%s,%.3f,%s" %
                  (label, spacing, output_free, replans, fallback, error,
                   "complete" if complete else "incomplete"), flush=True)


if __name__ == "__main__":
    main()
