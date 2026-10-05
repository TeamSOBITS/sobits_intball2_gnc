#!/usr/bin/env python3
"""Current local-only replanning after a delayed synthetic depth detection.

The person is deliberately absent at tracker construction.  It is inserted
only once the ideal tracked position is within ``detect_distance_m`` of its
centre, matching the causal order of a depth observation followed by a local
replan.  This is not a camera-visibility or dynamics validation.
"""
import argparse
import os
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import offline_common as base
from global_minco_candidate_selector import curve_is_free
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

FWD = np.array([1., 0., 0.])


def initial_facing_quat(start, goal):
    """The avoidance profile pre-aligns the +X camera to the initial tangent."""
    direction = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    yaw = float(np.arctan2(direction[1], direction[0]))
    return np.array([0., 0., np.sin(yaw / 2.), np.cos(yaw / 2.)])


def run_case(static, start, goal, person, detect_distance_m, include_static):
    points = static if include_static else []
    clear_grid = base.make_grid(points, [])
    detected_grid = base.make_grid(points, [(person, base.PERSON_HALF)])
    state = {"p": start.copy(), "t": 0.0}
    q0 = initial_facing_quat(start, goal)
    try:
        tracker = ReplanMincoTracker(
            start, goal, lambda: (state["p"], list(q0), state["t"]),
            lambda _stamp: True, q0, base.TARGET_SPEED, base.MAX_ACCEL,
            route_waypoints=None, via_half_width=0., face_travel=True,
            forward_axis=FWD, local_max_vel=base.LOCAL_MAX_VEL, local_piece_length_m=base.LOCAL_PIECE_LENGTH,
            obstacle_grid=clear_grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
            local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
            async_replan=False)
    except Exception as error:
        return "construct_error", 0, 0, np.nan, type(error).__name__
    detected, output_free, replans, fallbacks, t, next_depth_t = False, True, 0, 0, 0.0, .1
    while t < tracker.total_duration and t < 300.0:
        t += .05
        p, _v, _a, _q = tracker.sample(t)
        state["p"], state["t"] = p, t
        depth_frame = not detected and t + 1e-9 >= next_depth_t
        if depth_frame:
            next_depth_t += .1
        if depth_frame and np.linalg.norm(p - person) <= detect_distance_m:
            tracker.set_obstacle_grid(detected_grid)
            detected = True
        if detected:
            output_free &= not detected_grid.inflated_occupied(list(p))
        replans += int(tracker.last_replan_occurred)
        fallbacks += int(tracker.last_local_fallback)
        if not output_free or tracker.replanning_stopped:
            break
    error = float(np.linalg.norm(state["p"] - goal))
    status = ("complete" if detected and output_free and not tracker.replanning_stopped
              and error <= .10 else "incomplete")
    return status, replans, fallbacks, error, "detected=%s" % detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect-distance", type=float, required=True)
    parser.add_argument("--include-static", action="store_true")
    args = parser.parse_args()
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("# profile=%s target_speed=%.2f local=%.1fm/%.1fs/%.2fmps grid=%.2f/%.2f" %
          (base.OFFLINE_PROFILE, base.TARGET_SPEED, base.LOCAL_HORIZON,
           base.LOCAL_REPLAN_PERIOD, base.LOCAL_MAX_VEL, base.RESOLUTION, base.INFLATION))
    print("case,detect_distance_m,status,replans,fallbacks,goal_error_m,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            result = run_case(static, start, goal, np.array([x, y, 4.90]),
                              args.detect_distance, args.include_static)
            print("x=%.2f y=%.2f,%.1f,%s,%d,%d,%.3f,%s" %
                  (x, y, args.detect_distance, *result), flush=True)


if __name__ == "__main__":
    main()
