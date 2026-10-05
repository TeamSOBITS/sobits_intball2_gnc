#!/usr/bin/env python3
"""Plan 4 offline arrival run: replan MINCO with the shared A* reference global.

Wraps experiment_jaxa_baseline_offline.py; only ``run_minco`` is replaced. Usage: same
arguments as that script (use --methods minco).

Production code is no longer patched at run time: the route comes from
``guidance.search.reference_route`` and goes into the tracker's own
``reference_route`` argument, so this reproduces what ``guidance_executor`` does
for ``guidance.global_planner: astar`` (docs/minco_astar_reference_global.md 3).
The only substitution left is the sim-time thread, which stands in for ROS, not
for any planning code.
"""
import statistics
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_jaxa_baseline_offline as ex  # noqa: E402
from experiment_local_only_delayed_detection import initial_facing_quat  # noqa: E402
from sobits_intball2_gnc.guidance.search.reference_route import (  # noqa: E402
    plan_reference_route,
    route_is_free,
)
from sobits_intball2_gnc.guidance.trajectory_tracking import replan_minco_tracker  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker  # noqa: E402

base = ex.base
DEPTH_FRAMES = 6


def plan_route(grid, start, goal):
    return plan_reference_route(start, goal, grid.snapshot(), base.BOUNDS)


def run_minco_global(scen, trace=None):
    start, goal = scen.start, scen.goal
    grid = ex.WORLD.new_grid(scen)
    q_goal = initial_facing_quat(start, goal)
    for _ in range(DEPTH_FRAMES):  # pre-align 1: facing the goal
        ex.WORLD.integrate(grid, start, q_goal, scen.at(0.0))
    t0 = time.perf_counter()
    route = plan_route(grid, start, goal)
    q_first = initial_facing_quat(route[0], route[1])
    for _ in range(DEPTH_FRAMES):  # pre-align 2: facing the first segment
        ex.WORLD.integrate(grid, start, q_first, scen.at(0.0))
    replanned = not route_is_free(route, grid.snapshot(), base.BOUNDS)
    if replanned:
        route = plan_route(grid, start, goal)
    plan_s = time.perf_counter() - t0
    state = {"p": start.copy(), "t": 0.0, "q": np.asarray(q_goal, dtype=float)}
    box = [None]
    real_threading = replan_minco_tracker.threading
    if ex.sim_matched():
        replan_minco_tracker.threading = types.SimpleNamespace(
            Thread=lambda **kw: ex.SimTimeThread(box, **kw))
    try:
        tracker = ReplanMincoTracker(
            start, goal, lambda: (state["p"], list(state["q"]), state["t"]),
            lambda _stamp: True, q_goal, base.TARGET_SPEED, base.MAX_ACCEL,
            reference_route=route, via_half_width=0., face_travel=True,
            forward_axis=ex.FWD, local_max_vel=ex.OPTS["minco_max_vel"],
            local_piece_length_m=base.LOCAL_PIECE_LENGTH,
            obstacle_grid=grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
            local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
            async_replan=ex.sim_matched())
        box[0] = tracker
        solves = [tracker.last_replan_solve_seconds]

        def step(t, p):
            state["p"], state["t"] = p, t
            p_des, v_des, a_des, q = tracker.sample(min(t, tracker.total_duration))
            if tracker.last_replan_occurred:
                solves.append(tracker.last_replan_solve_seconds)
            body = tracker.last_body_angular
            return p_des, v_des, a_des, np.asarray(q, dtype=float), np.asarray(body[0], dtype=float)

        def finished():
            return "replanning_stopped" if tracker.replanning_stopped else None

        result = ex.simulate_jaxa(step, scen, grid, finished, trace, pose_out=state)
    finally:
        replan_minco_tracker.threading = real_threading
    result.update(plans=len(solves), plan_ms_med=1e3 * statistics.median(solves),
                  plan_ms_max=1e3 * max(solves),
                  note="verts=%d replanned=%s astar_s=%.1f global_T=%.1f" % (
                      len(route), replanned, plan_s, tracker.trajectory.global_total_duration))
    return result


if __name__ == "__main__":
    ex.run_minco = run_minco_global
    ex.main()
