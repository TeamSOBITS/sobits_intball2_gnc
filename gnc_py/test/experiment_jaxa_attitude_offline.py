#!/usr/bin/env python3
"""JAXA baseline attitude: face the goal (current) vs face travel (as our method does).

Production tracker with the shared A* route (``reference_route``); only the
attitude is replaced at run time. ``travel`` points the camera at the point
``TRAVEL_AHEAD_M`` ahead along the path being tracked, with the commanded
attitude rotated toward it at most ``TRAVEL_MAX_RATE_DEG`` per second (the
RRT path is piecewise linear, so the raw direction jumps at its vertices).
The roll is kept by ``compute_q_des`` as in the goal-facing case.

Usage: python3 experiment_jaxa_attitude_offline.py <layouts> <speeds> <goal,travel> <repeats> [known]
Results: docs/archive/2026-10-05_jaxa_astar_global_corner_offline.md
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_jaxa_baseline_offline as ex  # noqa: E402
import experiment_jaxa_astar_corner_offline as corner  # noqa: E402
from experiment_local_only_delayed_detection import initial_facing_quat  # noqa: E402
import sobits_intball2_gnc_cpp  # noqa: E402
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, slerp  # noqa: E402
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import JaxaPlannerConfig  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking import jaxa_tracking_point_tracker as jt  # noqa: E402
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des  # noqa: E402

base = ex.base
TRAVEL_AHEAD_M = 0.5
TRAVEL_MAX_RATE_DEG = 20.0


def point_ahead(path, p, ahead):
    """Point ``ahead`` metres along ``path`` past the projection of ``p``."""
    best, k_best, t_best = np.inf, 0, 0.0
    for k, (a, b) in enumerate(zip(path[:-1], path[1:])):
        d = b - a
        sq = float(np.dot(d, d))
        t = 0.0 if sq < 1e-12 else float(np.clip(np.dot(p - a, d) / sq, 0.0, 1.0))
        dist = float(np.linalg.norm(a + t * d - p))
        if dist < best:
            best, k_best, t_best = dist, k, t
    q = path[k_best] + t_best * (path[k_best + 1] - path[k_best])
    left = ahead
    for k in range(k_best, len(path) - 1):
        seg = path[k + 1] - q
        n = float(np.linalg.norm(seg))
        if n >= left:
            return q + seg * left / n
        left -= n
        q = path[k + 1]
    return path[-1]


class TravelFacingTracker(jt.JaxaTrackingPointTracker):
    def _facing_goal(self, p, q_prev):
        if not hasattr(self, "_path"):  # constructor; run() sets the first leg afterwards
            return super()._facing_goal(p, q_prev)
        direction = point_ahead(self._path, p, TRAVEL_AHEAD_M) - p
        if np.linalg.norm(direction) < 1e-6:
            return np.asarray(q_prev, dtype=float)
        q_new = np.asarray(compute_q_des(direction, q_prev, 0.0, self._forward_axis), dtype=float)
        now = self._last_t
        last = getattr(self, "_att_t", now)
        self._att_t = now
        step = np.radians(TRAVEL_MAX_RATE_DEG) * max(now - last, 0.0)
        angle = geodesic_angle(q_prev, q_new)
        if angle <= step or angle < 1e-9:
            return q_new
        return np.asarray(slerp(q_prev, q_new, step / angle), dtype=float)


def run(scen, lookahead_m, attitude):
    grid = ex.WORLD.new_grid(scen)
    route = corner.plan_route(scen, grid)
    q0 = initial_facing_quat(scen.start, scen.goal)
    state = {"p": scen.start.copy(), "t": 0.0, "q": np.asarray(q0, dtype=float)}
    box = [None]
    real_threading = jt.threading
    jt.threading = types.SimpleNamespace(Thread=lambda **kw: ex.SimTimeThread(box, **kw))
    cls = TravelFacingTracker if attitude == "travel" else jt.JaxaTrackingPointTracker
    try:
        tracker = cls(
            scen.start, scen.goal, lambda: (state["p"], list(state["q"]), state["t"]), lambda _s: True,
            q0, grid, base.BOUNDS, lookahead_m, JaxaPlannerConfig(),
            collision_check_period=0.05, goal_facing_hold_m=0.3, forward_axis=ex.FWD,
            async_replan=True, seed=0, reference_route=route)
        box[0] = tracker
        if attitude == "travel":  # initial attitude: the first leg, as pre_align does
            tracker._q = np.asarray(compute_q_des(tracker.path[1] - scen.start, q0, 0.0, ex.FWD), dtype=float)

        def step(t, p):
            state["p"], state["t"] = p, t
            p_des, v_des, a_des, q = tracker.sample(t)
            return p_des, v_des, a_des, q, np.zeros(3)

        result = ex.simulate_jaxa(step, scen, grid, lambda: "plan_failed" if tracker.failed else None,
                                  None, pose_out=state)
    finally:
        jt.threading = real_threading
    result.update(plans=len(tracker.plans), note=tracker.failed or "")
    return result


def main():
    layouts = [int(x) for x in sys.argv[1].split(",")]
    speeds = [float(x) for x in sys.argv[2].split(",")]
    attitudes = sys.argv[3].split(",")
    repeats = int(sys.argv[4])
    known = len(sys.argv) > 5 and sys.argv[5] == "known"
    resolution, ex.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    ex.WORLD = ex.DepthWorld(ex.STATIC, resolution)
    ex.OPTS.update(known_boxes=known, controller="jaxa", control_rate=42.0, fixed_plan_latency=None,
                   planner_backend="cpp", noise=False, noise_seed=0)
    ratio = ex.JAXA_CFG["pos_ctl"]["kd"] / ex.JAXA_CFG["pos_ctl"]["kp"]
    print("layout,attitude,v,rep,status,time_s,min_clear_m,wall,att_err_max_deg,att_err_mean_deg,"
          "sat_frac,max_speed,plans,note", flush=True)
    for L in layouts:
        for v in speeds:
            for att in attitudes:
                for k in range(repeats):
                    r = run(ex.paper_scenario(L), v * ratio, att)
                    print("%d,%s,%.2f,%d,%s,%.1f,%.3f,%s,%.1f,%.1f,%.3f,%.3f,%d,%s" % (
                        L, att, v, k, r["status"], r["time_s"], r["min_obstacle_clear_m"],
                        r["wall_contact"], r["att_err_max_deg"], r["att_err_mean_deg"], r["sat_frac"],
                        r["max_speed"], r["plans"], r["note"].replace(",", ";")), flush=True)


if __name__ == "__main__":
    main()
