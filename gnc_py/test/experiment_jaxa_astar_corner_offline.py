#!/usr/bin/env python3
"""JAXA baseline following the shared A* shortcut route: sparse vs densified (0.25 m).

Results: docs/archive/2026-10-05_jaxa_astar_global_corner_offline.md

Wraps experiment_jaxa_baseline_offline.py (sim-matched JAXA controller)
and experiment_astar_reference_global_offline.py (MINCO with the same route).
Production code is not modified; the JAXA tracker's straight-line global is
replaced at run time by the route polyline.

Usage: python3 experiment_jaxa_astar_corner_offline.py <layouts> <speeds> <variants> [known]
  variants: comma list of minco,sparse,dense,prod (prod = the production tracker's
  reference_route, no run-time patching; dense = the same through the patched subclass)
"""
import os
import sys
import statistics
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_jaxa_baseline_offline as ex  # noqa: E402
import experiment_astar_reference_global_offline as ag  # noqa: E402
from experiment_local_only_delayed_detection import initial_facing_quat  # noqa: E402
import sobits_intball2_gnc_cpp  # noqa: E402
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import JaxaPlannerConfig  # noqa: E402
from sobits_intball2_gnc.guidance.search.reference_route import densify, route_is_free  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking import jaxa_tracking_point_tracker as jt  # noqa: E402

base = ex.base


class RouteTracker(jt.JaxaTrackingPointTracker):
    """Global path = the given polyline; the rejoin point is searched by arc length."""
    _route = None

    @property
    def _global_path(self):
        return self._route_arr

    @_global_path.setter
    def _global_path(self, _value):
        self._route_arr = np.asarray(RouteTracker._route, dtype=float)

    def _rejoin_point(self, p):
        route = self._route_arr
        seg = np.diff(route, axis=0)
        lens = np.linalg.norm(seg, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(lens)))
        # projection of p onto the polyline
        best = (np.inf, 0.0)
        for k, (a, d, l) in enumerate(zip(route[:-1], seg, lens)):
            s = np.clip(np.dot(p - a, d) / max(l * l, 1e-12), 0, 1)
            dist = np.linalg.norm(a + s * d - p)
            if dist < best[0]:
                best = (dist, cum[k] + s * l)
        s0, total = best[1], cum[-1]

        def at(s):
            k = min(np.searchsorted(cum, s, side="right") - 1, len(seg) - 1)
            return route[k] + seg[k] * (s - cum[k]) / max(lens[k], 1e-12)

        ss = np.append(np.arange(s0, total, 0.5 * float(self._grid.resolution)), total)
        lo, hi = (np.asarray(b, dtype=float) for b in self._bounds)
        blocked = [bool(np.any(q < lo) or np.any(q > hi) or self._grid.inflated_occupied(q.tolist()))
                   for q in (at(s) for s in ss)]
        goal = route[-1]
        if not any(blocked) or blocked[-1]:
            return goal
        last = len(blocked) - 1 - blocked[::-1].index(True)
        s_rejoin = ss[last] + jt._REJOIN_MARGIN_M
        if s_rejoin >= total:
            return goal
        for s_free, b in zip(ss[last + 1:], blocked[last + 1:]):
            if s_free >= s_rejoin and not b:
                return at(s_free)
        return goal


def poly_dist(points, route):
    route = np.asarray(route)
    best = np.full(len(points), np.inf)
    for a, b in zip(route[:-1], route[1:]):
        d = b - a
        s = np.clip(((points - a) @ d) / max(d @ d, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(a + s[:, None] * d - points, axis=1))
    return best


def plan_route(scen, grid):
    q_goal = initial_facing_quat(scen.start, scen.goal)
    for _ in range(ag.DEPTH_FRAMES):
        ex.WORLD.integrate(grid, scen.start, q_goal, scen.at(0.0))
    route = ag.plan_route(grid, scen.start, scen.goal)
    q_first = initial_facing_quat(route[0], route[1])
    for _ in range(ag.DEPTH_FRAMES):
        ex.WORLD.integrate(grid, scen.start, q_first, scen.at(0.0))
    if not route_is_free(route, grid.snapshot(), base.BOUNDS):
        route = ag.plan_route(grid, scen.start, scen.goal)
    return route


def run_jaxa_route(scen, lookahead_m, dense, trace, production=False):
    grid = ex.WORLD.new_grid(scen)
    route = plan_route(scen, grid)
    RouteTracker._route = densify(route, 0.25) if dense else route
    q0 = initial_facing_quat(scen.start, scen.goal)
    state = {"p": scen.start.copy(), "t": 0.0, "q": np.asarray(q0, dtype=float)}
    box = [None]
    real_threading = jt.threading
    jt.threading = types.SimpleNamespace(Thread=lambda **kw: ex.SimTimeThread(box, **kw))
    devs = []
    try:
        tracker = (jt.JaxaTrackingPointTracker if production else RouteTracker)(
            scen.start, scen.goal, lambda: (state["p"], list(state["q"]), state["t"]), lambda _s: True,
            q0, grid, base.BOUNDS, lookahead_m, JaxaPlannerConfig(),
            collision_check_period=0.05, goal_facing_hold_m=0.3, forward_axis=ex.FWD,
            async_replan=True, seed=0,
            **(dict(reference_route=route) if production else {}))
        box[0] = tracker

        def step(t, p):
            state["p"], state["t"] = p, t
            devs.append(float(poly_dist(np.atleast_2d(p), route)[0]))
            p_des, v_des, a_des, q = tracker.sample(t)
            return p_des, v_des, a_des, q, np.zeros(3)

        def finished():
            return "plan_failed" if tracker.failed else None

        result = ex.simulate_jaxa(step, scen, grid, finished, trace, pose_out=state)
    finally:
        jt.threading = real_threading
    result.update(route_dev_max=max(devs), plans=len(tracker.plans), route=route,
                  note=tracker.failed or "")
    return result


def run_minco(scen, trace):
    grid_route = plan_route(scen, ex.WORLD.new_grid(scen))  # same deterministic route, for dev
    devs = []
    real_simulate = ex.simulate_jaxa

    def wrapped(step, *a, **k):
        def step2(t, p):
            devs.append(float(poly_dist(np.atleast_2d(p), grid_route)[0]))
            return step(t, p)
        return real_simulate(step2, *a, **k)

    ex.simulate_jaxa = wrapped
    try:
        result = ag.run_minco_global(scen, trace)
    finally:
        ex.simulate_jaxa = real_simulate
    result.update(route_dev_max=max(devs), route=grid_route)
    return result


def turn_angles(route):
    r = np.asarray(route)
    out = []
    for a, b, c in zip(r[:-2], r[1:-1], r[2:]):
        u, v = b - a, c - b
        out.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    return out


def main():
    layouts = [int(x) for x in sys.argv[1].split(",")]
    speeds = [float(x) for x in sys.argv[2].split(",")]
    variants = sys.argv[3].split(",")
    known = len(sys.argv) > 4 and sys.argv[4] == "known"
    resolution, ex.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    ex.WORLD = ex.DepthWorld(ex.STATIC, resolution)
    ex.OPTS.update(known_boxes=known, controller="jaxa", control_rate=42.0, fixed_plan_latency=None,
                   planner_backend="cpp", noise=False, noise_seed=0)
    ratio = ex.JAXA_CFG["pos_ctl"]["kd"] / ex.JAXA_CFG["pos_ctl"]["kp"]
    print("layout,variant,v,d,status,time_s,min_clear_m,wall,route_dev_max_m,max_speed,goal_err,plans,verts,seg_lens,turns_deg,note",
          flush=True)
    for L in layouts:
        scen = ex.paper_scenario(L)
        runs = []
        if "minco" in variants:
            runs.append(("minco", float("nan"), lambda: run_minco(scen, None)))
        for v in speeds:
            d = v * ratio
            for var in ("sparse", "dense", "prod"):
                if var in variants:
                    runs.append((var, v, lambda d=d, var=var: run_jaxa_route(scen, d, var == "dense", None, var == "prod")))
        for name, v, fn in runs:
            r = fn()
            route = np.asarray(r["route"])
            lens = np.linalg.norm(np.diff(route, axis=0), axis=1)
            print("%d,%s,%.2f,%.3f,%s,%.1f,%.3f,%s,%.3f,%.3f,%.3f,%s,%d,%s,%s,%s" % (
                L, name, v, v * ratio if v == v else float("nan"), r["status"], r["time_s"],
                r["min_obstacle_clear_m"], r["wall_contact"], r["route_dev_max"], r["max_speed"],
                r["goal_error_m"], r["plans"], len(route),
                "/".join("%.2f" % x for x in lens), "/".join("%.0f" % x for x in turn_angles(route)), r.get("note", "").replace(",", ";")),
                flush=True)


if __name__ == "__main__":
    main()
