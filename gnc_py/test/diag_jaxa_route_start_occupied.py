#!/usr/bin/env python3
"""Why the JAXA baseline on the shared A* route fails with "start occupied".

Depth-detected paper layouts (boxes not known), sim-matched JAXA controller.
Records every local-plan call (vehicle position, whether it is in the inflated
grid, and whether that is the static map's inflation or a detected box's) and,
per guidance step, how long the vehicle is inside the static map's inflation.
Runs the shared route (``route``) and the straight chord (``straight``).

Usage: python3 diag_jaxa_route_start_occupied.py <layouts> <speed> <route|straight> <repeats>
Results: docs/2026-10-05_jaxa_astar_global_corner_offline.md
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
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import JaxaPlannerConfig  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking import jaxa_tracking_point_tracker as jt  # noqa: E402

base = ex.base


def run(scen, lookahead_m, use_route, static_grid):
    grid = ex.WORLD.new_grid(scen)
    if os.environ.get("NO_PREDEPARTURE_DEPTH"):
        # As experiment_jaxa_baseline_offline.py: the grid starts without depth.
        route = corner.plan_route(scen, ex.WORLD.new_grid(scen))
    else:
        route = corner.plan_route(scen, grid)
    departure = grid.snapshot()
    q0 = initial_facing_quat(scen.start, scen.goal)
    state = {"p": scen.start.copy(), "t": 0.0, "q": np.asarray(q0, dtype=float)}
    box = [None]
    calls, steps = [], {"n": 0, "in_static": 0, "in_grid": 0}
    seen, next_seen_check = {}, [0.0]
    # Each entry into the inflated grid: did the vehicle move into an already inflated cell,
    # or did the cell it was in become inflated (newly seen depth)?
    entries, prev = [], {"p": None, "in": False, "occupied_then": None}
    real_plan, real_threading = jt.plan_local_path, jt.threading

    def nearest_box(p):
        return int(np.argmin([ex.box_distance(p, c, h)[0] for c, h in scen.boxes]))

    def check_seen(t, p):
        """First time each box has a depth cell on it: vehicle distance to it and the angle
        between the camera axis and the direction to its nearest point."""
        cells = np.asarray(tracker._grid.depth_occupied_cells(), dtype=float).reshape(-1, 3)
        fwd = ex.quat_rotate(state["q"], ex.FWD)
        for i, (c, h) in enumerate(scen.boxes):
            if i in seen or not len(cells):
                continue
            if np.any(ex.box_distance(cells, c, h) <= 2 * base.RESOLUTION):
                near = np.clip(p, c - h, c + h)
                to_box = near - p
                angle = np.degrees(np.arccos(np.clip(
                    np.dot(fwd, to_box) / max(np.linalg.norm(to_box), 1e-9), -1.0, 1.0)))
                seen[i] = dict(t=t, dist=float(np.linalg.norm(to_box)), angle=float(angle))

    def plan(start, goal, g, bounds, seed, config):
        start = np.asarray(start, dtype=float)
        row = dict(t=state["t"], p=start, box=nearest_box(start), in_grid=bool(g.inflated_occupied(list(start))),
                   in_static=bool(static_grid.inflated_occupied(list(start))),
                   in_departure=bool(departure.inflated_occupied(list(start))),
                   box_dist=ex.WORLD.clearance(start, scen.at(state["t"])) + ex.ROBOT_RADIUS,
                   route_dev=float(corner.poly_dist(np.atleast_2d(start), route)[0]), error="")
        calls.append(row)
        try:
            return real_plan(start, goal, g, bounds, seed, config)
        except jt.JaxaPlanError as error:
            row["error"] = str(error)
            raise

    jt.plan_local_path = plan
    jt.threading = types.SimpleNamespace(Thread=lambda **kw: ex.SimTimeThread(box, **kw))
    try:
        tracker = jt.JaxaTrackingPointTracker(
            scen.start, scen.goal, lambda: (state["p"], list(state["q"]), state["t"]), lambda _s: True,
            q0, grid, base.BOUNDS, lookahead_m, JaxaPlannerConfig(),
            collision_check_period=0.05, goal_facing_hold_m=0.3, forward_axis=ex.FWD,
            async_replan=True, seed=0, **(dict(reference_route=route) if use_route else {}))
        box[0] = tracker

        def step(t, p):
            state["p"], state["t"] = p, t
            steps["n"] += 1
            steps["in_static"] += bool(static_grid.inflated_occupied(list(p)))
            p_des, v_des, a_des, q = tracker.sample(t)
            inside = bool(tracker._grid.inflated_occupied(list(p)))
            steps["in_grid"] += inside
            if inside and not prev["in"] and prev["p"] is not None:
                # The previous step's cell: free then; inflated now means the grid grew there.
                path = tracker.path
                entries.append(dict(t=t, grew_under=bool(tracker._grid.inflated_occupied(list(prev["p"]))),
                                    in_departure=bool(departure.inflated_occupied(list(p))),
                                    in_static=bool(static_grid.inflated_occupied(list(p))),
                                    des_inside=bool(tracker._grid.inflated_occupied(list(p_des))),
                                    path_dev=float(corner.poly_dist(np.atleast_2d(p), path)[0]),
                                    route_dev=float(corner.poly_dist(np.atleast_2d(p), route)[0]),
                                    on_route=path.shape == tracker._global_path.shape and bool(np.allclose(path, tracker._global_path)),
                                    path_free=bool(jt.path_is_free(path, departure, base.BOUNDS))))
            prev["p"], prev["in"] = np.asarray(p, dtype=float).copy(), inside
            if t >= next_seen_check[0]:
                next_seen_check[0] = t + 0.2
                check_seen(t, p)
            return p_des, v_des, a_des, q, np.zeros(3)

        result = ex.simulate_jaxa(step, scen, grid, lambda: "plan_failed" if tracker.failed else None,
                                  None, pose_out=state)
    finally:
        jt.plan_local_path, jt.threading = real_plan, real_threading
    return result, calls, steps, route, seen, entries


def main():
    layouts = [int(x) for x in sys.argv[1].split(",")]
    speed = float(sys.argv[2])
    use_route = sys.argv[3] == "route"
    repeats = int(sys.argv[4])
    resolution, ex.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    ex.WORLD = ex.DepthWorld(ex.STATIC, resolution)
    ex.OPTS.update(known_boxes=False, controller="jaxa", control_rate=42.0, fixed_plan_latency=None,
                   planner_backend="cpp", noise=False, noise_seed=0)
    static_grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, base.INFLATION)
    static_grid.add_points(ex.STATIC)
    d = speed * ex.JAXA_CFG["pos_ctl"]["kd"] / ex.JAXA_CFG["pos_ctl"]["kp"]
    for L in layouts:
        for k in range(repeats):
            r, calls, steps, route, seen, entries = run(ex.paper_scenario(L), d, use_route, static_grid)
            print("RUN layout%d %s rep%d status=%s t=%.1f steps_in_static=%.1f%% steps_in_grid=%.1f%% plans=%d"
                  % (L, sys.argv[3], k, r["status"], r["time_s"], 100.0 * steps["in_static"] / steps["n"],
                     100.0 * steps["in_grid"] / steps["n"], len(calls)), flush=True)
            print("  first seen (box: t, vehicle dist m, camera angle deg): %s" % "; ".join(
                "%d: %.1f s %.2f m %.0f" % (i, v["t"], v["dist"], v["angle"]) for i, v in sorted(seen.items())))
            for e in entries:
                print("  entry t=%.1f %s in_departure=%s in_static=%s setpoint_inside=%s dev_from_tracked_path=%.3f"
                      " dev_from_route=%.3f tracking_global_route=%s tracked_path_free_in_departure_grid=%s"
                      % (e["t"], "grew" if e["grew_under"] else "moved", e["in_departure"], e["in_static"],
                         e["des_inside"], e["path_dev"], e["route_dev"], e["on_route"], e["path_free"]))
            for c in calls:
                if c["in_grid"] or c["error"]:
                    print("  plan t=%.1f p=%s in_grid=%s in_static=%s in_departure=%s box%d_dist=%.3f route_dev=%.3f %s"
                          % (c["t"], np.round(c["p"], 2).tolist(), c["in_grid"], c["in_static"],
                             c["in_departure"], c["box"], c["box_dist"], c["route_dev"], c["error"]), flush=True)


if __name__ == "__main__":
    main()
