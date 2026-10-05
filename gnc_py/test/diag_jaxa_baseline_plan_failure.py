"""Diagnose JAXA-baseline plan failures on the paper scenario (layout 0).

At each failed replan, re-runs every attempt and reports which stage collides
(RRT* raw / shortcut / B-spline), plus two variants: A = resample the shortcut
polyline every 0.5 m before interpolating, B = interpolate the raw RRT* nodes.
See docs/jaxa_baseline_jaxa_controller_comparison.md section 6. Written for the
earlier shortcut-then-interpolate pipeline; the production planner now
interpolates the raw RRT* nodes (variant B), so "shortcut" lines are reference only.
"""
import sys
sys.argv = ["x"]
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, sobits_intball2_gnc_cpp
import experiment_jaxa_baseline_offline as e
import sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner as jp
import sobits_intball2_gnc.guidance.trajectory_tracking.jaxa_tracking_point_tracker as jt
from sobits_intball2_gnc.guidance.search.path_shortcut import shortcut_path, point_is_free

res, e.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(e.base.MAP)
e.WORLD = e.DepthWorld(e.STATIC, res)
e.OPTS.update(controller="jaxa")
scen = e.paper_scenario(0)
orig = jp.plan_local_path
calls = []

def wrapped(start, goal, grid, bounds, seed, config):
    max_attempts, spacing = config.max_attempts, config.waypoint_spacing_m
    try:
        out = orig(start, goal, grid, bounds, seed, config)
        calls.append(("ok", np.array(start), seed))
        return out
    except RuntimeError as err:
        print("FAIL at start", np.round(start, 3), err, "start free:", point_is_free(start, grid, bounds))
        for a in range(max_attempts):
            try:
                raw = jp.RRTStar(grid, bounds, step=config.rrt_step_m, radius=config.rrt_radius_m,
                                 max_iterations=config.rrt_iterations, goal_bias=config.rrt_goal_bias,
                                 goal_tolerance=config.rrt_goal_tolerance_m, seed=seed + a).plan(start, goal)
            except RuntimeError as er:
                print("  attempt", a, "RRT*:", er); continue
            sc = np.asarray(shortcut_path(raw, grid, bounds), dtype=float)
            bs = jp.bspline_waypoints(sc, spacing)
            bad = [k for k, (p0, p1) in enumerate(zip(bs[:-1], bs[1:]))
                   if not jp.path_is_free([p0, p1], grid, bounds)]
            # depth into inflation: min clearance of shortcut corners to occupied (proxy: sample)
            print("  attempt", a, "raw free", jp.path_is_free(raw, grid, bounds), "n_raw", len(raw),
                  "shortcut free", jp.path_is_free(sc, grid, bounds), "corners", np.round(sc, 2).tolist(),
                  "bspline bad segs", bad, "of", len(bs) - 1)
            # Variant A: resample the shortcut polyline every `spacing` before interpolating.
            seg = np.linalg.norm(np.diff(sc, axis=0), axis=1); cum = np.concatenate(([0], np.cumsum(seg)))
            n = max(2, int(np.ceil(cum[-1] / spacing)) + 1)
            res_pts = np.array([np.interp(np.linspace(0, cum[-1], n), cum, sc[:, j]) for j in range(3)]).T
            bsA = jp.bspline_waypoints(res_pts, spacing)
            # Variant B: interpolate the raw RRT* nodes (no simplification).
            rw = np.asarray(raw, dtype=float); keep = np.concatenate(([True], np.linalg.norm(np.diff(rw, axis=0), axis=1) > 1e-6))
            bsB = jp.bspline_waypoints(rw[keep], spacing)
            dev = lambda bsx: max(np.min(np.linalg.norm(np.array([a + (b - a) * k / 20 for a, b in zip(sc[:-1], sc[1:]) for k in range(21)]) - q, axis=1)) for q in bsx)
            print("     A resampled free", jp.path_is_free(bsA, grid, bounds), "B raw free", jp.path_is_free(bsB, grid, bounds),
                  "| max dev from shortcut [m]: orig %.2f A %.2f" % (dev(bs), dev(bsA)))
        raise

jt.plan_local_path = wrapped
for d in (0.27, 0.11):
    print("=== d", d)
    r = e.run_jaxa(scen, d, 0, dict(max_iterations=1000, radius=0.6))
    print(r["status"], r["time_s"], r["note"])
