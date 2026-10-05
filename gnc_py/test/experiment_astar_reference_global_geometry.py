#!/usr/bin/env python3
"""Plan-4 (SCAN-Planner style) global reference, offline geometry check only.

A*6 -> shortcut -> densify -> position-only min-jerk with fixed time (dist / v,
first/last x2 as SCAN-Planner planGlobalTrajWaypoints). Measures how much of the
curve enters the inflated grid. No tracker involved.
"""
import os
import sys
import time

import numpy as np
import sobits_intball2_gnc_cpp

TEST = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST)
import experiment_jaxa_baseline_offline as ex  # noqa: E402
from experiment_local_only_delayed_detection import initial_facing_quat  # noqa: E402

base = ex.base
RES = base.RESOLUTION
V = base.LOCAL_MAX_VEL
DEPTH_FRAMES = 6
_, STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)


def min_jerk(points, times):
    """Per-axis C4 quintic spline through fixed points, zero v/a at both ends."""
    pts = np.asarray(points, float)
    m = len(times)
    coeffs = np.zeros((m, 6, 3))

    def row(t, d):
        r = np.zeros(6)
        for j in range(d, 6):
            r[j] = np.prod(range(j - d + 1, j + 1)) * t ** (j - d)
        return r

    a = np.zeros((6 * m, 6 * m))
    b = np.zeros((6 * m, 3))
    k = 0
    for d in range(3):
        a[k, 0:6] = row(0.0, d)
        b[k] = pts[0] if d == 0 else 0.0
        k += 1
    for i in range(m - 1):
        a[k, 6 * i:6 * i + 6] = row(times[i], 0)
        b[k] = pts[i + 1]
        k += 1
        a[k, 6 * (i + 1):6 * (i + 1) + 6] = row(0.0, 0)
        b[k] = pts[i + 1]
        k += 1
        for d in range(1, 5):
            a[k, 6 * i:6 * i + 6] = row(times[i], d)
            a[k, 6 * (i + 1):6 * (i + 1) + 6] = -row(0.0, d)
            k += 1
    for d in range(3):
        a[k, 6 * (m - 1):6 * m] = row(times[-1], d)
        b[k] = pts[-1] if d == 0 else 0.0
        k += 1
    sol = np.linalg.solve(a, b)
    coeffs[:] = sol.reshape(m, 6, 3)
    return coeffs


def sample(coeffs, times, dt=0.01):
    pos, vel = [], []
    for c, t_seg in zip(coeffs, times):
        for t in np.arange(0.0, t_seg, dt):
            pw = t ** np.arange(6)
            pos.append(pw @ c)
            vel.append((np.arange(1, 6) * t ** np.arange(5)) @ c[1:])
    return np.asarray(pos), np.asarray(vel)


def densify(route, spacing):
    if spacing is None:
        return [np.asarray(p, float) for p in route]
    out = [np.asarray(route[0], float)]
    for a, b in zip(route[:-1], route[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


def known_grid(scen, inflation):
    g = sobits_intball2_gnc_cpp.OccupancyGrid(RES, inflation)
    g.add_points(STATIC)
    for c, h in scen.boxes:
        g.add_box(list(c), list(h))
    for x in scen.people:  # person as its JEM-case box
        g.add_box(list(x), base.PERSON_HALF)
    return g


def depth_grid(scen, inflation, world):
    d = base._GUIDANCE["depth"]
    g = sobits_intball2_gnc_cpp.OccupancyGrid(RES, inflation)
    g.add_points(STATIC)
    g.enable_depth_layer(world.bounds[0], world.bounds[1], d["p_hit"], d["p_miss"], d["p_min"],
                         d["p_max"], d["p_occ"], d["min_range"], d["max_range"], d["skip_pixel"])
    q = initial_facing_quat(scen.start, scen.goal)
    for _ in range(DEPTH_FRAMES):
        world.integrate(g, scen.start, q, scen.at(0.0))
    return g


def inside_len(pos, grid):
    occ = np.array([grid.inflated_occupied(list(p)) for p in pos])
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return float(np.sum(seg[occ[1:] | occ[:-1]])) if len(seg) else 0.0


def plan_route(scen, plan_grid):
    return plan_reference_route(scen.start, scen.goal, plan_grid, base.BOUNDS)


def run(label, scen, grids, astar_extra):
    """grids: inflation -> grid (0.2 is the eval/production grid)."""
    plan_grid = grids[round(base.INFLATION + astar_extra, 3)]
    t0 = time.time()
    try:
        route = plan_reference_route(scen.start, scen.goal, plan_grid, base.BOUNDS)
    except Exception as exc:
        print("%s,astar+%.1f,-,ASTAR_FAIL %s" % (label, astar_extra, type(exc).__name__))
        return
    t_astar = time.time() - t0
    straight = len(route) == 2
    yaw = np.degrees(np.arctan2(*(np.asarray(route[1]) - np.asarray(route[0]))[[1, 0]]))
    for spacing in (None, 0.5, 0.25):
        pts = densify(route, spacing)
        times = np.array([np.linalg.norm(b - a) / V for a, b in zip(pts[:-1], pts[1:])])
        times[0] *= 2.0
        times[-1] *= 2.0
        coeffs = min_jerk(pts, times)
        pos, vel = sample(coeffs, times)
        sp = np.linalg.norm(vel, axis=1)
        print("%s,astar+%.1f,%s,verts=%d,len=%.2f,T=%.1f,vmax=%.3f,in0.2=%.2f,in0.1=%.2f,in0.0=%.2f,astar_s=%.1f,yaw0=%.0f%s" % (
            label, astar_extra, "none" if spacing is None else spacing, len(route),
            float(np.sum(np.linalg.norm(np.diff(np.asarray(route), axis=0), axis=1))),
            times.sum(), sp.max(), inside_len(pos, grids[0.2]), inside_len(pos, grids[0.1]),
            inside_len(pos, grids[0.0]), t_astar, yaw, ",STRAIGHT" if straight else ""))


def main():
    world = ex.DepthWorld(STATIC, RES)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    scens = [ex.paper_scenario(s) for s in range(5)]
    scens.append(ex.Scenario("person_x_plus_015", start, goal, people=[[11.10, -6.60, 4.90]]))
    print("case,astar,spacing,...  (in*: curve length [m] inside grid of that inflation)")
    for scen in scens:
        for cond in ("known", "depth"):
            grids = {}
            for infl in (0.0, 0.1, 0.2, 0.3):
                grids[infl] = known_grid(scen, infl) if cond == "known" else depth_grid(scen, infl, world)
            for extra in (0.0, 0.1):
                run("%s/%s" % (scen.label, cond), scen, grids, extra)
            if cond == "depth":
                # Pre-align to the first segment of the planned global, re-integrate, re-plan.
                try:
                    route = plan_route(scen, grids[0.2])
                except Exception as exc:
                    print("%s/depth_rot,ASTAR_FAIL %s" % (scen.label, type(exc).__name__))
                    continue
                q = initial_facing_quat(route[0], route[1])
                for g in grids.values():
                    for _ in range(DEPTH_FRAMES):
                        world.integrate(g, scen.start, q, scen.at(0.0))
                for extra in (0.0, 0.1):
                    run("%s/depth_rot" % scen.label, scen, grids, extra)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
