#!/usr/bin/env python3
"""Frozen Python reference for native JAXA planner equivalence tests.

Re-implements T. Nishishita et al., "Dynamic Motion Planning of FPV Camera
Free-Flyers for Autonomous Crew Tracking and Collision Avoidance", IAC-22-D1.6.2
(2022), 3.1-3.2 from the paper's text (no JAXA source exists for this part):
RRT* on the inflated occupancy grid -> cubic B-spline interpolation of the RRT*
nodes sampled into waypoints -> collision check, replanning on collision; and
the tracking point of Eqs. (1)-(2). The paper names no separate simplification
step: Fig. 4(b) is "the path simplified and smoothed by B-Spline interpolation".
``trajectory_tracking/jaxa_tracking_point_tracker.py`` drives it. Design and the
values the paper does not give: docs/jaxa_baseline_gazebo_port_plan.md.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline

from sobits_intball2_gnc.guidance.global_planner.path_shortcut import (
    point_is_free,
    segment_is_free,
)


class JaxaPlanError(RuntimeError):
    """No collision-free path (start/goal occupied, RRT* failed, or every attempt collided)."""


@dataclass(frozen=True)
class JaxaPlannerConfig:
    """Values the paper does not give; see docs/jaxa_baseline_gazebo_port_plan.md 2 節."""
    rrt_step_m: float = 0.3
    rrt_radius_m: float = 0.6
    rrt_iterations: int = 1000
    rrt_goal_bias: float = 0.1
    rrt_goal_tolerance_m: float = 0.3
    waypoint_spacing_m: float = 0.5
    max_attempts: int = 5


class RRTStar:
    """RRT* (fixed rewire radius) on the inflated grid inside ``bounds``; searches
    the full iteration budget and returns the cheapest start->goal path."""

    def __init__(self, grid, bounds, step=0.3, radius=0.6, max_iterations=1000,
                 goal_bias=0.1, goal_tolerance=0.3, seed=None):
        self.grid = grid
        self.bounds = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
        self.step = float(step)
        self.radius = float(radius)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.goal_tolerance = float(goal_tolerance)
        self.rng = np.random.RandomState(seed)

    def _free(self, a, b):
        return segment_is_free(a, b, self.grid, self.bounds)

    def plan(self, start, goal):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if not point_is_free(start, self.grid, self.bounds):
            raise JaxaPlanError("start occupied")
        if not point_is_free(goal, self.grid, self.bounds):
            raise JaxaPlanError("goal occupied")
        n_max = self.max_iterations + 1
        nodes = np.empty((n_max, 3))
        parent = np.full(n_max, -1)
        cost = np.zeros(n_max)
        children = [[] for _ in range(n_max)]
        nodes[0], n = start, 1
        best_goal_parent, best_goal_cost = -1, np.inf
        lower, upper = self.bounds
        for _ in range(self.max_iterations):
            sample = goal if self.rng.random_sample() < self.goal_bias else self.rng.uniform(lower, upper)
            dists = np.linalg.norm(nodes[:n] - sample, axis=1)
            nearest = int(np.argmin(dists))
            direction = sample - nodes[nearest]
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            new = nodes[nearest] + direction * min(1.0, self.step / length)
            if not self._free(nodes[nearest], new):
                continue
            near_d = np.linalg.norm(nodes[:n] - new, axis=1)
            near = np.flatnonzero(near_d <= self.radius)
            best_parent, best_cost = nearest, cost[nearest] + np.linalg.norm(new - nodes[nearest])
            for j in near[np.argsort(cost[near] + near_d[near])]:
                if cost[j] + near_d[j] >= best_cost:
                    break
                if j != nearest and self._free(nodes[j], new):
                    best_parent, best_cost = int(j), cost[j] + near_d[j]
                    break
            k = n
            nodes[k], parent[k], cost[k] = new, best_parent, best_cost
            children[best_parent].append(k)
            n += 1
            for j in near:
                if j == best_parent or cost[k] + near_d[j] >= cost[j] - 1e-9:
                    continue
                if not self._free(new, nodes[j]):
                    continue
                children[parent[j]].remove(j)
                parent[j] = k
                children[k].append(int(j))
                delta = cost[k] + near_d[j] - cost[j]
                stack = [int(j)]
                while stack:
                    m = stack.pop()
                    cost[m] += delta
                    stack.extend(children[m])
            to_goal = np.linalg.norm(goal - new)
            if to_goal <= self.goal_tolerance and cost[k] + to_goal < best_goal_cost and self._free(new, goal):
                best_goal_parent, best_goal_cost = k, cost[k] + to_goal
            if best_goal_parent >= 0:
                # Earlier goal connections may have been rewired to cheaper parents.
                best_goal_cost = min(best_goal_cost, cost[best_goal_parent]
                                     + np.linalg.norm(goal - nodes[best_goal_parent]))
        if best_goal_parent < 0:
            raise JaxaPlanError("RRT* found no path")
        path, idx = [goal], best_goal_parent
        while idx != -1:
            path.append(nodes[idx].copy())
            idx = parent[idx]
        return path[::-1]


def bspline_waypoints(points, spacing=0.5):
    """Cubic B-spline interpolation through ``points`` (chord-length parameter),
    sampled at ``ceil(length / spacing)`` equal parameter steps.

    The paper smooths by "B-Spline interpolation" and represents the local path
    as waypoints whose spacing "is not necessarily constant"; equal parameter
    steps give that.  0.5 m is the paper's tracking-test waypoint spacing.
    """
    pts = np.asarray(points, dtype=float)
    # RRT* may end with a node at the goal plus the appended goal; knots must be distinct.
    pts = pts[np.concatenate(([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9))]
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(chord.sum())
    count = max(2, int(np.ceil(length / spacing)) + 1)
    u = np.linspace(0.0, 1.0, count)
    if len(pts) == 2:
        return pts[0] + u[:, None] * (pts[1] - pts[0])
    knots_u = np.concatenate(([0.0], np.cumsum(chord))) / length
    spline = make_interp_spline(knots_u, pts, k=min(3, len(pts) - 1))
    return spline(u)


def path_is_free(path, grid, bounds):
    return all(segment_is_free(a, b, grid, bounds) for a, b in zip(path[:-1], path[1:]))


def plan_local_path(start, goal, grid, bounds, seed, config):
    """Returns ``(waypoints (N,3), info)``; raises :class:`JaxaPlanError` if every attempt fails.

    A colliding smoothed path is replanned with a fresh RRT*, as in the paper;
    the attempt limit is ours (the paper gives none).
    """
    t0 = time.perf_counter()
    info = {"attempts": 0}
    max_attempts = config.max_attempts
    for attempt in range(max_attempts):
        info["attempts"] = attempt + 1
        raw = RRTStar(grid, bounds, step=config.rrt_step_m, radius=config.rrt_radius_m,
                      max_iterations=config.rrt_iterations, goal_bias=config.rrt_goal_bias,
                      goal_tolerance=config.rrt_goal_tolerance_m, seed=seed + attempt).plan(start, goal)
        path = bspline_waypoints(raw, config.waypoint_spacing_m)
        if path_is_free(path, grid, bounds):
            info["plan_s"] = time.perf_counter() - t0
            return path, info
    info["plan_s"] = time.perf_counter() - t0
    raise JaxaPlanError("no collision-free smoothed path after %d attempts" % max_attempts)


def tracking_point(path, p, lookahead):
    """Paper Eqs. (1)-(2) on the closest waypoint ``i`` and segment ``i -> i+1``.

    Eq. (1) is the foot of the perpendicular on that segment's line, written
    with squared norms (the printed equation drops the squares).  Eq. (2)
    multiplies ``d`` by the raw segment vector; we use its unit vector so that
    ``d`` is a length, as the text and Fig. 5 describe.  Clamping the point at
    the goal on the last segment is ours (the paper does not say).
    """
    i = int(np.argmin(np.linalg.norm(path - p, axis=1)))
    i = min(i, len(path) - 2)
    r_i, r_n = path[i] - p, path[i + 1] - p
    seg = r_n - r_i
    seg_sq = float(seg @ seg)
    if seg_sq < 1e-12:
        return path[i + 1], i
    n = ((r_n @ r_n - r_i @ r_n) * r_i + (r_i @ r_i - r_i @ r_n) * r_n) / seg_sq
    unit = seg / np.sqrt(seg_sq)
    t = n + lookahead * unit
    if i == len(path) - 2 and (t - r_n) @ unit > 0.0:
        t = r_n
    return p + t, i
