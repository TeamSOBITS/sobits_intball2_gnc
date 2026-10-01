#!/usr/bin/env python3
"""Offline re-implementation of the IAC-22-D1.6.2 (Nishishita et al.) local
planner and tracking-point follower, used only as a comparison baseline.

Pipeline per plan: RRT* on the inflated occupancy grid -> shortcut
simplification -> cubic B-spline interpolation sampled into waypoints ->
collision check (replan on collision).  The follower commands the paper's
tracking point (Eqs. (1)-(2)), with ``v_des = a_des = 0`` (feedback only).  Parameters the paper does not state are listed in
``docs/jaxa_baseline_offline_verification.md``.  Not connected to guidance.
"""
import time

import numpy as np
from scipy.interpolate import make_interp_spline

from sobits_intball2_gnc.guidance.global_planner.path_shortcut import (
    point_is_free,
    segment_is_free,
    shortcut_path,
)


class RRTStar:
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
            raise RuntimeError("start occupied")
        if not point_is_free(goal, self.grid, self.bounds):
            raise RuntimeError("goal occupied")
        if self._free(start, goal):
            return [start, goal]
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
            raise RuntimeError("RRT* found no path")
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


def plan_local_path(start, goal, grid, bounds, seed, rrt_kwargs, max_attempts=5, spacing=0.5):
    """Returns ``(waypoints (N,3), info)``; raises RuntimeError if every attempt fails.

    A colliding smoothed path is replanned with a fresh RRT*, as in the paper;
    the attempt limit is ours (the paper gives none).
    """
    t0 = time.perf_counter()
    info = {"attempts": 0}
    for attempt in range(max_attempts):
        info["attempts"] = attempt + 1
        raw = RRTStar(grid, bounds, seed=seed + attempt, **rrt_kwargs).plan(start, goal)
        path = bspline_waypoints(shortcut_path(raw, grid, bounds), spacing)
        if path_is_free(path, grid, bounds):
            info["plan_s"] = time.perf_counter() - t0
            return path, info
    info["plan_s"] = time.perf_counter() - t0
    raise RuntimeError("no collision-free smoothed path after %d attempts" % max_attempts)


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


def facing_quat(direction, fallback):
    """+X toward ``direction`` with no roll (attitude only sets the body-frame force clamp offline)."""
    d = np.asarray(direction, dtype=float)
    if np.linalg.norm(d) < 1e-6:
        return fallback
    yaw = np.arctan2(d[1], d[0])
    pitch = -np.arctan2(d[2], np.hypot(d[0], d[1]))
    cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


class JaxaLookaheadFollower:
    """Tracking-point follower with collision-triggered replanning (sync, offline)."""

    def __init__(self, start, goal, grid, bounds, lookahead_m, q0, seed=0,
                 rrt_kwargs=None, collision_check_period=0.05):
        self.goal = np.asarray(goal, dtype=float)
        self.bounds = bounds
        self.grid = grid
        self.lookahead = float(lookahead_m)
        self.rrt_kwargs = rrt_kwargs or {}
        self.collision_check_period = float(collision_check_period)
        self.seed = int(seed)
        self.q = np.asarray(q0, dtype=float)
        self.plans = []
        self.failed = None
        self._last_check_t = -np.inf
        self._replan(np.asarray(start, dtype=float))

    def _replan(self, p):
        self.seed += 100
        try:
            path, info = plan_local_path(p, self.goal, self.grid, self.bounds, self.seed, self.rrt_kwargs)
        except RuntimeError as error:
            self.failed = str(error)
            self.plans.append({"plan_s": float("nan"), "attempts": 0})
            return False
        self.plans.append(info)
        self.path = path
        return True

    def set_obstacle_grid(self, grid):
        self.grid = grid
        self._last_check_t = -np.inf

    def setpoint(self, t, p):
        """Returns ``(p_des, q_des)``; holds ``p`` once planning has failed."""
        if self.failed:
            return p, self.q
        target, i = tracking_point(self.path, p, self.lookahead)
        if t - self._last_check_t >= self.collision_check_period - 1e-9:
            self._last_check_t = t
            remaining = self.path[i:]
            if len(remaining) >= 2 and not path_is_free(remaining, self.grid, self.bounds):
                if not self._replan(p):
                    return p, self.q
                target, _i = tracking_point(self.path, p, self.lookahead)
        self.q = facing_quat(target - p, self.q)
        return target, self.q
