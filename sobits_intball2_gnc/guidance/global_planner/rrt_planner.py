#!/usr/bin/env python3
"""Sampling-based RRT global planner (ROS-agnostic, pure)."""
import numpy as np

from sobits_intball2_gnc.guidance.global_planner.base_global_planner import (
    BaseGlobalPlanner,
)


class RRTPlanner(BaseGlobalPlanner):
    """Rapidly-exploring random tree (RRT) search in continuous 3D space.

    ``obstacles`` (see ``base_global_planner.py`` for why this is
    provisional) is an optional iterable of ``(center, radius)`` sphere
    obstacles -- not yet the real Phase 4 obstacle-map format.

    ``bounds``, if given, is ``[(xmin, xmax), (ymin, ymax), (zmin, zmax)]``
    for the sampling region; otherwise it is derived from start/goal plus
    ``bounds_margin``.
    """

    def __init__(
        self,
        bounds=None,
        step_size=0.3,
        goal_tolerance=0.2,
        goal_bias=0.1,
        max_iterations=5000,
        bounds_margin=1.0,
        seed=None,
    ):
        self.bounds = bounds
        self.step_size = float(step_size)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_bias = float(goal_bias)
        self.max_iterations = int(max_iterations)
        self.bounds_margin = float(bounds_margin)
        self._rng = np.random.RandomState(seed)

    def plan(self, start, goal, obstacles=None):
        obstacles = list(obstacles) if obstacles else []
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if not self._segment_collides(start, goal, obstacles):
            return [start, goal]

        bounds = self.bounds or self._default_bounds(start, goal)
        nodes = [start]
        parents = [-1]

        for _ in range(self.max_iterations):
            sample = goal if self._rng.random_sample() < self.goal_bias else self._sample(bounds)
            nearest_idx = self._nearest(nodes, sample)
            new_point = self._steer(nodes[nearest_idx], sample)

            if self._segment_collides(nodes[nearest_idx], new_point, obstacles):
                continue

            nodes.append(new_point)
            parents.append(nearest_idx)

            near_goal = np.linalg.norm(new_point - goal) <= self.goal_tolerance
            if near_goal and not self._segment_collides(new_point, goal, obstacles):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                return self._reconstruct(nodes, parents, len(nodes) - 1)

        raise RuntimeError("RRTPlanner: exceeded max_iterations without reaching goal")

    def _default_bounds(self, start, goal):
        lo = np.minimum(start, goal) - self.bounds_margin
        hi = np.maximum(start, goal) + self.bounds_margin
        return list(zip(lo, hi))

    def _sample(self, bounds):
        return np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])

    @staticmethod
    def _nearest(nodes, point):
        dists = [np.linalg.norm(n - point) for n in nodes]
        return int(np.argmin(dists))

    def _steer(self, origin, target):
        direction = target - origin
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return target
        return origin + direction / dist * self.step_size

    @staticmethod
    def _segment_collides(a, b, obstacles):
        for center, radius in obstacles:
            if _point_segment_distance(np.asarray(center, dtype=float), a, b) < radius:
                return True
        return False

    @staticmethod
    def _reconstruct(nodes, parents, goal_idx):
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path


def _point_segment_distance(point, seg_a, seg_b):
    seg = seg_b - seg_a
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return np.linalg.norm(point - seg_a)
    t = np.clip(np.dot(point - seg_a, seg) / seg_len_sq, 0.0, 1.0)
    closest = seg_a + t * seg
    return np.linalg.norm(point - closest)
