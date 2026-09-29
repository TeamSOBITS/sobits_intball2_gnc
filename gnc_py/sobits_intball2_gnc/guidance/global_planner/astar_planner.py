#!/usr/bin/env python3
"""Grid-based A* global planner (ROS-agnostic, pure)."""
import heapq

import numpy as np

from sobits_intball2_gnc.guidance.global_planner.base_global_planner import (
    BaseGlobalPlanner,
)

_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


class AStarPlanner(BaseGlobalPlanner):
    """A* search over a uniform 3D grid.

    ``obstacles`` (see ``base_global_planner.py`` for why this is
    provisional) is an optional set/iterable of blocked grid-index tuples
    ``(i, j, k)`` in this planner's own ``resolution``-sized lattice -- not
    yet the real Phase 4 obstacle-map format.
    """

    def __init__(self, resolution=0.1, search_margin=10, max_expansions=200000):
        self.resolution = float(resolution)
        self.search_margin = int(search_margin)
        self.max_expansions = int(max_expansions)

    def plan(self, start, goal, obstacles=None):
        obstacles = set(obstacles) if obstacles else set()
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        start_idx = self._to_grid(start)
        goal_idx = self._to_grid(goal)

        if start_idx == goal_idx:
            return [start, goal]

        bounds = self._search_bounds(start_idx, goal_idx)
        path_idx = self._search(start_idx, goal_idx, obstacles, bounds)

        return [start] + [self._to_world(idx) for idx in path_idx[1:-1]] + [goal]

    def _to_grid(self, point):
        return tuple(int(round(c / self.resolution)) for c in point)

    def _to_world(self, index):
        return np.array([c * self.resolution for c in index])

    def _search_bounds(self, start_idx, goal_idx):
        lo = tuple(min(s, g) - self.search_margin for s, g in zip(start_idx, goal_idx))
        hi = tuple(max(s, g) + self.search_margin for s, g in zip(start_idx, goal_idx))
        return lo, hi

    @staticmethod
    def _in_bounds(idx, bounds):
        lo, hi = bounds
        return all(lo[a] <= idx[a] <= hi[a] for a in range(3))

    @staticmethod
    def _heuristic(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _search(self, start_idx, goal_idx, obstacles, bounds):
        open_heap = [(self._heuristic(start_idx, goal_idx), start_idx)]
        came_from = {}
        g_score = {start_idx: 0.0}
        visited = set()
        expansions = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_idx:
                return self._reconstruct(came_from, current)

            expansions += 1
            if expansions > self.max_expansions:
                raise RuntimeError(
                    "AStarPlanner: exceeded max_expansions without reaching goal"
                )

            for offset in _NEIGHBOR_OFFSETS:
                neighbor = tuple(c + o for c, o in zip(current, offset))
                if neighbor in obstacles or not self._in_bounds(neighbor, bounds):
                    continue
                tentative_g = g_score[current] + float(np.linalg.norm(offset))
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    priority = tentative_g + self._heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, neighbor))

        raise RuntimeError("AStarPlanner: no path found to goal")

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
