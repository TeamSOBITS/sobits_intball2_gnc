#!/usr/bin/env python3
"""Grid-based A* global planner (ROS-agnostic, pure)."""
import heapq
import math

import numpy as np

from sobits_intball2_gnc.guidance.global_planner.base_global_planner import (
    BaseGlobalPlanner,
)

# Legacy (non-occupancy) mode only; occupancy mode sizes its cap from the box.
LEGACY_MAX_EXPANSIONS = 200000

_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]
_FACE_NEIGHBOR_OFFSETS = (
    (-1, 0, 0), (1, 0, 0), (0, -1, 0),
    (0, 1, 0), (0, 0, -1), (0, 0, 1),
)
# (offset, step cost) pairs: the cost is fixed per offset, and recomputing it
# with np.linalg.norm per edge cost ~38% of a whole JEM-wide search.
_NEIGHBOR_STEPS = tuple(
    (offset, math.sqrt(sum(c * c for c in offset))) for offset in _NEIGHBOR_OFFSETS)
_FACE_NEIGHBOR_STEPS = tuple((offset, 1.0) for offset in _FACE_NEIGHBOR_OFFSETS)


class AStarPlanningError(RuntimeError):
    """Base error for a failed occupancy-grid A* request."""


class StartOccupiedError(AStarPlanningError):
    """The current-position cell is inflated occupied."""


class GoalOccupiedError(AStarPlanningError):
    """The requested goal cell is inflated occupied."""


class SearchBoundsError(AStarPlanningError):
    """Start or goal is outside the explicitly permitted search box."""


class AStarPlanner(BaseGlobalPlanner):
    """A* search over a uniform 3D grid.

    Legacy mode accepts ``obstacles`` as grid-index tuples and retains the
    original round-to-grid behavior for its existing contract tests.

    Occupancy mode is selected by passing both ``grid`` and ``search_bounds``
    at construction. ``grid`` is a (normally snapshot) object exposing
    ``resolution`` and ``inflated_occupied(point)``; only its inflated layer
    is consulted. ``search_bounds`` is ``(lower, upper)`` in world coordinates
    and is mandatory because the map treats points outside its stored voxels as
    free. Occupancy mode uses floor-to-cell and cell-center coordinates, the
    same convention as ``OccupancyGrid``. ``connectivity`` selects 6 face
    neighbors or 26 neighbors. Before accepting a diagonal edge, it requires
    every voxel touched at the shared edge/corner to be free, preventing
    obstacle corner-cutting.
    """

    def __init__(self, resolution=0.1, search_margin=10, max_expansions=None,
                 grid=None, search_bounds=None, connectivity=26):
        if (grid is None) != (search_bounds is None):
            raise ValueError("grid and search_bounds must be supplied together")
        if grid is not None and abs(float(grid.resolution) - float(resolution)) > 1e-12:
            raise ValueError("resolution must equal grid.resolution in occupancy mode")
        if connectivity not in (6, 26):
            raise ValueError("connectivity must be 6 or 26")
        self.resolution = float(resolution if grid is None else grid.resolution)
        self.search_margin = int(search_margin)
        self._grid = grid
        self._search_bounds_world = self._validate_search_bounds(search_bounds)
        self.connectivity = int(connectivity)
        if max_expansions is not None:
            self.max_expansions = int(max_expansions)
        elif grid is None:
            self.max_expansions = LEGACY_MAX_EXPANSIONS
        else:
            # Every cell of the box, which the search cannot exceed: it never
            # leaves the box and never expands a cell twice. A smaller cap
            # cannot tell "no path exists" from "I stopped looking", and the
            # 200000 it used to default to is 36% of the free cells of one JEM
            # (docs/minco_astar_reference_global.md 3.8).
            lo, hi = self._occupancy_bounds()
            self.max_expansions = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)
        self.last_expansions = 0

    def plan(self, start, goal, obstacles=None):
        if self._grid is not None:
            if obstacles is not None:
                raise ValueError("occupancy mode reads obstacles only from grid")
            return self._plan_occupancy(start, goal)

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

    def _plan_occupancy(self, start, goal):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("start and goal must each be 3-element points")
        start_idx = self._to_grid_floor(start)
        goal_idx = self._to_grid_floor(goal)
        bounds = self._occupancy_bounds()
        if not self._in_bounds(start_idx, bounds) or not self._in_bounds(goal_idx, bounds):
            raise SearchBoundsError("start or goal is outside search_bounds")
        if self._occupied(start_idx):
            raise StartOccupiedError("start cell is inflated occupied")
        if self._occupied(goal_idx):
            raise GoalOccupiedError("goal cell is inflated occupied")
        if start_idx == goal_idx:
            return [start, goal]

        path_idx = self._search_occupancy(start_idx, goal_idx, bounds)
        return [start] + [self._to_cell_center(idx) for idx in path_idx[1:-1]] + [goal]

    def _to_grid(self, point):
        return tuple(int(round(c / self.resolution)) for c in point)

    def _to_grid_floor(self, point):
        return tuple(int(np.floor(c / self.resolution)) for c in point)

    def _to_world(self, index):
        return np.array([c * self.resolution for c in index])

    def _to_cell_center(self, index):
        return (np.asarray(index, dtype=float) + 0.5) * self.resolution

    @staticmethod
    def _validate_search_bounds(search_bounds):
        if search_bounds is None:
            return None
        if len(search_bounds) != 2:
            raise ValueError("search_bounds must be (lower, upper)")
        lower, upper = (np.asarray(v, dtype=float) for v in search_bounds)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("search_bounds lower/upper must be ordered 3-element points")
        return lower, upper

    def _occupancy_bounds(self):
        lower, upper = self._search_bounds_world
        # Retain cells whose centers are in the explicit continuous box.
        lo = tuple(np.ceil(lower / self.resolution - 0.5 - 1e-9).astype(int))
        hi = tuple(np.floor(upper / self.resolution - 0.5 + 1e-9).astype(int))
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("search_bounds contains no grid cell centers")
        return lo, hi

    def _occupied(self, index):
        res = self.resolution
        return bool(self._grid.inflated_occupied(
            [(index[0] + 0.5) * res, (index[1] + 0.5) * res, (index[2] + 0.5) * res]))

    def _search_bounds(self, start_idx, goal_idx):
        lo = tuple(min(s, g) - self.search_margin for s, g in zip(start_idx, goal_idx))
        hi = tuple(max(s, g) + self.search_margin for s, g in zip(start_idx, goal_idx))
        return lo, hi

    @staticmethod
    def _in_bounds(idx, bounds):
        lo, hi = bounds
        return (lo[0] <= idx[0] <= hi[0] and lo[1] <= idx[1] <= hi[1]
                and lo[2] <= idx[2] <= hi[2])

    @staticmethod
    def _heuristic(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

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
            self.last_expansions = expansions
            if expansions > self.max_expansions:
                raise RuntimeError(
                    "AStarPlanner: exceeded max_expansions without reaching goal"
                )

            steps = _FACE_NEIGHBOR_STEPS if self.connectivity == 6 else _NEIGHBOR_STEPS
            for offset, step_cost in steps:
                neighbor = (current[0] + offset[0], current[1] + offset[1],
                            current[2] + offset[2])
                if neighbor in obstacles or not self._in_bounds(neighbor, bounds):
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    priority = tentative_g + self._heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, neighbor))

        raise RuntimeError("AStarPlanner: no path found to goal")

    def _search_occupancy(self, start_idx, goal_idx, bounds):
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
            self.last_expansions = expansions
            if expansions > self.max_expansions:
                raise AStarPlanningError("A* exceeded max_expansions")

            steps = _FACE_NEIGHBOR_STEPS if self.connectivity == 6 else _NEIGHBOR_STEPS
            for offset, step_cost in steps:
                neighbor = (current[0] + offset[0], current[1] + offset[1],
                            current[2] + offset[2])
                if (not self._in_bounds(neighbor, bounds)
                        or not self._edge_is_free(current, neighbor, bounds)):
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    priority = tentative_g + self._heuristic(neighbor, goal_idx)
                    heapq.heappush(open_heap, (priority, neighbor))

        raise AStarPlanningError("A* found no path inside search_bounds")

    def _edge_is_free(self, current, neighbor, bounds):
        """True only when all voxels touched by this one-cell edge are free.

        A diagonal center-to-center move crosses a voxel edge or corner.  The
        subset cells are the supercover at that crossing; checking them all
        prevents a route from slipping between inflated occupied voxels.
        """
        delta = (neighbor[0] - current[0], neighbor[1] - current[1],
                 neighbor[2] - current[2])
        active_axes = [axis for axis in (0, 1, 2) if delta[axis]]
        if len(active_axes) == 1:
            # A face move touches only the destination cell; the general
            # supercover loop below would rebuild a list to say the same.
            return self._in_bounds(neighbor, bounds) and not self._occupied(neighbor)
        for mask in range(1, 1 << len(active_axes)):
            touched = list(current)
            for bit, axis in enumerate(active_axes):
                if mask & (1 << bit):
                    touched[axis] += delta[axis]
            touched = tuple(touched)
            if not self._in_bounds(touched, bounds) or self._occupied(touched):
                return False
        return True

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
