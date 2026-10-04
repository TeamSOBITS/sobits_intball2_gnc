"""Pre-departure route shared by both tracking methods.

A*6 over the inflated occupancy grid, shortcut to a polyline. Built once before
departure and not rebuilt in flight, so the two local methods differ only in how
they follow it (docs/minco_astar_reference_global.md 2). ROS-free: the callers
supply the grid snapshot and the bounds.
"""

import numpy as np

from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import (
    segment_is_free,
    shortcut_path,
)

ASTAR_CONNECTIVITY = 6


def plan_reference_route(start, goal, grid, bounds):
    """Shortcut A*6 polyline from ``start`` to ``goal``, ends included.

    Raises :class:`AStarPlanningError` when either end is occupied or outside
    ``bounds``, or when no path exists.
    """
    path = AStarPlanner(grid.resolution, grid=grid, search_bounds=bounds,
                        connectivity=ASTAR_CONNECTIVITY).plan(start, goal)
    return [np.asarray(point, dtype=float) for point in shortcut_path(path, grid, bounds)]


def route_is_free(route, grid, bounds):
    """Whether every segment of ``route`` clears the inflated grid.

    Checked at half a voxel, as :func:`segment_is_free` does by default, against
    a newer grid than the one the route was planned on.
    """
    return all(segment_is_free(a, b, grid, bounds)
               for a, b in zip(route[:-1], route[1:]))


def densify(route, spacing):
    """``route`` with each segment split into equal steps of at most ``spacing``.

    The polyline's own vertices are kept, so the shape is unchanged; the extra
    points hold a spline through them against rounding the corners.
    """
    spacing = float(spacing)
    if not spacing > 0.0:
        raise ValueError("spacing must be positive")
    points = [np.asarray(route[0], dtype=float)]
    for a, b in zip(route[:-1], route[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        steps = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        points += [a + (b - a) * k / steps for k in range(1, steps + 1)]
    return points
