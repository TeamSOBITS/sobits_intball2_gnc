"""Pre-departure route shared by both tracking methods.

A*6 over the inflated occupancy grid, shortcut to a polyline. Built once before
departure and not rebuilt in flight, so the two local methods differ only in how
they follow it (docs/minco_astar_reference_global.md 2). ROS-free: the callers
supply the grid snapshot and the bounds.
"""

import numpy as np

from sobits_intball2_gnc.guidance.search.path_shortcut import (
    segment_is_free,
    shortcut_path,
)


class AStarPlanningError(RuntimeError):
    """Base error for a failed occupancy-grid A* request."""


class StartOccupiedError(AStarPlanningError):
    """The current-position cell is inflated occupied."""


class GoalOccupiedError(AStarPlanningError):
    """The requested goal cell is inflated occupied."""


class SearchBoundsError(AStarPlanningError):
    """Start or goal is outside the explicitly permitted search box."""


ASTAR_CONNECTIVITY = 6
# Try the endpoints' box grown by this margin before the whole one; 0 disables
# it. Off by default: measured over a JEM the narrow box saves nothing, because
# A* is goal-directed and barely leaves it anyway, while a route that does need
# the room pays for both passes (docs/minco_astar_reference_global.md 3.8).
ASTAR_NARROW_MARGIN_M = 0.0


def plan_reference_route(start, goal, grid, bounds):
    """Shortcut A*6 polyline from ``start`` to ``goal``, ends included.

    Raises :class:`AStarPlanningError` when either end is occupied or outside
    ``bounds``, or when no path exists.
    """
    import sobits_intball2_gnc_cpp  # deferred, as obstacle_map.py does

    lower, upper = bounds
    search = sobits_intball2_gnc_cpp.GlobalAStar(
        grid, list(np.asarray(lower, dtype=float)), list(np.asarray(upper, dtype=float)),
        ASTAR_CONNECTIVITY, 0, ASTAR_NARROW_MARGIN_M)
    result, path = search.search(list(np.asarray(start, dtype=float)),
                                 list(np.asarray(goal, dtype=float)))
    _raise_for(result, sobits_intball2_gnc_cpp.GlobalAStar.Result, search)
    # The search works in cell centres; the ends are what the caller asked for.
    path = [np.asarray(start, dtype=float)] + [np.asarray(p, dtype=float) for p in path[1:-1]] \
        + [np.asarray(goal, dtype=float)]
    return [np.asarray(point, dtype=float) for point in shortcut_path(path, grid, bounds)]


def _raise_for(result, codes, search):
    if result == codes.Success:
        return
    if result == codes.StartOccupied:
        raise StartOccupiedError("the start is in the inflated grid")
    if result == codes.GoalOccupied:
        raise GoalOccupiedError("the goal is in the inflated grid")
    if result == codes.OutOfBounds:
        raise SearchBoundsError("the start or the goal is outside the search bounds")
    if result == codes.ExceededMaxExpansions:
        # The cap is the box's own cell count, which a search cannot reach, so
        # this means the cap was overridden or the box was mis-sized.
        raise AStarPlanningError(
            "A* exceeded max_expansions (%d)" % search.max_expansions)
    raise AStarPlanningError("A* found no path inside search_bounds")


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
