"""Collision-checked polyline shortcutting for offline global-planner tests.

The predicate deliberately matches occupancy-mode RRT: all sample points on an
edge must be inside the explicit search box and free in the inflated map.
This module has no ROS dependency and is not connected to guidance yet.
"""
import numpy as np


def point_is_free(point, grid, search_bounds):
    """Return whether a world-coordinate point is in bounds and unoccupied."""
    point = np.asarray(point, dtype=float)
    lower, upper = (np.asarray(v, dtype=float) for v in search_bounds)
    return (point.shape == (3,) and np.all(point >= lower) and np.all(point <= upper)
            and not grid.inflated_occupied(list(point)))


def segment_is_free(start, end, grid, search_bounds, collision_step=None):
    """Check a segment at no more than half a voxel by default."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        return False
    step = 0.5 * float(grid.resolution) if collision_step is None else float(collision_step)
    if step <= 0.0:
        raise ValueError("collision_step must be positive")
    count = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
    return all(point_is_free(start + (end - start) * k / count, grid, search_bounds)
               for k in range(count + 1))


def shortcut_path(path, grid, search_bounds, collision_step=None):
    """Greedily replace each path prefix with its farthest collision-free edge."""
    points = [np.asarray(point, dtype=float) for point in path]
    if len(points) < 2:
        raise ValueError("path must contain at least start and goal")
    if not all(point_is_free(point, grid, search_bounds) for point in points):
        raise ValueError("path contains a point outside search_bounds or in inflated occupancy")

    result = [points[0]]
    current = 0
    while current < len(points) - 1:
        next_index = current + 1
        for candidate in range(len(points) - 1, current, -1):
            if segment_is_free(points[current], points[candidate], grid, search_bounds,
                               collision_step):
                next_index = candidate
                break
        result.append(points[next_index])
        current = next_index
    return result
