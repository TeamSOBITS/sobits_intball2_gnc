"""ROS-independent A* reference path and FIRI safe-corridor construction.

This module owns global-route decisions only.  It deliberately does not create
MINCO trajectories or local pieces; those belong to ``local_planner``.  The
split keeps a future global planner (RRT, map server, etc.) from depending on
the tracker implementation.
"""
from dataclasses import dataclass

import numpy as np

from sobits_intball2_gnc.guidance.search.path_shortcut import segment_is_free
from sobits_intball2_gnc.guidance.search.reference_route import plan_reference_route


@dataclass(frozen=True)
class GlobalCorridorPlan:
    """An occupancy-safe reference polyline and FIRI half-spaces per segment."""

    route: np.ndarray
    planes: tuple[float, ...]
    bounds: tuple[np.ndarray, np.ndarray]

    @classmethod
    def astar_firi(cls, start, goal, grid, obstacle_points, bounds, *,
                   inflation_m, local_margin_m=0.75):
        """Build an A*6 reference and its bounded FIRI corridor from one snapshot.

        ``obstacle_points`` must represent the same static and sensed obstacle
        snapshot as ``grid``.  The FIRI bounding box is limited to each route
        segment plus ``local_margin_m``; it is therefore safe to omit points
        outside that box because the resulting corridor cannot enter it.
        """
        if local_margin_m <= 0.0:
            raise ValueError("local_margin_m must be positive")
        import sobits_intball2_gnc_cpp as cpp

        route = plan_reference_route(start, goal, grid, bounds)
        points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
        planes = cpp.firi_corridor_planes(
            np.asarray(route, dtype=float).ravel().tolist(), points.ravel().tolist(),
            float(inflation_m), list(bounds[0]), list(bounds[1]), float(local_margin_m),
        )
        return cls(np.asarray(route, dtype=float), tuple(float(value) for value in planes),
                   tuple(np.asarray(bound, dtype=float) for bound in bounds))

    def forward_is_free(self, current, start_segment, grid):
        """Check only the untraversed A* reference, not a smoothing trajectory."""
        if not 0 <= start_segment < len(self.route) - 1:
            raise ValueError("start_segment outside route")
        points = [np.asarray(current, dtype=float), *self.route[start_segment + 1:]]
        return all(segment_is_free(a, b, grid, self.bounds,
                                   collision_step=0.5 * grid.resolution)
                   for a, b in zip(points[:-1], points[1:]))
