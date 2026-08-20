#!/usr/bin/env python3
"""Common interface for segment-time allocators (ROS-agnostic, pure).

A segment-time allocator turns a waypoint list into the per-segment duration
array (``segment_times``) that :func:`sobits_intball2_gnc.guidance.utils.min_snap.solve_min_snap`
expects (see ``docs/min_snap_interface_contract.md`` 2 節/6 節). Kept as a
separate stage from the min-snap coefficient solve itself so the allocation
policy can be swapped without touching ``min_snap.py`` (``docs/main_plan.md``
Phase 3b's torque-budget problem is expected to be addressed here, by giving
sharp turns more time, rather than by re-deriving control gains).

Package-ized per ``docs/architecture_guidelines.md`` 2 節: a distance/curvature
heuristic and Mellinger & Kumar (2011) V-C's gradient-descent "optimal segment
times" (``docs/minimum_snap/minimum_snap.md``) are two concrete, named
candidates, so this lives in its own package with a shared base class rather
than a flat ``utils/segment_time_allocator.py`` file.
"""
from abc import ABC, abstractmethod


class BaseSegmentTimeAllocator(ABC):
    """Shared contract for segment-time allocators; see module docstring."""

    @abstractmethod
    def allocate(self, waypoints):
        """Return ``segment_times``: a 1-D ``numpy.ndarray`` of ``float``,
        shape ``(n_waypoints - 1,)``, units seconds, every element strictly
        positive (``> 0``) -- this is exactly the ``segment_times`` argument
        :func:`~sobits_intball2_gnc.guidance.utils.min_snap.solve_min_snap`
        expects (``docs/min_snap_interface_contract.md`` 2 節: "0以下の値がある
        -> ValueError"), so an implementation must never return a
        zero/negative entry itself.

        Args:
            waypoints: ``(n_waypoints, 3)`` reference-frame position array
                (same convention as ``solve_min_snap``).

        Raises:
            ValueError: if ``waypoints`` has fewer than 2 rows (no segment
                can be formed).
        """
        raise NotImplementedError
