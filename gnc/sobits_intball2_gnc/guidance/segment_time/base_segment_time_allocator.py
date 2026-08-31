#!/usr/bin/env python3
"""Common interface for segment-time allocators (ROS-agnostic, pure).

A segment-time allocator turns a waypoint list into the per-segment duration
array (``segment_times``) that a trajectory generator's ``generate(waypoints,
segment_times)`` expects (see
:mod:`sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator`).
Kept as a separate stage from coefficient generation itself so the allocation
policy can be swapped independently (``docs/main_plan.md`` Phase 3b's
torque-budget problem is expected to be addressed here, by giving sharp turns
more time, rather than by re-deriving control gains).

Package-ized per ``docs/architecture_guidelines.md`` 2 節: a distance/curvature
heuristic and Mellinger & Kumar (2011) V-C's gradient-descent "optimal segment
times" (``docs/minimum_snap/minimum_snap.md``) are two concrete, named
candidates, so this lives in its own package with a shared base class rather
than a flat ``utils/segment_time_allocator.py`` file.
"""
from abc import ABC, abstractmethod


class SegmentTimeInfeasibleError(ValueError):
    """No segment time satisfies both the acceleration budget and the
    overshoot-avoidance bound for a ``v0``-aware first segment (see
    ``docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md`` 3 節).

    Callers doing real-time replanning are expected to catch this and fall
    back to a static trajectory / stop replanning rather than treat it as a
    generic ``ValueError`` bug -- it signals a genuine kinematic dead end
    (the vehicle is too close to the target, at too high a residual speed,
    for this trajectory representation to reach it without either violating
    the acceleration budget or overshooting), not a caller mistake.
    """


class BaseSegmentTimeAllocator(ABC):
    """Shared contract for segment-time allocators; see module docstring."""

    @abstractmethod
    def allocate(self, waypoints, v0=None):
        """Return ``segment_times``: a 1-D ``numpy.ndarray`` of ``float``,
        shape ``(n_waypoints - 1,)``, units seconds, every element strictly
        positive (``> 0``) -- this is exactly the ``segment_times`` argument
        a :class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
        expects, so an implementation must never return a zero/negative entry
        itself.

        Args:
            waypoints: ``(n_waypoints, 3)`` reference-frame position array
                (same convention as the trajectory generators).
            v0: optional start velocity vector, shape ``(3,)``, at
                ``waypoints[0]`` (e.g. the vehicle's real TF-estimated
                velocity when re-planning mid-flight, see
                ``docs/guidance_realtime_replanning_design.md``). ``None``
                (default) reproduces the original rest-to-rest allocation.
                Only the first segment's timing is affected; interior
                waypoints are unaffected. An implementation that cannot
                honor a nonzero ``v0`` (e.g. one with no acceleration-budget
                concept) may raise ``ValueError`` if given one.

        Raises:
            ValueError: if ``waypoints`` has fewer than 2 rows (no segment
                can be formed), or if ``v0`` is given but this
                implementation cannot use it (e.g. no acceleration budget
                configured).
            SegmentTimeInfeasibleError: if ``v0`` is given and no segment
                time exists that satisfies both the acceleration budget and
                the overshoot-avoidance bound for the first segment.
        """
        raise NotImplementedError
