#!/usr/bin/env python3
"""Common interface for trajectory (coefficient) generators (ROS-agnostic, pure).

A trajectory generator turns a waypoint list + per-segment durations into the
polynomial coefficients that
:class:`sobits_intball2_gnc.guidance.utils.trajectory.Trajectory` samples
(``docs/min_snap_interface_contract.md`` 2 節's ``coeffs`` layout). Package-ized
per ``docs/architecture_guidelines.md`` 2 節: a min-snap solver (Mellinger &
Kumar 2011, core solve not implemented -- 2026-08-24 decision, see
:mod:`sobits_intball2_gnc.guidance.trajectory_generation.min_snap_trajectory_generator`)
and a Hermite-spline stand-in (this package's degraded-but-usable, actually
wired-in placeholder) are two concrete, named candidates.

Pure function-like (no shared internal state between calls), so this follows
``docs/architecture_guidelines.md`` 4 節's guidance to prefer
``typing.Protocol`` over ``ABC`` here.
"""
from typing import Protocol


class BaseTrajectoryGenerator(Protocol):
    """Shared contract for trajectory generators; see module docstring."""

    def generate(self, waypoints, segment_times, v0=None):
        """Return ``coeffs``: shape ``(n_segments, 3, 8)``, ascending-power
        per-axis polynomial coefficients evaluated in local segment time
        ``tau`` (``docs/min_snap_interface_contract.md`` 2 節/3 節) --
        exactly what
        :class:`~sobits_intball2_gnc.guidance.utils.trajectory.Trajectory`
        expects as its ``coeffs`` argument.

        Args:
            waypoints: ``(n_waypoints, 3)`` reference-frame position array.
            segment_times: ``(n_waypoints - 1,)`` per-segment durations [s],
                every element ``> 0`` (e.g. from a
                :class:`~sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator.BaseSegmentTimeAllocator`,
                and matching the ``v0`` passed there -- see below).
            v0: optional start velocity vector, shape ``(3,)``, at
                ``waypoints[0]`` (e.g. the vehicle's real TF-estimated
                velocity when re-planning mid-flight, see
                ``docs/guidance_realtime_replanning_design.md``). ``None``
                (default) reproduces the original rest-to-rest start (the
                trajectory's first derivative is ``0`` at ``t=0``). Only the
                overall trajectory's start is affected -- interior waypoint
                tangents (e.g. this class's Catmull-Rom estimate) are
                unaffected.

        Raises:
            ValueError: if ``waypoints``/``segment_times`` are inconsistent
                (fewer than 2 waypoints, wrong length, non-positive
                duration), or if ``v0`` is given with the wrong shape.
        """
        ...
