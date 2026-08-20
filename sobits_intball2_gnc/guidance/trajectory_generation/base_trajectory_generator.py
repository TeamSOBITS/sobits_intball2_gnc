#!/usr/bin/env python3
"""Common interface for trajectory (coefficient) generators (ROS-agnostic, pure).

A trajectory generator turns a waypoint list + per-segment durations into the
polynomial coefficients that
:class:`sobits_intball2_gnc.guidance.utils.trajectory.Trajectory` samples
(``docs/min_snap_interface_contract.md`` 2 節's ``coeffs`` layout). Package-ized
per ``docs/architecture_guidelines.md`` 2 節: the real min-snap solver
(Mellinger & Kumar 2011, owned by a separate implementer, not expected to
land soon) and a Hermite-spline stand-in (this package's degraded-but-usable
placeholder) are two concrete, named candidates.

Pure function-like (no shared internal state between calls), so this follows
``docs/architecture_guidelines.md`` 4 節's guidance to prefer
``typing.Protocol`` over ``ABC`` here.

**Important**: this package does not change
:func:`sobits_intball2_gnc.guidance.utils.min_snap.solve_min_snap`'s already
-agreed signature (``docs/min_snap_interface_contract.md`` 2 節, negotiated
with that module's separate implementer). ``MinSnapTrajectoryGenerator`` in
this package is a thin adapter around that function, not a replacement for
it.
"""
from typing import Protocol


class BaseTrajectoryGenerator(Protocol):
    """Shared contract for trajectory generators; see module docstring."""

    def generate(self, waypoints, segment_times):
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
                :class:`~sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator.BaseSegmentTimeAllocator`).

        Raises:
            ValueError: if ``waypoints``/``segment_times`` are inconsistent
                (fewer than 2 waypoints, wrong length, non-positive duration).
        """
        ...
