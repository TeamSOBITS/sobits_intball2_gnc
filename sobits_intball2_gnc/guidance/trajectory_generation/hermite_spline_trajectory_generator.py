#!/usr/bin/env python3
"""Cubic Hermite spline trajectory generator (ROS-agnostic, pure).

Degraded-but-usable stand-in for :mod:`min_snap_trajectory_generator` while
that module's core (a separate implementer's KKT solve, not expected to land
soon) is unavailable. Only guarantees C1 continuity (position + velocity)
across segment boundaries, not the full snap-minimizing smoothness of a real
min-snap solution -- degree-3 per segment (4 of the 8 coefficient slots used,
the rest zero) instead of degree-7. This is enough to exercise the rest of
the Guidance pipeline (:mod:`sobits_intball2_gnc.guidance.segment_time`,
:class:`~sobits_intball2_gnc.guidance.utils.trajectory.Trajectory`,
``scripts/plot_trajectory.py``) end-to-end today; swap in
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.min_snap_trajectory_generator.MinSnapTrajectoryGenerator`
once ``min_snap.py``'s core lands (same
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
contract, so no caller changes needed).

Tangents at interior waypoints use a Catmull-Rom-style estimate (weighted by
the two adjacent segment durations); start/end tangents are zero, matching
this project's convention that a trajectory begins/ends at rest (see
:mod:`sobits_intball2_gnc.guidance.utils.trajectory` module docstring on
terminal behavior, and ``trajectory_force_duration_investigation.md`` 6-3
節's note on why a moving reference should not start at nonzero ``v``
unannounced).
"""
import numpy as np


class HermiteSplineTrajectoryGenerator:
    """Piecewise cubic Hermite interpolation through the given waypoints."""

    def generate(self, waypoints, segment_times):
        waypoints = np.asarray(waypoints, dtype=float)
        segment_times = np.asarray(segment_times, dtype=float)

        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (n_waypoints, 3)")
        n_waypoints = waypoints.shape[0]
        if n_waypoints < 2:
            raise ValueError("need at least 2 waypoints to form a segment")
        if segment_times.shape != (n_waypoints - 1,):
            raise ValueError("segment_times must have shape (n_waypoints - 1,)")
        if np.any(segment_times <= 0.0):
            raise ValueError("segment_times must all be > 0")

        tangents = self._estimate_tangents(waypoints, segment_times)

        n_segments = n_waypoints - 1
        coeffs = np.zeros((n_segments, 3, 8))
        for i in range(n_segments):
            p0, p1 = waypoints[i], waypoints[i + 1]
            m0, m1 = tangents[i], tangents[i + 1]
            T = segment_times[i]
            coeffs[i, :, 0] = p0
            coeffs[i, :, 1] = m0
            coeffs[i, :, 2] = (-3.0 * p0 + 3.0 * p1 - 2.0 * T * m0 - T * m1) / T ** 2
            coeffs[i, :, 3] = (2.0 * p0 - 2.0 * p1 + T * m0 + T * m1) / T ** 3
        return coeffs

    def _estimate_tangents(self, waypoints, segment_times):
        """Return per-waypoint velocity tangents (shape ``(n_waypoints, 3)``).

        Endpoints are clamped to zero (start/end at rest). Interior waypoint
        ``j`` uses a Catmull-Rom-style estimate weighted by its two adjacent
        segment durations, so a long segment on one side doesn't dominate a
        short one on the other.
        """
        n_waypoints = waypoints.shape[0]
        tangents = np.zeros_like(waypoints)
        for j in range(1, n_waypoints - 1):
            t_in, t_out = segment_times[j - 1], segment_times[j]
            tangents[j] = (waypoints[j + 1] - waypoints[j - 1]) / (t_in + t_out)
        return tangents
