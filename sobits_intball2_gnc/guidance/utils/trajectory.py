#!/usr/bin/env python3
"""Sampleable trajectory wrapper (ROS-agnostic).

Holds the per-segment polynomial coefficients produced by
:mod:`sobits_intball2_gnc.guidance.utils.min_snap` (see
``docs/min_snap_interface_contract.md`` for the exact data layout) and
exposes ``sample(t) -> (p, v, a, q_des)`` for a control-layer consumer (Phase
3a/3b in ``docs/main_plan.md``). ``q_des`` is computed internally via
:mod:`sobits_intball2_gnc.guidance.utils.attitude_reference`.

``sample(t)`` converts the global time ``t`` into a segment index and a
segment-local ``tau`` (``min_snap_interface_contract.md`` 3 節: coefficients
are defined per-segment in local time, not global time) before evaluating
the polynomial via :mod:`sobits_intball2_gnc.guidance.utils.polynomial`.

Past the trajectory's total duration, ``sample(t)`` holds the final waypoint
with zero velocity/acceleration; ``q_des`` then holds its last value too,
since :func:`attitude_reference.compute_q_des` already treats near-zero
``v_des`` as "below the low-speed threshold" (no separate terminal-state
handling needed for attitude).
"""
import numpy as np

from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    IDENTITY_QUAT,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector


class Trajectory:
    """Time-sampleable min-snap trajectory.

    Args:
        waypoints: shape ``(n_waypoints, 3)``, reference-frame positions
            (same list ``min_snap.solve_min_snap()`` was given).
        segment_times: shape ``(n_segments,)`` per-segment durations [s],
            ``n_segments == n_waypoints - 1``.
        coeffs: shape ``(n_segments, 3, 8)``, ``min_snap.solve_min_snap()``'s
            output (ascending-power per-axis polynomial coefficients).
        attitude_speed_threshold: passed through to
            :func:`attitude_reference.compute_q_des`.
        forward_axis: passed through to
            :func:`attitude_reference.compute_q_des`.
        max_angular_rate: passed through to
            :func:`attitude_reference.compute_q_des` as its ``max_angular_rate``
            [rad/s]. ``None`` (default) disables rate-limiting, matching prior
            behavior. See that function's docstring for why an unlimited
            ``q_des`` can produce a large, slow-to-clear tracking error.
        initial_q_des: seeds ``sample(0)``'s ``prev_q_des`` (e.g. the
            vehicle's actual current attitude from TF), so the first sample
            doesn't fall back to the identity quaternion -- a min-jerk
            trajectory has zero velocity at ``t=0`` by construction, so
            without this every first call would otherwise hit the "speed
            below threshold, no previous q_des" case (see
            docs/trajectory_force_duration_investigation.md 6-3).
        face_travel: when ``False``, ``sample(t)`` never calls
            :func:`attitude_reference.compute_q_des` and instead holds
            ``q_des`` fixed at ``initial_q_des`` (or identity if not given)
            for the entire trajectory -- for moves where facing the direction
            of travel is undesired (e.g. a fast transit with no particular
            heading requirement), so translation isn't gated by an attitude
            reference or a pre-alignment step at all. Default ``True``
            preserves the original "face direction of travel" behavior.
    """

    def __init__(
        self,
        waypoints,
        segment_times,
        coeffs,
        attitude_speed_threshold=0.02,
        forward_axis=(1.0, 0.0, 0.0),
        max_angular_rate=None,
        initial_q_des=None,
        face_travel=True,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.segment_times = np.asarray(segment_times, dtype=float)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self._cum_times = np.concatenate([[0.0], np.cumsum(self.segment_times)])
        self.total_duration = float(self._cum_times[-1])
        self._attitude_speed_threshold = float(attitude_speed_threshold)
        self._forward_axis = forward_axis
        self._max_angular_rate = max_angular_rate
        self._face_travel = bool(face_travel)
        self._last_q_des = (
            IDENTITY_QUAT.copy() if initial_q_des is None
            else np.asarray(initial_q_des, dtype=float)
        )
        self._last_sample_t = None

    def sample(self, t):
        """Return ``(p, v, a, q_des)`` at global time ``t`` (clamped to ``t >= 0``)."""
        t = max(float(t), 0.0)

        if t >= self.total_duration:
            p = self.waypoints[-1].copy()
            v = np.zeros(3)
            a = np.zeros(3)
        else:
            seg_idx = self._segment_index(t)
            tau = t - self._cum_times[seg_idx]
            seg_coeffs = self.coeffs[seg_idx]
            p = evaluate_vector(seg_coeffs, tau, order=0)
            v = evaluate_vector(seg_coeffs, tau, order=1)
            a = evaluate_vector(seg_coeffs, tau, order=2)

        if not self._face_travel:
            return p, v, a, self._last_q_des

        dt = None if self._last_sample_t is None else t - self._last_sample_t
        self._last_sample_t = t
        if dt is not None and dt <= 0.0:
            # Non-advancing or out-of-order sample() call: nothing to rate-limit
            # against (there's no elapsed time to have moved during).
            dt = None

        q_des = compute_q_des(
            v, self._last_q_des, self._attitude_speed_threshold, self._forward_axis,
            dt=dt, max_angular_rate=self._max_angular_rate,
        )
        self._last_q_des = q_des
        return p, v, a, q_des

    def _segment_index(self, t):
        idx = int(np.searchsorted(self._cum_times, t, side="right")) - 1
        return min(max(idx, 0), len(self.segment_times) - 1)
