#!/usr/bin/env python3
"""Minimum-snap position trajectory with a fixed attitude (ROS-agnostic).

``trajectory_tracking_mode="static_min_snap"`` (``motion_profile: min_snap``)
の軌道．goal を受けた時点で一度だけ解き，
:class:`~sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker.StaticTrajectoryTracker`
でそのまま最後まで追う（``static_minco`` と同じ形）．

位置は Mellinger & Kumar (2011) の多区間 minimum snap
（:mod:`~sobits_intball2_gnc.guidance.trajectory.generation.min_snap_trajectory_generator`）:
    1. 区間時間を区間の長さに比例して決める（``T_i = |Δp_i| / v_max``）
    2. ``optimize`` なら合計時間を変えずに比を最適化（論文 V-C (14)）
    3. 全区間を同じ倍率で伸び縮みさせ，``max|v| = v_max`` か
       ``max|a| = a_max`` のどちらかちょうどにする（論文 V-A-1）
    4. その区間時間で QP（論文 (10)(11)）を解く

姿勢は移動中ずっと ``q0``（出発時の姿勢）のまま．``face_travel`` には対応しない
（到着後の姿勢合わせは ``align_at_arrival`` が受け持つ）．

制約は ``|v| <= v_max`` と ``|a| <= a_max`` のスカラー上限だけで，
TOPP-RA・MINCO のような wrench envelope（ファンの力・トルクの範囲）は考えない．
"""
import time

import numpy as np

from sobits_intball2_gnc.guidance.trajectory.generation.min_snap_trajectory_generator import (
    plan_segment_times,
    solve_min_snap,
)

MIN_SNAP_ORDERS = frozenset({2, 3, 4})


class MinSnapInfeasibleError(ValueError):
    """The minimum-snap trajectory could not be built (e.g. two consecutive
    waypoints coincide, or the linear system is singular)."""


class MinSnapTrajectory:
    """Minimum-snap position path, attitude held at ``q0``; see module docstring.

    Satisfies the duck-typed interface ``StaticTrajectoryTracker`` needs
    (``sample(t) -> (p, v, a, q)``, ``sample_body_angular(t)`` and
    ``global_total_duration``), same as ``ToppraTrajectory``/``MincoTrajectory``.

    Args:
        position_waypoints: ``(n, 3)`` reference-frame positions, ``n >= 2``.
        q0: departure attitude (shape ``(4,)``), held for the whole trajectory.
        max_vel: ``|v|`` limit [m/s].
        max_accel: ``|a|`` limit [m/s^2].
        order: derivative order ``k`` whose squared integral is minimized
            (4 = snap, the paper's choice; 3 = jerk; 2 = acceleration).
        optimize_segment_times: optimize the ratio of segment times
            (paper V-C (14)) before scaling to the limits.

    Raises:
        MinSnapInfeasibleError: if no trajectory can be built.
    """

    def __init__(self, position_waypoints, q0, max_vel, max_accel, order=4,
                 optimize_segment_times=True):
        if order not in MIN_SNAP_ORDERS:
            raise ValueError("order must be one of %s" % sorted(MIN_SNAP_ORDERS))
        if not (max_vel > 0.0 and max_accel > 0.0):
            raise ValueError("max_vel and max_accel must be > 0")
        waypoints = np.asarray(position_waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] != 3:
            raise ValueError("position_waypoints must have shape (n >= 2, 3)")
        self._q0 = np.asarray(q0, dtype=float)
        self.num_waypoints = waypoints.shape[0]

        t0 = time.perf_counter()
        try:
            segment_times = plan_segment_times(
                waypoints, max_vel, max_accel, k=order, optimize=optimize_segment_times)
            self._traj = solve_min_snap(waypoints, segment_times, k=order)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise MinSnapInfeasibleError(str(exc)) from exc
        self.solve_wall_seconds = time.perf_counter() - t0
        self.segment_times = np.asarray(segment_times, dtype=float)
        if not np.all(np.isfinite(self.segment_times)):
            raise MinSnapInfeasibleError("non-finite segment times")

    @property
    def global_total_duration(self):
        return self._traj.duration

    def sample(self, t):
        """Return ``(p, v, a, q)`` at time ``t`` (clamped to the duration,
        holding the final state past it); ``q`` is always ``q0``."""
        t = float(t)
        p = self._traj.evaluate(t, 0)
        v = self._traj.evaluate(t, 1)
        a = self._traj.evaluate(t, 2)
        return p, v, a, self._q0.copy()

    def sample_body_angular(self, t):
        """Return body-frame ``(omega, omega_dot)``: always zero (fixed attitude)."""
        return np.zeros(3), np.zeros(3)
