"""Position-only polynomial reference through fixed points.

A global trajectory for the replanning local planner that follows a given
polyline instead of the straight chord, so the local target is already clear of
the obstacles that were visible before departure
(docs/minco_astar_reference_global.md 1 and 2). It carries no attitude and no
wrench feasibility: the local solves own those, and this only says where to aim.

The shape and the time law follow SCAN-Planner's reference mode (arXiv
2606.19555): a position-only min-jerk spline through the points, each segment
given dist / speed, with the first and last doubled so the ends start and stop
from rest. Written from the paper's description, not from its code, which is
not under a license we can copy from.
"""

import numpy as np

# Quintic pieces: position, velocity and acceleration fixed at both ends, and
# continuous up to snap across each interior point.
_COEFFS_PER_PIECE = 6


class MinJerkReference:
    """C4 quintic spline through ``points``, at rest at both ends.

    Each segment runs for its chord length over ``speed``, with the first and
    last doubled (once when there is only one). ``sample`` returns
    ``(p, v, a, q0)`` with ``v`` capped at ``speed``; the attitude is always the
    given ``q0``, since nothing here plans rotation.
    """

    def __init__(self, points, speed, q0):
        points = _drop_repeats(np.asarray(points, dtype=float))
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("points must be at least two distinct 3-element points")
        speed = float(speed)
        if not speed > 0.0:
            raise ValueError("speed must be positive")
        times = np.linalg.norm(np.diff(points, axis=0), axis=1) / speed
        times[0] *= 2.0
        if len(times) > 1:
            # With a single piece the first is also the last; doubling twice
            # would stretch a straight goal-rescue global to four times its time.
            times[-1] *= 2.0
        self._times = times
        self._coeffs = _solve_min_jerk(points, times)
        self._starts = np.concatenate([[0.0], np.cumsum(times)])
        self._speed = speed
        self._q0 = np.asarray(q0, dtype=float)
        self.global_total_duration = float(self._starts[-1])
        self.num_waypoints = len(points)
        self.solve_wall_seconds = 0.0

    def sample(self, t):
        t = min(max(float(t), 0.0), self.global_total_duration)
        i = min(int(np.searchsorted(self._starts, t, side="right")) - 1, len(self._times) - 1)
        tau = t - self._starts[i]
        c = self._coeffs[i]
        p = (tau ** np.arange(6)) @ c
        v = (np.arange(1, 6) * tau ** np.arange(5)) @ c[1:]
        a = (np.arange(2, 6) * np.arange(1, 5) * tau ** np.arange(4)) @ c[2:]
        speed = float(np.linalg.norm(v))
        if speed > self._speed:
            # The spline overshoots the nominal speed around the short pieces;
            # the local solve cannot end above its own cap.
            v = v * (self._speed / speed)
        return p, v, a, self._q0


def _drop_repeats(points):
    """``points`` without consecutive duplicates, which would make a zero-length
    piece and a singular system."""
    if points.ndim != 2 or len(points) < 2:
        return points
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    return points[keep]


def _solve_min_jerk(points, times):
    """``(m, 6, 3)`` quintic coefficients, solved for all three axes at once."""
    m = len(times)
    size = _COEFFS_PER_PIECE * m
    a = np.zeros((size, size))
    b = np.zeros((size, 3))
    k = 0
    for d in range(3):  # start: position, at rest
        a[k, 0:6] = _basis(0.0, d)
        b[k] = points[0] if d == 0 else 0.0
        k += 1
    for i in range(m - 1):
        a[k, 6 * i:6 * i + 6] = _basis(times[i], 0)
        b[k] = points[i + 1]
        k += 1
        a[k, 6 * (i + 1):6 * (i + 1) + 6] = _basis(0.0, 0)
        b[k] = points[i + 1]
        k += 1
        for d in range(1, 5):  # velocity through snap continuous
            a[k, 6 * i:6 * i + 6] = _basis(times[i], d)
            a[k, 6 * (i + 1):6 * (i + 1) + 6] = -_basis(0.0, d)
            k += 1
    for d in range(3):  # end: position, at rest
        a[k, 6 * (m - 1):6 * m] = _basis(times[-1], d)
        b[k] = points[-1] if d == 0 else 0.0
        k += 1
    return np.linalg.solve(a, b).reshape(m, 6, 3)


def _basis(t, d):
    """Row of the ``d``-th derivative of ``[1, t, ..., t^5]``."""
    row = np.zeros(6)
    for j in range(d, 6):
        row[j] = np.prod(range(j - d + 1, j + 1)) * t ** (j - d)
    return row
