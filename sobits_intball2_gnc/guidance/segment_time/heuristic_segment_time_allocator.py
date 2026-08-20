#!/usr/bin/env python3
"""Distance + turn-angle heuristic segment-time allocator (ROS-agnostic, pure).

Baseline candidate for ``docs/architecture_guidelines.md``'s 2-candidate
threshold (see :mod:`base_segment_time_allocator`). Allocates ``distance /
target_speed`` per segment, then adds extra time around interior waypoints in
proportion to the deviation angle between the incoming and outgoing segment
directions (split evenly between the two adjacent segments) -- a sharper turn
gets more time. This directly targets the mechanism documented in
``docs/main_plan.md`` Phase 3b and
``docs/archive/achieved/trajectory_force_duration_investigation.md``: required
peak angular acceleration (and thus torque) scales as ``1/T**2``, so a sharp
corner that would otherwise saturate the attitude torque budget can instead be
given more time, independent of any control-gain retuning.

Not a substitute for Mellinger & Kumar (2011) V-C's gradient-descent "optimal
segment times" (which minimizes the actual min-snap cost via
:func:`~sobits_intball2_gnc.guidance.utils.min_snap.solve_min_snap`, see
:mod:`optimal_segment_time_allocator`) -- this is a cheap, min-snap-independent
first pass that can run before that module exists.
"""
import numpy as np

from sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator import (
    BaseSegmentTimeAllocator,
)

DEFAULT_MIN_SEGMENT_TIME = 1e-3


class HeuristicSegmentTimeAllocator(BaseSegmentTimeAllocator):
    """Distance/target-speed allocation with a turn-angle time boost.

    Plain ``distance / target_speed`` assumes the vehicle cruises the whole
    segment at ``target_speed``. The trajectory this feeds
    (``HermiteSplineTrajectoryGenerator``) actually starts and ends each
    segment at rest, so over the same duration it must accelerate up past
    ``target_speed`` and back down again -- its peak speed and acceleration
    are higher than a naive cruise estimate assumes. With ``max_accel`` unset
    this class doesn't know the vehicle's force budget and can allocate less
    time than the required peak acceleration, causing the setpoint to reach
    the target long before the real vehicle can track it (observed 2026-08-20,
    see docs/guidance_move_to_debug_2026-08-20.md). Passing ``max_accel``
    (``trajectory_controller.max_force / mass``) raises each segment's time,
    when needed, to what a symmetric trapezoidal (accel/cruise/decel, or
    triangular for short segments) velocity profile at that acceleration and
    cruise cap would take -- a closed-form minimum-time estimate, not a
    trajectory-shape optimization (that remains min_snap.py's separate,
    not-yet-implemented scope; this only affects segment *duration*, the
    existing Hermite spline still generates the actual shape).
    """

    def __init__(self, target_speed, angle_time_gain=0.0,
                 min_segment_time=DEFAULT_MIN_SEGMENT_TIME, max_accel=None):
        """
        Args:
            target_speed: assumed cruise speed [m/s], must be > 0.
            angle_time_gain: extra seconds allocated per radian of deviation
                angle at an interior waypoint, split evenly between the two
                segments adjacent to that waypoint. 0.0 (default) disables
                the turn-angle boost and reduces this to plain
                distance/target_speed allocation.
            min_segment_time: floor applied to every segment's final time [s],
                so a degenerate (near-zero distance, or heavily-discounted)
                segment never reaches ``solve_min_snap`` with a
                non-positive duration.
            max_accel: vehicle's achievable acceleration [m/s^2] (per axis,
                magnitude), used to raise a segment's time to the trapezoidal-
                profile minimum when the naive distance/target_speed estimate
                would demand more acceleration than this. ``None`` (default)
                disables this check, reproducing the old pure
                distance/target_speed behavior.
        """
        if target_speed <= 0.0:
            raise ValueError("target_speed must be > 0")
        if angle_time_gain < 0.0:
            raise ValueError("angle_time_gain must be >= 0")
        if min_segment_time <= 0.0:
            raise ValueError("min_segment_time must be > 0")
        if max_accel is not None and max_accel <= 0.0:
            raise ValueError("max_accel must be > 0 when given")
        self.target_speed = float(target_speed)
        self.angle_time_gain = float(angle_time_gain)
        self.min_segment_time = float(min_segment_time)
        self.max_accel = None if max_accel is None else float(max_accel)

    def allocate(self, waypoints):
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (n_waypoints, 3)")
        n_waypoints = waypoints.shape[0]
        if n_waypoints < 2:
            raise ValueError("need at least 2 waypoints to form a segment")

        deltas = np.diff(waypoints, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        segment_times = distances / self.target_speed

        if self.max_accel is not None:
            segment_times = np.maximum(
                segment_times,
                self._trapezoidal_time(distances, self.target_speed, self.max_accel),
            )

        if self.angle_time_gain > 0.0:
            segment_times = segment_times + self._turn_angle_boost(deltas)

        return np.maximum(segment_times, self.min_segment_time)

    @staticmethod
    def _trapezoidal_time(distances, v_cap, a_max):
        """Minimum time to cover each distance from rest to rest.

        Symmetric accel/cruise/decel profile capped at ``v_cap``, or a
        triangular (never reaching ``v_cap``) profile for short distances.
        """
        d_accel = v_cap * v_cap / (2.0 * a_max)
        reaches_cruise = distances >= 2.0 * d_accel
        trapezoid = 2.0 * (v_cap / a_max) + (distances - 2.0 * d_accel) / v_cap
        v_peak = np.sqrt(a_max * distances)
        triangle = 2.0 * v_peak / a_max
        return np.where(reaches_cruise, trapezoid, triangle)

    def _turn_angle_boost(self, deltas):
        """Extra time per segment from deviation angles at interior waypoints.

        ``deltas[i]`` is the direction of segment ``i`` (waypoint ``i`` ->
        ``i+1``). At interior waypoint ``j`` (``1 <= j <= n_segments - 1``),
        the deviation angle is the angle between incoming segment ``j - 1``
        and outgoing segment ``j``; ``angle_time_gain * angle`` is split
        evenly onto both.
        """
        boost = np.zeros(len(deltas))
        norms = np.linalg.norm(deltas, axis=1)
        for j in range(1, len(deltas)):
            n_in, n_out = norms[j - 1], norms[j]
            if n_in < 1e-12 or n_out < 1e-12:
                continue
            cos_angle = np.dot(deltas[j - 1], deltas[j]) / (n_in * n_out)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            extra = self.angle_time_gain * angle / 2.0
            boost[j - 1] += extra
            boost[j] += extra
        return boost
