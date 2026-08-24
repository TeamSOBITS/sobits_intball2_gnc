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
segment times" (which would minimize the actual min-snap cost, see
:mod:`optimal_segment_time_allocator`) -- this is a cheap, min-snap-independent
first pass, and the only one usable now that min-snap's core solve won't be
implemented (see
:mod:`sobits_intball2_gnc.guidance.trajectory_generation.min_snap_trajectory_generator`).
"""
import numpy as np

from sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator import (
    BaseSegmentTimeAllocator,
    SegmentTimeInfeasibleError,
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
    trajectory-shape optimization (that would have been min-snap's separate
    scope, but its core solve won't be implemented; this only affects segment
    *duration*, the existing Hermite spline still generates the actual
    shape).

    ``allocate()``'s optional ``v0`` handles the case this rest-to-rest
    model does not cover at all: a first segment that starts at a nonzero
    real velocity (real-time re-planning mid-flight, see
    ``docs/guidance_realtime_replanning_design.md``). That case is computed
    by a separate, exact method (:meth:`_v0_aware_bounds`) derived directly
    from the Hermite cubic polynomial, not this trapezoidal model --
    reusing this model for ``v0 != 0`` was tried and shown to be actively
    wrong (larger T can *increase* overshoot risk once the start velocity is
    nonzero, the opposite of the rest-to-rest case; see
    ``docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md`` 2 節).
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
                segment never reaches the trajectory generator with a
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

    def allocate(self, waypoints, v0=None):
        """See ``BaseSegmentTimeAllocator.allocate``'s docstring for ``v0``'s
        contract. ``v0`` only affects the first segment's time -- it is
        computed from the *exact* cubic-Hermite polynomial that
        ``HermiteSplineTrajectoryGenerator.generate(..., v0=v0)`` would build
        for that segment (``m0=v0, m1=0``), not this class's rest-to-rest
        ``_trapezoidal_time`` model, which does not apply when the start
        velocity is nonzero (see
        ``docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md`` 2
        節 for why: larger T does not monotonically improve safety once
        ``v0 != 0``, unlike the rest-to-rest case).
        """
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (n_waypoints, 3)")
        n_waypoints = waypoints.shape[0]
        if n_waypoints < 2:
            raise ValueError("need at least 2 waypoints to form a segment")
        if v0 is not None:
            if self.max_accel is None:
                raise ValueError(
                    "v0 requires max_accel to be set (no acceleration "
                    "budget to size the first segment's time against)"
                )
            v0 = np.asarray(v0, dtype=float)
            if v0.shape != (3,):
                raise ValueError("v0 must have shape (3,)")

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

        segment_times = np.maximum(segment_times, self.min_segment_time)

        if v0 is not None and distances[0] > 1e-9:
            segment_times[0] = self._apply_v0_bound(
                segment_times[0], distances[0], deltas[0], v0, self.max_accel,
            )

        return segment_times

    @staticmethod
    def _apply_v0_bound(naive_t, d, delta0, v0, a_max):
        """Clamp the first segment's naive time into the ``[T_min, T_max]``
        range the ``v0``-aware Hermite cubic requires (exact, see
        :meth:`_v0_aware_bounds`), raising if that range is empty.

        Unlike this class's other adjustments (which only ever raise a
        segment time via ``np.maximum``), this can *lower* ``naive_t`` --
        e.g. the plain ``distance/target_speed`` estimate or the turn-angle
        boost (``docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md``
        4-4 節) could otherwise leave the first segment above ``T_max``,
        which would make the actual trajectory overshoot the target.
        """
        v_parallel = max(float(np.dot(v0, delta0 / d)), 0.0)
        t_min, t_max = HeuristicSegmentTimeAllocator._v0_aware_bounds(
            d, v_parallel, a_max
        )
        if t_min > t_max:
            raise SegmentTimeInfeasibleError(
                "no feasible first-segment time for the given v0: "
                "T_min=%.4fs > T_max=%.4fs (d=%.4fm, v_parallel=%.4fm/s, "
                "max_accel=%.4fm/s^2) -- caller must fall back to a static "
                "trajectory instead of replanning this close/this fast"
                % (t_min, t_max, d, v_parallel, a_max)
            )
        return min(max(naive_t, t_min), t_max)

    @staticmethod
    def _v0_aware_bounds(d, v_parallel, a_max):
        """Return ``(T_min, T_max)`` for a first segment starting at speed
        ``v_parallel`` (component of ``v0`` along the segment direction,
        already clamped to ``>= 0`` by the caller) toward a target at
        distance ``d``, derived directly from the cubic Hermite polynomial
        ``HermiteSplineTrajectoryGenerator`` builds for ``m0=v_parallel,
        m1=0`` (not an idealized trapezoidal motion model -- see
        ``docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md`` 2
        節/3 節 for the full derivation).

        ``T_max = 3d/v_parallel`` is the exact boundary beyond which the
        cubic overshoots ``d`` (``T*v_parallel > 3d``). ``T_min`` is the
        smallest T keeping both endpoint accelerations
        (``a(0)=(6d-4*T*v_parallel)/T**2``, ``a(T)=(2*T*v_parallel-6d)/T**2``,
        both linear-in-tau so their peak is always at one of the two
        endpoints) within ``a_max`` in magnitude. Restricting the search to
        ``T <= T_max`` (which the caller enforces anyway, via ``T_max``)
        makes both endpoint-acceleration curves monotonic over the range
        that matters, so solving each ``|a(endpoint)| = a_max`` as a
        quadratic in T gives the exact answer, not just a bound (verified
        analytically, see the design doc's section 5).

        Provable fact worth calling out (also design doc section 5): for
        ``v_parallel > 0``, ``T_min < T_max`` *always* holds (for any
        ``d > 0, a_max > 0``) -- substituting ``T_max`` into either
        quadratic shows it always overshoots past that quadratic's root, so
        this method's ``T_min`` is never above ``T_max``.
        :meth:`_apply_v0_bound`'s ``SegmentTimeInfeasibleError`` branch is
        therefore expected to be unreachable in practice for ``v_parallel >
        0`` and is kept only as a defensive contract-level guard (e.g.
        against a future change to this derivation, or ``a_max <= 0``
        slipping through). It does NOT mean re-planning arbitrarily close to
        the target is safe in practice: as ``d -> 0`` both bounds converge
        toward ``0`` together, which is exactly the 50Hz-chattering/noise-
        amplification regime documented separately (design doc 6-3/6-4
        節) -- that risk is about a vanishingly small, noise-sensitive
        segment time being technically "feasible", not about no feasible
        time existing. Task #2 (fallback to a static trajectory near the
        target) is about *that* risk, not this one.
        """
        v_parallel = max(v_parallel, 0.0)
        # Numerically stable form of the positive quadratic root (avoids
        # subtracting two nearly-equal large numbers, which the naive
        # (-b+sqrt(disc))/(2a) form does when a_max is small relative to
        # v_parallel -- verified against a mpmath/Decimal high-precision
        # reference during implementation).
        t1 = 12.0 * d / (
            4.0 * v_parallel + np.sqrt(16.0 * v_parallel ** 2 + 24.0 * a_max * d)
        )
        t3 = 12.0 * d / (
            2.0 * v_parallel + np.sqrt(4.0 * v_parallel ** 2 + 24.0 * a_max * d)
        )
        t_max = np.inf if v_parallel <= 0.0 else 3.0 * d / v_parallel
        return max(t1, t3), t_max

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
