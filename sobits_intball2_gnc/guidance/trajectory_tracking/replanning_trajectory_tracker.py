#!/usr/bin/env python3
"""Real-time re-planning trajectory tracker (ROS-agnostic, pure).

Implements the "毎tick実TFを見て並進軌道を更新する" idea investigated in
``docs/guidance_realtime_replanning_design.md``, with every safeguard that
investigation's 6 節 found to be load-bearing (a naive always-replan
implementation was shown to be non-terminating, overshoot-prone, and
saturating -- see that section). This class only re-plans on a slower,
explicitly-decided cadence (``docs/archive/achieved/
2026-08-24_replan_rate_design.md``), and permanently stops re-planning
(falling back to whatever the last successful re-plan produced) once either
of two conditions holds (``docs/archive/achieved/
2026-08-24_replanning_distance_fallback_decision.md``):

1. the remaining distance to the target drops below ``distance_fallback_m``
   (re-planning near the target is where 6-3/6-4 節's chattering/noise-
   force-budget failures live), or
2. the live pose is unavailable or its TF stamp is stale (re-planning off a
   frozen/corrupted pose reintroduces the failure modes 6-8 節 documents).

That fallback is a one-way latch for the lifetime of one tracker instance
(one goal) -- it never re-arms re-planning, to avoid a re-plan/static
flip-flop right at the distance boundary.

Time-allocation policy: **re-calculate** the arrival time on every re-plan
(``HeuristicSegmentTimeAllocator.allocate(waypoints, v0=...)`` from the
*current* position/velocity), not fixed -- decided in ``docs/archive/
achieved/2026-08-24_replan_arrival_time_recompute_decision.md``. This is also
why ``max_accel`` is mandatory here (``HeuristicSegmentTimeAllocator``'s
``v0``-aware bound derivation requires it -- ``allocate()`` itself raises
``ValueError`` without it).
"""
import numpy as np

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)

DEFAULT_DISTANCE_FALLBACK_M = 0.3
DEFAULT_REPLAN_EVERY_N_TICKS = 5


class ReplanningTrajectoryTracker:
    """See module docstring.

    Args:
        trajectory: an already-built
            :class:`~sobits_intball2_gnc.guidance.utils.trajectory.Trajectory`
            for the initial ``(p0, p_target)`` leg -- mutated in place via
            ``replace_coeffs`` on each re-plan (never replaced with a new
            instance, so its ``_last_q_des``/``_last_sample_t`` survive, per
            ``docs/archive/achieved/2026-08-24_trajectory_state_carryover_design.md``).
        p_target: target position, shape ``(3,)``.
        pose_fn: callable ``() -> (pos, quat, stamp) | None``, e.g.
            ``tf_client.get_pose`` -- the *only* place this class reads TF.
        tf_fresh_fn: callable ``(stamp) -> bool``, e.g. ``GuidanceExecutor.
            _tf_pose_fresh`` -- kept as an injected callable (rather than this
            class owning its own staleness state) so goal-level TF liveness
            stays centralized in one place, matching how the rest of
            ``GuidanceExecutor`` already tracks it.
        velocity_fn: callable ``() -> VelocityEstimate`` (``.vel``: ``[vx,
            vy, vz]``), e.g. ``VelocityEstimator.get`` (``docs/archive/
            achieved/2026-08-24_guidance_velocity_estimator.md``).
        target_speed: passed to ``HeuristicSegmentTimeAllocator`` on every
            re-plan.
        max_accel: passed to ``HeuristicSegmentTimeAllocator`` on every
            re-plan; required (not ``None``) because ``v0``-aware allocation
            needs it (see module docstring).
        distance_fallback_m: remaining-distance threshold below which
            re-planning permanently stops (default matches the decision doc's
            chosen value, 0.3m -- see module docstring).
        replan_every_n_ticks: number of ``sample()`` calls between re-plan
            attempts (default 5, i.e. 10Hz out of a 50Hz ``sample()`` cadence
            -- see module docstring).

    Raises:
        ValueError: if ``max_accel`` is ``None``.
    """

    def __init__(self, trajectory, p_target, pose_fn, tf_fresh_fn, velocity_fn,
                 target_speed, max_accel,
                 distance_fallback_m=DEFAULT_DISTANCE_FALLBACK_M,
                 replan_every_n_ticks=DEFAULT_REPLAN_EVERY_N_TICKS):
        if max_accel is None:
            raise ValueError(
                "max_accel is required for replanning mode (v0-aware "
                "segment-time allocation needs an acceleration budget to "
                "size the re-planned segment's time against)"
            )
        self._trajectory = trajectory
        self._p_target = np.asarray(p_target, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._velocity_fn = velocity_fn
        self._target_speed = float(target_speed)
        self._max_accel = float(max_accel)
        self._distance_fallback_m = float(distance_fallback_m)
        self._replan_every_n_ticks = int(replan_every_n_ticks)
        self._tick_count = 0
        # One-way latch (module docstring): once True, sample() never
        # attempts another re-plan for the rest of this goal.
        self._fallen_back = False
        # Whether the most recent sample() call also re-planned (i.e.
        # replaced self._trajectory's coefficients) -- read by a caller that
        # wants to keep an RViz preview of the current trajectory in sync
        # with re-planning (e.g. GuidanceExecutor's speed-path Marker), not
        # consumed internally by this class.
        self.last_replan_occurred = False
        # Which condition (module docstring) caused the one-way fallback
        # latch to trip on the most recent sample() call: "tf_stale" | "distance"
        # | None (latch not tripped this tick). Stays populated (not reset)
        # after the tick it tripped on, so a caller logging once on the
        # rising edge (docs/main_plan.md "[C] Controller内部値の可観測性強化")
        # can still read *why* after the fact.
        self.last_fallback_reason = None

    def sample(self, t):
        self.last_replan_occurred = False
        if not self._fallen_back:
            self._tick_count += 1
            if self._tick_count >= self._replan_every_n_ticks:
                self._tick_count = 0
                self._maybe_replan(t)
        return self._trajectory.sample(t)

    @property
    def total_duration(self):
        return self._trajectory.global_total_duration

    @property
    def trajectory(self):
        """The underlying, in-place-mutated ``Trajectory`` -- read-only
        access for a caller that needs the current ``waypoints``/
        ``segment_times``/``coeffs`` after a re-plan (e.g. to rebuild an
        RViz preview), without reaching into this class's private state."""
        return self._trajectory

    def _maybe_replan(self, t_global):
        pose = self._pose_fn()
        if pose is None or not self._tf_fresh_fn(pose[2]):
            # Condition 2 (module docstring): no valid current pose to
            # re-plan from -- freeze on whatever the last successful re-plan
            # (or the initial trajectory, if none yet) already produced,
            # rather than fabricating a position.
            self._fallen_back = True
            self.last_fallback_reason = "tf_stale"
            return

        p_now, _quat, _stamp = pose
        p_now = np.asarray(p_now, dtype=float)
        distance = float(np.linalg.norm(self._p_target - p_now))

        vel_estimate = self._velocity_fn()
        v0 = (
            np.zeros(3) if vel_estimate is None
            else np.asarray(vel_estimate.vel, dtype=float)
        )

        self._replan(p_now, v0, t_global)
        self.last_replan_occurred = True

        if distance < self._distance_fallback_m:
            # Condition 1 (module docstring): this was the last re-plan --
            # the freshly replaced trajectory now carries the vehicle the
            # rest of the way as a static leg.
            self._fallen_back = True
            self.last_fallback_reason = "distance"

    def _replan(self, p_now, v0, t_global):
        waypoints = np.array([p_now, self._p_target])
        segment_times = HeuristicSegmentTimeAllocator(
            target_speed=self._target_speed, max_accel=self._max_accel,
        ).allocate(waypoints, v0=v0)
        coeffs = HermiteSplineTrajectoryGenerator().generate(
            waypoints, segment_times, v0=v0
        )
        self._trajectory.replace_coeffs(waypoints, segment_times, coeffs, t_global)
