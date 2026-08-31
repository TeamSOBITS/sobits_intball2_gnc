#!/usr/bin/env python3
"""Real-time re-planning trajectory tracker (ROS-agnostic, pure).

Implements the "毎tick実TFを見て並進軌道を更新する" idea investigated in
``docs/guidance_realtime_replanning_design.md``, with every safeguard that
investigation's 6 節 found to be load-bearing (a naive always-replan
implementation was shown to be non-terminating, overshoot-prone, and
saturating -- see that section). This class only re-plans on a slower,
explicitly-decided cadence (``docs/archive/achieved/
2026-08-24_replan_rate_design.md``), and permanently stops re-planning
(falling back to whatever the last successful re-plan produced) once any of
three conditions holds (conditions 1-2: ``docs/archive/achieved/
2026-08-24_replanning_distance_fallback_decision.md``; condition 3:
``docs/2026-08-25_v0_aware_time_allocation_lateral_velocity_fix.md`` 論点4):

1. the remaining distance to the target drops below ``distance_fallback_m``
   (re-planning near the target is where 6-3/6-4 節's chattering/noise-
   force-budget failures live), or
2. the live pose is unavailable or its TF stamp is stale (re-planning off a
   frozen/corrupted pose reintroduces the failure modes 6-8 節 documents), or
3. ``HeuristicSegmentTimeAllocator.allocate`` raises
   ``SegmentTimeInfeasibleError`` for the current ``(p_now, v0)`` -- a
   genuine kinematic dead end (e.g. a large lateral velocity residual right
   after a disturbance or mid-turn, too close to the target to size a
   feasible first-segment time against) that this class must not let
   propagate out of ``sample()``.

Optional ordered interior relay points (``route_waypoints``, ``docs/
2026-08-25_guidance_waypoint_insertion_curve_verification.md``, generalized
from a single point to a list 2026-08-31, ``docs/
2026-08-31_curve_aware_realtime_replanning_design_discussion.md``): every
re-plan routes through ``[p_now, *route_waypoints[_next_idx:], p_target]``
instead of the plain ``[p_now, p_target]``. "Passed" is judged, for each
pending waypoint in order, by comparing the current re-plan's live remaining
distance to the target against that waypoint's own (fixed) distance to the
target -- once the vehicle is closer to the target than a pending waypoint
ever was, that waypoint (and, by the loop below, any earlier pending one) is
treated as having gone by, and dropped from every subsequent re-plan's route.
This is deliberately progress-toward-target-based, not raw distance-to-
waypoint: a disturbance that pushes the vehicle sideways (this feature's
whole motivation) must not make it look "stuck approaching the waypoint" and
never advance to the next leg. The check is a ``while`` loop, not a single
``if``, because one re-plan interval can advance past more than one pending
waypoint at once (a long ``replan_every_n_ticks`` or closely-spaced
waypoints) -- this invariant only holds if ``route_waypoints`` is ordered
with strictly decreasing distance to ``p_target``; a route that temporarily
increases that distance (e.g. a future obstacle-avoidance detour) would
defeat this "passed" check and needs its own design (see that doc's 未解決の
穴2).

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

from sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator import (
    SegmentTimeInfeasibleError,
)
from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
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
            :class:`~sobits_intball2_gnc.guidance.trajectory.trajectory.Trajectory`
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
        route_waypoints: optional ordered interior relay points, shape
            ``(N, 3)`` (module docstring). ``None``/``[]`` (default)
            reproduces the prior 2-waypoint-only re-planning exactly.
        use_minco: when ``True``, each re-plan builds a fresh :class:`~
            sobits_intball2_gnc.guidance.trajectory.minco_trajectory.
            MincoTrajectory` (wrench-envelope-aware 6-DOF, ``docs/archive/
            achieved/2026-08-30_minco_attitude_torque_status_and_next_steps.md``)
            instead of ``HeuristicSegmentTimeAllocator`` +
            ``HermiteSplineTrajectoryGenerator``. Default ``False``
            reproduces prior behavior exactly. Experimental (Phase 1, not
            yet sim-validated against the default path).
        q0: reference attitude ``[x,y,z,w]`` MINCO's rotation-vector
            waypoints are expressed relative to. Required when
            ``use_minco=True``, ignored otherwise.
        minco_via_half_width: ``use_minco=True`` only -- forwarded to each
            re-plan's ``MincoTrajectory(..., via_half_width=...)`` (see that
            class's docstring and ``docs/
            2026-08-30_static_minco_face_travel_gap.md`` 追記3). Ignored
            when ``use_minco=False``.
        minco_attitude_resample_spacing_m: ``use_minco=True`` only --
            forwarded to each re-plan's ``MincoTrajectory(...,
            attitude_resample_spacing_m=...)`` (``docs/
            2026-08-30_static_minco_face_travel_gap.md`` 追記4). ``None``
            (default) reproduces prior behavior. Ignored when
            ``use_minco=False``.
        minco_wrench_safety_margin: ``use_minco=True`` only -- forwarded to
            each re-plan's ``MincoTrajectory(..., wrench_safety_margin=...)``
            (``docs/2026-08-30_static_minco_face_travel_gap.md`` 追記2).
            ``1.0`` (default) reproduces prior behavior. Ignored when
            ``use_minco=False``.

    Raises:
        ValueError: if ``max_accel`` is ``None``, or if ``use_minco=True``
            and ``q0`` is ``None``.
    """

    def __init__(self, trajectory, p_target, pose_fn, tf_fresh_fn, velocity_fn,
                 target_speed, max_accel,
                 distance_fallback_m=DEFAULT_DISTANCE_FALLBACK_M,
                 replan_every_n_ticks=DEFAULT_REPLAN_EVERY_N_TICKS,
                 route_waypoints=None, use_minco=False, q0=None,
                 minco_via_half_width=0.3,
                 minco_attitude_resample_spacing_m=None,
                 minco_wrench_safety_margin=1.0):
        if max_accel is None:
            raise ValueError(
                "max_accel is required for replanning mode (v0-aware "
                "segment-time allocation needs an acceleration budget to "
                "size the re-planned segment's time against)"
            )
        if use_minco and q0 is None:
            raise ValueError(
                "q0 is required when use_minco=True (MincoTrajectory needs "
                "a reference attitude to express its rotation-vector "
                "waypoints against)"
            )
        self._use_minco = bool(use_minco)
        self._q0 = None if q0 is None else np.asarray(q0, dtype=float)
        self._minco_via_half_width = float(minco_via_half_width)
        self._minco_attitude_resample_spacing_m = (
            None if minco_attitude_resample_spacing_m is None
            else float(minco_attitude_resample_spacing_m)
        )
        self._minco_wrench_safety_margin = float(minco_wrench_safety_margin)
        self._trajectory = trajectory
        # MincoTrajectoryはToppraTrajectory同様、毎回新規インスタンスとして
        # 差し替える設計（Trajectory.replace_coeffsのようなin-place更新は
        # しない、minco_trajectory.pyのクラスdocstring参照）。sample(t)の
        # tはtracker全体で単調増加するグローバル時刻なので、直近の差し替え
        # 時点を_t_originとして憶えておき、MincoTrajectory.sample()には
        # ローカル時刻(t - _t_origin)を渡す（Trajectory側は_t_originを
        # 自前で持つのでuse_minco=Falseのときは未使用）。
        self._t_origin = 0.0
        self._p_target = np.asarray(p_target, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._velocity_fn = velocity_fn
        self._target_speed = float(target_speed)
        self._max_accel = float(max_accel)
        self._distance_fallback_m = float(distance_fallback_m)
        self._replan_every_n_ticks = int(replan_every_n_ticks)
        self._tick_count = 0
        # route_waypoints "passed" index (module docstring): _next_idx starts
        # at 0 and only ever advances -- once a pending waypoint's own
        # (fixed) distance to target is undercut by the live remaining
        # distance, it (and any earlier still-pending one) is dropped for
        # good, same one-way-advance shape as _fallen_back's latch above
        # (never re-armed).
        self._route_waypoints = (
            np.zeros((0, 3)) if route_waypoints is None
            else np.asarray(route_waypoints, dtype=float).reshape(-1, 3)
        )
        self._next_idx = 0
        self._route_target_dists = np.linalg.norm(
            self._route_waypoints - self._p_target, axis=1
        )
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
        # latch to trip on the most recent sample() call: "tf_stale" |
        # "distance" | "segment_time_infeasible" | None (latch not tripped
        # this tick). Stays populated (not reset)
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
        if self._use_minco:
            return self._trajectory.sample(t - self._t_origin)
        return self._trajectory.sample(t)

    @property
    def total_duration(self):
        if self._use_minco:
            return self._t_origin + self._trajectory.global_total_duration
        return self._trajectory.global_total_duration

    @property
    def trajectory(self):
        """The underlying trajectory object -- read-only access for a caller
        that needs the current ``waypoints``/``segment_times``/``coeffs``
        after a re-plan (e.g. to rebuild an RViz preview), without reaching
        into this class's private state. Updated in place on each re-plan
        when ``use_minco=False`` (``Trajectory.replace_coeffs``); replaced
        wholesale with a new instance when ``use_minco=True`` (``minco_
        trajectory.MincoTrajectory`` has no in-place update, module
        docstring)."""
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

        # Progress-toward-target-based "passed" check (module docstring):
        # a while loop, not if, since one re-plan interval can advance past
        # more than one pending waypoint at once.
        while (self._next_idx < len(self._route_waypoints)
               and distance < self._route_target_dists[self._next_idx]):
            self._next_idx += 1

        vel_estimate = self._velocity_fn()
        v0 = (
            np.zeros(3) if vel_estimate is None
            else np.asarray(vel_estimate.vel, dtype=float)
        )

        try:
            self._replan(p_now, v0, t_global)
        except SegmentTimeInfeasibleError:
            # Condition 3 (module docstring): no feasible first-segment time
            # exists for the current (p_now, v0) -- freeze on whatever the
            # last successful re-plan (or the initial trajectory, if none
            # yet) already produced, same as conditions 1/2, rather than let
            # this propagate out of sample() and crash the guidance node.
            self._fallen_back = True
            self.last_fallback_reason = "segment_time_infeasible"
            return
        except MincoInfeasibleError:
            # use_minco=True版のcondition 3相当: plan_mincoがwrench envelope
            # 制約を満たす解に収束しなかった -- 同じくフォールバック。
            self._fallen_back = True
            self.last_fallback_reason = "minco_infeasible"
            return
        self.last_replan_occurred = True

        if distance < self._distance_fallback_m:
            # Condition 1 (module docstring): this was the last re-plan --
            # the freshly replaced trajectory now carries the vehicle the
            # rest of the way as a static leg.
            self._fallen_back = True
            self.last_fallback_reason = "distance"

    def _replan(self, p_now, v0, t_global):
        pending = self._route_waypoints[self._next_idx:]
        if len(pending):
            waypoints = np.vstack([p_now, pending, self._p_target])
        else:
            waypoints = np.array([p_now, self._p_target])

        if self._use_minco:
            # 角速度w0の推定は現状未配線（VelocityEstimatorは並進速度のみ、
            # docs/archive/achieved/2026-08-30_minco_attitude_torque_status_and_next_steps.md
            # の課題外）。Phase 1は零で妥協する。
            new_trajectory = MincoTrajectory(
                waypoints, self._q0, v0=v0, w0=np.zeros(3),
                via_half_width=self._minco_via_half_width,
                attitude_resample_spacing_m=self._minco_attitude_resample_spacing_m,
                wrench_safety_margin=self._minco_wrench_safety_margin,
            )
            self._trajectory = new_trajectory
            self._t_origin = t_global
            return

        segment_times = HeuristicSegmentTimeAllocator(
            target_speed=self._target_speed, max_accel=self._max_accel,
        ).allocate(waypoints, v0=v0)
        coeffs = HermiteSplineTrajectoryGenerator().generate(
            waypoints, segment_times, v0=v0
        )
        self._trajectory.replace_coeffs(waypoints, segment_times, coeffs, t_global)
