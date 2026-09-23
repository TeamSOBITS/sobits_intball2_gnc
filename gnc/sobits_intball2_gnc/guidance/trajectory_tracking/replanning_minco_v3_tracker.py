#!/usr/bin/env python3
"""replanning_minco v3: EGO-Planner v2-style global/local replanning tracker
(ROS-agnostic, pure).

Production port of the offline-verified design in ``docs/
2026-09-20_ego_v2_style_replan_migration_plan.md`` (offline verification log:
``docs/2026-09-20_ego_planner_v2_aligned_local_replan_design_and_offline_
verification.md``, facts 44-54). This is a *new*, separate class from
:class:`~sobits_intball2_gnc.guidance.trajectory_tracking.
replanning_minco_v2_tracker.ReplanningMincoV2Tracker` (which keeps its
periodic-full-global-reopt + closed-form-Hermite-local design unchanged) --
not yet the default; opt-in via ``trajectory_tracking_mode=
"replanning_minco_v3"``. **Experimental -- offline-verified only, not yet
sim-validated** (same caveat as v2 originally carried).

Architecture, matching EGO-Planner v2's actual global/local role split
(inverted from v2's "heavy global, cheap local"):

- **Global layer**: built **once**, at construction, from ``p0`` +
  ``route_waypoints`` + ``p_target`` via :class:`~sobits_intball2_gnc.
  guidance.trajectory.minco_trajectory.MincoTrajectory` (heuristic-time if
  ``target_speed``/``max_accel`` given, free-time ``plan_minco`` otherwise).
  Never rebuilt -- no via-waypoint retirement logic is needed (unlike v2's
  ``_next_idx``) because there is no periodic rebuild to retire waypoints
  *for* (offline verification 追記5: reasoned to be unnecessary for a
  build-once global, not independently implementation-verified with
  multiple via points).
- **Local layer**: every ``local_replan_period`` seconds, a fresh
  single-segment (K=1, no interior via points) free-time ``MincoTrajectory``
  connects the previous local trajectory's own position/velocity/acceleration
  at the current time (EGO-Planner v2 ``planFromLocalTraj``; only the very
  first local plan starts from the measured pose) to a look-ahead point
  ``planning_horizon_m`` ahead along the global trajectory, ending at the
  global trajectory's velocity there (EGO-Planner v2 ``getLocalTarget``; zero
  at the goal or once inside braking distance of it). Starting from the
  measured state with zero acceleration and ending at rest instead -- the
  original port -- made the vehicle crawl, overshoot the goal, and absorb
  disturbance-induced velocity into every plan as lateral drift (docs/
  2026-09-23_replanning_minco_v3_sim_verification_result_and_root_cause.md).
  (:meth:`_get_local_target`, ported from ``prototype_ego_v2_
  style_local_replan.py``'s ``get_local_target``). This *is* the trajectory
  tracked -- the closed-form quintic Hermite connector layer v2 has is
  dropped entirely. K=1 matches the offline-verified baseline; K>1 (interior
  via points per local segment) was offline-verified to help (~29.7% faster
  convergence, offline verification fact 51) but is deferred as a follow-up
  (fact 54: combining it with warm start showed no measurable benefit, and
  K>1's own production wiring is still open).

Attitude is out of scope for this initial production port (依頼者の意向):
every ``MincoTrajectory`` this class builds uses ``face_travel=False`` (fixed
at ``q0``), and ``w0=0`` always. ``attitude_resample_spacing_m`` is still
forwarded to the *global* build for spatial densify (the global route's own
smoothness, independent of whether attitude is tracked) -- this only works
because of the ``face_travel``/densify decoupling fix in ``minco_trajectory.
py`` (2026-09-23, ``docs/2026-09-20_ego_v2_style_replan_migration_plan.md``
"やるべき内容1"); before that fix, ``face_travel=False`` silently disabled
densify.

Fallback semantics (``docs/2026-09-20_ego_v2_style_replan_migration_plan.md``
"1.5", decided 2026-09-23 -- simplicity-first, ported from v2's local-Hermite
fallback rather than inventing new latch/hard-stop machinery):

- TF stale/unavailable: freezes *everything*, same one-way latch semantics
  as v2/``ReplanningTrajectoryTracker`` -- ``sample()`` just keeps returning
  the last output.
- Local replan failing (``MincoInfeasibleError``): keeps using the
  *previous* local trajectory untouched; retried every subsequent period,
  no latch (self-healing if a later solve succeeds). Unlike v2's closed-form
  Hermite (which is re-evaluated past its own horizon by construction, since
  it's built to reach the look-ahead point exactly), a free-time-solved
  ``MincoTrajectory`` sampled past its own ``global_total_duration`` would
  need to extrapolate a quintic polynomial past the domain it was fit to,
  which can diverge -- but ``MincoTrajectory.sample()`` already clamps at
  its own duration and holds the terminal state (module docstring of that
  class), so no special handling is needed here.
- First-ever local build failing (in the constructor, no previous trajectory
  to fall back to): raises ``MincoInfeasibleError``, same as the global
  build failing -- the caller (``GuidanceExecutor``) already handles this
  uniformly (falls back to ``"static"``).
- Once the local target reaches ``p_target`` (``touch_goal``), local
  replanning stops and that last local trajectory plays out to its end,
  which is also ``total_duration`` (EGO-Planner v2 ``EXEC_TRAJ``: done when
  ``touch_the_goal`` and ``t_cur > duration``, reference time only). An
  earlier distance-based freeze was harmful because its one frozen solve
  started from a noisy *measured* velocity; replans now start from the
  reference state, so re-solving toward the fixed goal adds nothing. The
  previous ETA-only ``total_duration`` never ended the goal under a
  steady-state TF offset (``docs/2026-09-23_replanning_minco_v3_fix_live_
  sim_verification.md``).

No velocity estimator: replans start from the reference state, so measured
pose is only used for the first plan, the TF-freshness latch and the ETA.
"""
import numpy as np

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)

DEFAULT_LOCAL_REPLAN_PERIOD_S = 1.0
DEFAULT_PLANNING_HORIZON_M = 2.0
# Resolution for the internal forward-scan of the (already-solved, analytic)
# global trajectory in _get_local_target -- not a control-loop timing
# decision (no clock/sleep involved), just how finely that scan samples the
# trajectory function, so CLAUDE.mdのreal-time禁止の対象外。
_LOCAL_TARGET_SEARCH_DT = 0.05


class ReplanningMincoV3Tracker:
    """See module docstring.

    Args:
        p0: initial position, shape ``(3,)``.
        p_target: target position, shape ``(3,)``.
        pose_fn: callable ``() -> (pos, quat, stamp) | None``.
        tf_fresh_fn: callable ``(stamp) -> bool``.
        q0: reference attitude ``[x,y,z,w]`` -- ``p0``'s attitude is assumed
            to equal ``q0`` exactly, matching ``MincoTrajectory``'s
            head-attitude convention. Attitude is out of scope (module
            docstring); this class always commands ``q0``.
        target_speed, max_accel: forwarded to the (one-time) global
            ``MincoTrajectory(..., target_speed=..., max_accel=...)``. Both
            non-None selects ``plan_minco_heuristic_time``, both ``None``
            selects the free-time ``plan_minco`` path (must be given
            together, same rule as ``MincoTrajectory`` itself). The local
            layer always uses the free-time path regardless of this choice;
            ``max_accel`` (if given) also sets the braking distance inside
            which the local terminal velocity is forced to zero.
        route_waypoints: optional ordered interior via points, shape
            ``(N, 3)``, included once in the global route. No retirement
            logic (module docstring).
        local_replan_period: seconds between local replans.
        planning_horizon_m: look-ahead distance, meters, for the local
            segment's target point along the global trajectory.
        via_half_width, wrench_safety_margin, attitude_resample_spacing_m:
            forwarded to the global ``MincoTrajectory(...)`` build only (the
            local layer is always a 2-waypoint, no-via-point segment, so
            ``via_half_width`` doesn't apply to it).
        initial_v0: initial velocity, shape ``(3,)``, default zero.

    Raises:
        ValueError: if ``target_speed``/``max_accel`` are not given together.
            Also if ``local_replan_period``/``planning_horizon_m`` is not
            positive.
        MincoInfeasibleError: if the initial global or local solve fails.
    """

    def __init__(self, p0, p_target, pose_fn, tf_fresh_fn, q0,
                 target_speed, max_accel, route_waypoints=None,
                 local_replan_period=DEFAULT_LOCAL_REPLAN_PERIOD_S,
                 planning_horizon_m=DEFAULT_PLANNING_HORIZON_M,
                 via_half_width=0.3, wrench_safety_margin=1.0,
                 attitude_resample_spacing_m=None,
                 initial_v0=None):
        if (target_speed is None) != (max_accel is None):
            raise ValueError(
                "target_speed and max_accel must be given together (both "
                "None selects MincoTrajectory's free-time plan_minco path, "
                "both non-None selects plan_minco_heuristic_time)"
            )
        if local_replan_period <= 0.0 or planning_horizon_m <= 0.0:
            raise ValueError(
                "local_replan_period and planning_horizon_m must be positive "
                "(got %r, %r)" % (local_replan_period, planning_horizon_m)
            )

        p0 = np.asarray(p0, dtype=float)
        self._p_target = np.asarray(p_target, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._q0 = np.asarray(q0, dtype=float)
        self._target_speed = None if target_speed is None else float(target_speed)
        self._max_accel = None if max_accel is None else float(max_accel)
        self._local_replan_period = float(local_replan_period)
        self._planning_horizon_m = float(planning_horizon_m)
        self._via_half_width = float(via_half_width)
        self._wrench_safety_margin = float(wrench_safety_margin)
        self._attitude_resample_spacing_m = attitude_resample_spacing_m

        route_waypoints = (
            np.zeros((0, 3)) if route_waypoints is None
            else np.asarray(route_waypoints, dtype=float).reshape(-1, 3)
        )

        v0 = np.zeros(3) if initial_v0 is None else np.asarray(initial_v0, dtype=float)
        self._last_p_now = p0.copy()

        self.last_fallback_reason = None
        self.last_replan_occurred = False
        self.last_local_fallback = False
        self.last_replan_solve_seconds = None

        if len(route_waypoints):
            waypoints = np.vstack([p0, route_waypoints, self._p_target])
        else:
            waypoints = np.array([p0, self._p_target])
        self._global_trajectory = MincoTrajectory(
            waypoints, self._q0, v0=v0, w0=np.zeros(3), face_travel=False,
            via_half_width=self._via_half_width,
            attitude_resample_spacing_m=self._attitude_resample_spacing_m,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=self._target_speed, max_accel=self._max_accel,
        )
        route_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        # Always sized from the global trajectory's own average speed (not
        # target_speed directly), matching v2's _freetime_avg_speed -- well
        # defined immediately, never collapses near zero the way a live
        # velocity estimate transiently can right at start-of-motion, and
        # works identically whether the global build used the heuristic-time
        # or free-time path.
        self._global_avg_speed = route_length / self._global_trajectory.global_total_duration
        self._global_search_t = 0.0

        self._local_elapsed = 0.0
        self._since_replan_attempt = 0.0
        self._local_trajectory, self._local_touches_goal = self._build_local(
            p0, v0, np.zeros(3))
        self.last_replan_solve_seconds = self._local_trajectory.solve_wall_seconds

        self._fallen_back = False
        self._prev_t = 0.0
        p_out, v_out, a_out, q_out = self._local_trajectory.sample(0.0)
        self._last_output = (p_out, v_out, a_out, q_out)

    def sample(self, t):
        self.last_replan_occurred = False
        self.last_local_fallback = False
        if self._fallen_back:
            return self._last_output

        pose = self._pose_fn()
        if pose is None or not self._tf_fresh_fn(pose[2]):
            self._fallen_back = True
            self.last_fallback_reason = "tf_stale"
            return self._last_output

        self._last_p_now = np.asarray(pose[0], dtype=float)

        dt = float(t) - self._prev_t
        if dt <= 0.0:
            return self._last_output
        self._prev_t = float(t)

        self._local_elapsed += dt
        self._since_replan_attempt += dt
        if (not self._local_touches_goal
                and self._since_replan_attempt >= self._local_replan_period):
            self._since_replan_attempt = 0.0
            p_ref, v_ref, a_ref, _q = self._local_trajectory.sample(self._local_elapsed)
            try:
                new_local, self._local_touches_goal = self._build_local(p_ref, v_ref, a_ref)
                self._local_trajectory = new_local
                self._local_elapsed = 0.0
                self.last_replan_occurred = True
                self.last_replan_solve_seconds = new_local.solve_wall_seconds
            except MincoInfeasibleError:
                # _local_elapsed deliberately not reset: keeps the reference
                # continuous on the old local trajectory until a retry succeeds.
                self.last_fallback_reason = "minco_infeasible_local"
                self.last_local_fallback = True

        p_out, v_out, a_out, q_out = self._local_trajectory.sample(self._local_elapsed)

        self._last_output = (p_out, v_out, a_out, q_out)
        return self._last_output

    @property
    def total_duration(self):
        """End of the goal-touching local trajectory once it exists, else a
        genuine ETA (remaining measured distance over the global average
        speed, always ahead of ``t`` so the reference keeps advancing)."""
        if self._local_touches_goal:
            local_start_t = self._prev_t - self._local_elapsed
            return local_start_t + self._local_trajectory.global_total_duration
        remaining = float(np.linalg.norm(self._p_target - self._last_p_now))
        eta = remaining / max(self._global_avg_speed, 1e-6)
        return self._prev_t + eta

    @property
    def trajectory(self):
        """The (one-time-built) global trajectory -- analogous to v2's
        ``trajectory`` property, e.g. for an RViz preview."""
        return self._global_trajectory

    @property
    def local_trajectory(self):
        return self._local_trajectory

    def _build_local(self, p0, v0, a0):
        """Solve a fresh K=1 (2-waypoint, no interior via points) free-time
        local segment from the head state ``p0``/``v0``/``a0`` to the
        look-ahead target, ending at the global trajectory's velocity there
        (zero at the goal or inside braking distance of it, EGO-Planner v2
        ``getLocalTarget``)."""
        target_pos, target_vel, touch_goal = self._get_local_target(p0)
        inside_braking = (
            self._max_accel is not None
            and np.linalg.norm(self._p_target - target_pos)
            < float(target_vel @ target_vel) / (2.0 * self._max_accel)
        )
        v_tail = np.zeros(3) if (touch_goal or inside_braking) else target_vel
        local = MincoTrajectory(
            [p0, target_pos], self._q0, v0=v0, w0=np.zeros(3),
            face_travel=False, via_half_width=self._via_half_width,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=None, max_accel=None, a0=a0, v_tail=v_tail,
        )
        return local, touch_goal

    def _get_local_target(self, p_from):
        """Walk the global trajectory forward from the last search cursor
        until its sampled position is ``planning_horizon_m`` away from
        ``p_from``, or the global trajectory's own end is reached (in which
        case the target is ``p_target`` directly and ``touch_goal=True``).
        Returns ``(pos, vel, touch_goal)``. Ported from
        ``prototype_ego_v2_style_local_replan.py``'s ``get_local_target``."""
        total_dur = self._global_trajectory.global_total_duration
        t_step = max(
            self._planning_horizon_m / 20.0 / max(self._global_avg_speed, 1e-6),
            _LOCAL_TARGET_SEARCH_DT,
        )
        t = self._global_search_t
        while t < total_dur:
            pos_t, vel_t, _a, _q = self._global_trajectory.sample(t)
            if np.linalg.norm(pos_t - p_from) >= self._planning_horizon_m:
                self._global_search_t = t
                return pos_t, vel_t, False
            t += t_step
        self._global_search_t = total_dur
        return self._p_target.copy(), np.zeros(3), True
