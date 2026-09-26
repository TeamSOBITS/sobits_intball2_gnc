#!/usr/bin/env python3
"""replan_minco: EGO-Planner v2-style global/local replanning tracker
(ROS-agnostic, pure).

Production port of the offline-verified design in ``docs/
2026-09-20_ego_v2_style_replan_migration_plan.md`` (offline verification log:
``docs/2026-09-20_ego_planner_v2_aligned_local_replan_design_and_offline_
verification.md``, facts 44-54). This is a *new*, separate class from
:class:`~sobits_intball2_gnc.guidance.trajectory_tracking.
replanning_minco_v2_tracker.ReplanningMincoV2Tracker` (which keeps its
periodic-full-global-reopt + closed-form-Hermite-local design unchanged) --
not yet the default; opt-in via ``trajectory_tracking_mode=
"replan_minco"``. **Experimental -- offline-verified only, not yet
sim-validated** (same caveat as v2 originally carried).

Architecture, matching EGO-Planner v2's actual global/local role split
(inverted from v2's "heavy global, cheap local"): a global MINCO trajectory
built once, and every ``local_replan_period`` seconds a fresh local
``MincoTrajectory`` from the previous local's own state at the current time
(EGO-Planner v2 ``planFromLocalTraj``; only the very first local plan starts
from the measured pose) to a look-ahead point on the global. That local *is*
the tracked trajectory -- the closed-form quintic Hermite connector layer v2
has is dropped entirely. How the global and each local are built (look-ahead
target, face travel two-solve, multi-piece seeds, obstacle rebound) lives in
:class:`~sobits_intball2_gnc.guidance.local_planner.minco_local_planner.
MincoLocalPlanner` (EGO-Planner v2 ``planner_manager``); this class is the
``ego_replan_fsm`` side: when to replan, collision checks, emergency stop.

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
  uniformly (falls back to ``"static_toppra"``).
- Replans keep going after the local target reaches ``p_target``
  (``touch_goal``) until the goal-touching local has played out, which is
  also ``total_duration`` (EGO-Planner v2 ``EXEC_TRAJ``: replan once the
  current local has run ``thresh_replan_time``, done when ``touch_the_goal``
  and ``t_cur > duration``). Stopping at ``touch_goal`` left the rest of the
  route unreplanned -- no reaction to obstacles there (``docs/2026-09-23_
  replanning_minco_v3_remaining_tasks.md`` A1). The previous ETA-only
  ``total_duration`` never ended the goal under a steady-state TF offset
  (``docs/2026-09-23_replanning_minco_v3_fix_live_sim_verification.md``).

No velocity estimator: replans start from the reference state, so measured
pose is only used for the first plan, the TF-freshness latch and the ETA.

``async_replan=True`` solves the local in a background thread while
``sample()`` keeps playing the old local (EGO-Planner v2's planner/
``traj_server`` split), so a slow solve no longer stalls the setpoint stream.
The new local's t=0 is when its start state was sampled, not when the solve
finished as in EGO v2: with 0.2-0.35 s solves the latter jumps the reference
back by v*solve_time (``docs/2026-09-24_replanning_minco_replan_face_travel_
replan_gap_offline_check.md``).
"""
import threading

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.guidance.local_planner.minco_local_planner import MincoLocalPlanner
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoInfeasibleError

DEFAULT_LOCAL_REPLAN_PERIOD_S = 1.0
DEFAULT_PLANNING_HORIZON_M = 2.0
# EGO-Planner v2 safety_timer_ period.
DEFAULT_COLLISION_CHECK_PERIOD_S = 0.05


class ReplanMincoTracker:
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
        face_travel, forward_axis, local_max_vel: see module docstring;
            ``local_max_vel`` [m/s] is only used with ``face_travel``.
        local_piece_length_m: with ``face_travel``, splits the local's
            position-shape solve into ``ceil(distance / this)`` (>= 2) pieces
            with unboxed interior points, seeded from the previous local then
            the global (EGO-Planner v2 ``polyTraj_piece_length``/
            ``computeInitState``). ``None`` keeps the single-piece shape solve.
        obstacle_grid, obstacle_clearance_soft: ``minco_native_py.OccupancyGrid``
            the local shape solve avoids (EGO-Planner v2 rebound, needs
            ``local_piece_length_m``); a goal inside it is moved back along the
            global to the first free point (``mondifyInCollisionFinalGoal``).
        stop_profile_fn, emergency_time_s, collision_check_period: EGO-Planner v2
            ``checkCollisionCallback`` (with ``obstacle_grid``): every period the
            current local is checked ahead (first 3/4, all when it touches the
            goal); on a collision it replans at once, and if that fails with the
            collision under ``emergency_time_s`` away it plays
            ``stop_profile_fn(p, v, q, omega)`` (an object with ``sample(t) ->
            (p, v, a, q, omega, alpha)`` and ``duration``, e.g. ``StoppingProfile``)
            and holds its end, replanning from rest every period until one
            succeeds (``EMERGENCY_STOP`` then ``GEN_NEW_TRAJ``). ``emergency_time_s=None``
            uses the stop profile's own duration (EGO's fixed 1 s assumes a drone
            that brakes in a fraction of a second).
        async_replan: see module docstring. ``False`` solves inside
            ``sample()`` (deterministic, for offline/unit use).

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
                 initial_v0=None, face_travel=False, forward_axis=(1.0, 0.0, 0.0),
                 local_max_vel=None, async_replan=False, local_piece_length_m=None,
                 obstacle_grid=None, obstacle_clearance_soft=0.5, stop_profile_fn=None,
                 emergency_time_s=None,
                 collision_check_period=DEFAULT_COLLISION_CHECK_PERIOD_S):
        if (target_speed is None) != (max_accel is None):
            raise ValueError(
                "target_speed and max_accel must be given together (both "
                "None selects MincoTrajectory's free-time plan_minco path, "
                "both non-None selects plan_minco_heuristic_time)"
            )
        if local_piece_length_m is not None and (not face_travel or local_piece_length_m <= 0.0):
            raise ValueError("local_piece_length_m needs face_travel and must be positive")
        if obstacle_grid is not None and local_piece_length_m is None:
            raise ValueError("obstacle_grid needs local_piece_length_m")
        if local_replan_period <= 0.0 or planning_horizon_m <= 0.0:
            raise ValueError(
                "local_replan_period and planning_horizon_m must be positive "
                "(got %r, %r)" % (local_replan_period, planning_horizon_m)
            )

        p0 = np.asarray(p0, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._q0 = np.asarray(q0, dtype=float)
        self._local_replan_period = float(local_replan_period)
        self._async_replan = bool(async_replan)
        self._stop_profile_fn = stop_profile_fn
        self._emergency_time_s = None if emergency_time_s is None else float(emergency_time_s)
        self._collision_check_period = float(collision_check_period)
        self._since_collision_check = 0.0
        self._stop_profile = None
        self._stop_elapsed = 0.0
        self.last_collision_ahead_s = None
        self.emergency_stops = 0
        self._rest_replan_failures = 0
        self._pending_thread = None
        self._pending_result = None
        self._pending_lag = 0.0
        self._collision_replan_pending = False
        self._pending_is_collision_replan = False

        v0 = np.zeros(3) if initial_v0 is None else np.asarray(initial_v0, dtype=float)
        self._last_p_now = p0.copy()

        self.last_fallback_reason = None
        self.last_replan_occurred = False
        self.last_local_fallback = False
        self.last_replan_solve_seconds = None
        self.last_replan_lag_seconds = None

        self._planner = MincoLocalPlanner(
            p0, v0, p_target, self._q0, target_speed, max_accel, route_waypoints,
            planning_horizon_m, via_half_width, wrench_safety_margin,
            attitude_resample_spacing_m, face_travel, forward_axis, local_max_vel,
            local_piece_length_m, obstacle_grid, obstacle_clearance_soft)
        self.last_goal_moved_out_of_obstacle = False

        self._local_elapsed = 0.0
        self._since_replan_attempt = 0.0
        self._local_trajectory, self._local_touches_goal = self._build_local(
            p0, v0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), None, 0.0)
        self.last_replan_solve_seconds = self._local_trajectory.solve_wall_seconds

        self._fallen_back = False
        self._prev_t = 0.0
        p_out, v_out, a_out, q_out = self._local_trajectory.sample(0.0)
        self._last_output = (p_out, v_out, a_out, q_out)
        self.last_body_angular = self._local_trajectory.sample_body_angular(0.0)

    def sample(self, t):
        self.last_replan_occurred = False
        self.last_local_fallback = False
        if self._fallen_back:
            return self._last_output

        pose = self._pose_fn()
        if pose is None or not self._tf_fresh_fn(pose[2]):
            self._fallen_back = True
            self.last_fallback_reason = "tf_stale"
            self.last_body_angular = (np.zeros(3), np.zeros(3))
            return self._last_output

        self._last_p_now = np.asarray(pose[0], dtype=float)

        dt = float(t) - self._prev_t
        if dt <= 0.0:
            return self._last_output
        self._prev_t = float(t)

        self._local_elapsed += dt
        self._since_replan_attempt += dt
        if self._pending_thread is None and self._planner.goal_in_obstacle():
            p_ref, v_ref, _a, _q = self._local_trajectory.sample(self._local_elapsed)
            if self._planner.move_goal_out_of_obstacle(p_ref, v_ref):
                self.last_goal_moved_out_of_obstacle = True
        if self._stop_profile is not None:
            return self._sample_emergency_stop(dt)
        if self._planner.obstacle_grid is not None:
            self._since_collision_check += dt
            if self._since_collision_check >= self._collision_check_period:
                self._since_collision_check = 0.0
                self._check_collision()
                if self._stop_profile is not None:
                    return self._sample_emergency_stop(0.0)
        if self._pending_thread is not None:
            self._pending_lag += dt
            if not self._pending_thread.is_alive():
                self._pending_thread = None
                self._adopt_local(self._pending_result, self._pending_lag)
                if self._collision_replan_pending:
                    self._resolve_collision_replan()
                    if self._stop_profile is not None:
                        return self._sample_emergency_stop(0.0)
        elif (not self._goal_local_played_out()
                and self._local_elapsed >= self._local_replan_period
                and self._since_replan_attempt >= self._local_replan_period):
            self._since_replan_attempt = 0.0
            if self._async_replan:
                self._start_background_replan()
            else:
                self._adopt_local(self._try_build_local(self._reference_start_state()), 0.0)

        p_out, v_out, a_out, q_out = self._local_trajectory.sample(self._local_elapsed)
        self.last_body_angular = self._local_trajectory.sample_body_angular(
            self._local_elapsed)

        self._last_output = (p_out, v_out, a_out, q_out)
        return self._last_output

    def _reference_start_state(self):
        p_ref, v_ref, a_ref, _q = self._local_trajectory.sample(self._local_elapsed)
        rv_ref = self._local_trajectory.sample_rotvec_derivatives(self._local_elapsed)
        return (p_ref, v_ref, a_ref, *rv_ref, self._local_trajectory, self._local_elapsed)

    def _collision_ahead_s(self):
        """Time until the current local first enters the inflated grid, checked from now
        over its first 3/4 (all of it when it touches the goal), or ``None``."""
        local = self._local_trajectory
        end = local.global_total_duration * (1.0 if self._local_touches_goal else 0.75)
        grid = self._planner.obstacle_grid
        speed = self._planner.local_max_vel or self._planner.global_avg_speed
        t_step = grid.resolution / 2.0 / max(speed, 1e-6)
        t = self._local_elapsed
        while t <= end:
            if grid.inflated_occupied(list(local.sample(t)[0])):
                return t - self._local_elapsed
            t += t_step
        return None

    def set_obstacle_grid(self, grid):
        """Swap in a rebuilt grid (obstacles changed); the next collision check uses it."""
        if self._planner.obstacle_grid is None:
            raise ValueError("set_obstacle_grid needs a tracker built with obstacle_grid")
        self._planner.obstacle_grid = grid

    def _check_collision(self):
        self.last_collision_ahead_s = self._collision_ahead_s()
        if self.last_collision_ahead_s is None:
            return
        if self._async_replan:
            # Like EGO v2: keep playing the old local while replanning, decide to stop on the result.
            self._collision_replan_pending = True
            if self._pending_thread is None:
                self._start_background_replan()
            return
        result = self._try_build_local(self._reference_start_state())
        if result is not None:
            self._adopt_local(result, 0.0)
            self._since_replan_attempt = 0.0
            return
        self._emergency_stop_if_collision_close()

    def _start_background_replan(self):
        self._since_replan_attempt = 0.0
        self._pending_lag = 0.0
        self._pending_is_collision_replan = self._collision_replan_pending
        self._pending_thread = threading.Thread(
            target=self._solve_local_in_background, args=(self._reference_start_state(),),
            daemon=True)
        self._pending_thread.start()

    def _resolve_collision_replan(self):
        if not self._pending_is_collision_replan:
            # Started before the collision was seen (possibly on the old grid): solve again.
            self._start_background_replan()
            return
        self._collision_replan_pending = False
        self.last_collision_ahead_s = self._collision_ahead_s()
        if self.last_collision_ahead_s is not None:
            self._emergency_stop_if_collision_close()

    def _emergency_stop_if_collision_close(self):
        if self._stop_profile_fn is None:
            return
        p_ref, v_ref, _a, q_ref = self._local_trajectory.sample(self._local_elapsed)
        omega_ref, _alpha = self._local_trajectory.sample_body_angular(self._local_elapsed)
        profile = self._stop_profile_fn(p_ref, v_ref, q_ref, omega_ref)
        emergency_time = (profile.duration if self._emergency_time_s is None
                          else self._emergency_time_s)
        if self.last_collision_ahead_s < emergency_time:
            self._stop_profile = profile
            self._stop_elapsed = 0.0
            self.emergency_stops += 1
            self.last_fallback_reason = "emergency_stop"

    def _sample_emergency_stop(self, dt):
        """Play the stop profile, then hold its end and replan from rest every period."""
        self._stop_elapsed += dt
        profile = self._stop_profile
        t = min(self._stop_elapsed, profile.duration)
        p, v, a, q, omega, alpha = profile.sample(t)
        self.last_body_angular = (np.asarray(omega), np.asarray(alpha))
        self._last_output = (p, v, a, q)
        if self._stop_elapsed >= profile.duration:
            if self._since_replan_attempt >= self._local_replan_period:
                self._since_replan_attempt = 0.0
                rv = quat_log(quat_mul(quat_conj(self._q0), np.asarray(q, dtype=float)))
                result = self._try_build_local(
                    (np.asarray(p), np.zeros(3), np.zeros(3), rv, np.zeros(3), np.zeros(3), None, 0.0))
                if result is not None:
                    self._stop_profile = None
                    self._rest_replan_failures = 0
                    self._adopt_local(result, 0.0)
                else:
                    self._rest_replan_failures += 1
        return self._last_output

    def _goal_local_played_out(self):
        return (self._local_touches_goal
                and self._local_elapsed >= self._local_trajectory.global_total_duration)

    def _try_build_local(self, start_state):
        try:
            return self._build_local(*start_state)
        except MincoInfeasibleError:
            return None

    def _solve_local_in_background(self, start_state):
        try:
            self._pending_result = self._try_build_local(start_state)
        except Exception as exc:  # re-raised on the sampling thread in _adopt_local
            self._pending_result = exc

    def _adopt_local(self, result, lag):
        if isinstance(result, Exception):
            raise result
        if result is None:
            # _local_elapsed deliberately not reset: keeps the reference
            # continuous on the old local trajectory until a retry succeeds.
            self.last_fallback_reason = "minco_infeasible_local"
            self.last_local_fallback = True
            return
        self._local_trajectory, self._local_touches_goal = result
        self._local_elapsed = lag
        self.last_replan_occurred = True
        self.last_replan_solve_seconds = self._local_trajectory.solve_wall_seconds
        self.last_replan_lag_seconds = lag

    @property
    def total_duration(self):
        """End of the goal-touching local trajectory once it exists, else a
        genuine ETA (remaining measured distance over the global average
        speed, always ahead of ``t`` so the reference keeps advancing)."""
        if self._local_touches_goal and self._stop_profile is None:
            local_start_t = self._prev_t - self._local_elapsed
            return local_start_t + self._local_trajectory.global_total_duration
        remaining = float(np.linalg.norm(self._planner.p_target - self._last_p_now))
        eta = remaining / max(self._planner.global_avg_speed, 1e-6)
        return self._prev_t + eta

    @property
    def trajectory(self):
        """The (one-time-built) global trajectory -- analogous to v2's
        ``trajectory`` property, e.g. for an RViz preview."""
        return self._planner.global_trajectory

    @property
    def local_trajectory(self):
        return self._local_trajectory

    @property
    def goal_position(self):
        """Where the vehicle ends up: the requested goal, or the free point it was
        moved back to when an obstacle covers it."""
        return self._planner.p_target.copy()

    def _build_local(self, p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local, prev_elapsed):
        return self._planner.build_local(p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local,
                                         prev_elapsed, rest_failures=self._rest_replan_failures)
