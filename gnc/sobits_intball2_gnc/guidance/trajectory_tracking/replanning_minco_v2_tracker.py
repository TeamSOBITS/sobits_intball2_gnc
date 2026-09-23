#!/usr/bin/env python3
"""replanning_minco v4 design: global/local 2-layer replanning tracker
(ROS-agnostic, pure).

Production port of the offline-verified design in ``docs/
2026-09-01_replanning_minco_v4_production_port_plan.md`` Phase 4 (full
history: ``docs/2026-09-01_replanning_minco_v4_open_issues.md``, ``docs/
archive/2026-09-17_replanning_minco_v4_lag_compensation_noise_robustness.md``).
This is a *new*, separate class from :class:`~sobits_intball2_gnc.guidance.
trajectory_tracking.replanning_trajectory_tracker.ReplanningTrajectoryTracker`
(which keeps its single-cadence full-re-solve design unchanged) -- not yet
wired into ``guidance_executor.py``/a public ``trajectory_tracking_mode``
(that wiring, and sim validation, are separate follow-up work).

Two cadences, decoupled:

- **Global layer**: every ``global_replan_period`` seconds, a fresh
  :class:`~sobits_intball2_gnc.guidance.trajectory.minco_trajectory.
  MincoTrajectory` (built with ``target_speed``/``max_accel`` so it uses
  ``plan_minco_heuristic_time`` -- heuristic segment times + analytic
  wrench-violation stretch, no free-time LBFGS) is solved for the
  *remaining* route (existing via-retirement ``_next_idx`` logic, reused
  verbatim from ``ReplanningTrajectoryTracker``).
- **Local layer**: every tick (``sample()`` call), a closed-form quintic
  Hermite segment (:func:`~sobits_intball2_gnc.guidance.utils.
  quintic_hermite.solve_quintic_hermite_coeffs`, no LBFGS) connects the
  current estimated state to a look-ahead point ``t_local`` seconds into
  the global trajectory.

State estimation (:class:`~sobits_intball2_gnc.guidance.utils.
model_kf_estimator.ModelKfEstimator`, "対策案4"/MODEL_KF, one instance for
translation and one for rotation) supplies the *velocity*/*angular-velocity*
boundary conditions fed to both layers; the *position*/*rotation* boundary
conditions use the raw observed pose directly (bench_v10/v11's
``localHead.col(0) = headPosTrue`` pattern -- position is trusted as
observed, only velocity needs estimating). A validity guard
(``guard_global_replan_v0``, 追試16) protects the global layer's ``v0``
input against a single-tick TF glitch that happens to coincide with a
global-replan tick.

Fallback semantics (``docs/
2026-09-17_replanning_minco_v4_fallback_latch_integration_design.md``,
decisions confirmed 2026-09-17):

- TF stale/unavailable: freezes *everything* (same one-way latch semantics
  as ``ReplanningTrajectoryTracker``'s condition 2) -- ``sample()`` just
  keeps returning the last output.
- Remaining distance below ``distance_fallback_m``: a one-way latch that
  permanently stops further *global* replans; the local layer switches to
  connecting directly to ``p_target`` (zero velocity/acceleration) instead
  of a global-trajectory look-ahead point -- same "final leg is static"
  outcome as ``ReplanningTrajectoryTracker``'s condition 1, expressed at
  the global-layer level only.
- Global replan solver reports infeasible (``MincoInfeasibleError``):
  freezes the *global* layer only (keeps using the last successful global
  trajectory) -- the local layer is unaffected and keeps running. This is
  the v4-specific improvement over the single-layer tracker's "freeze
  everything" for the equivalent condition 3.
- Global replan solve taking longer than ``global_replan_period`` (never
  reproduced in the offline benches, only estimated as a risk from real
  ``minco_native_py`` solve-time measurements): no special-cased code --
  the call is synchronous/blocking, so "not done in time" simply means the
  next ``sample()`` call happens correspondingly later (with a
  correspondingly larger ``dt``), naturally continuing to serve the last
  successful global trajectory in the meantime. Chosen over an async/
  threaded global solve (higher implementation/thread-safety cost) as the
  simplest option pending real evidence this matters (real solve times
  measured without bridge contention: K=5 in 115-175ms, comfortably inside
  the default 2.0s period -- ``docs/
  2026-09-01_replanning_minco_v4_production_port_plan.md``).
- Local layer's closed-form solve failing (only possible via a degenerate
  ``t_local`` -- validated at construction time to be > 0 and <
  ``global_replan_period``, so this should not occur in practice; kept as
  defensive fallback rather than assumed impossible): reuses the previous
  tick's local segment, sampled further along it instead of rebuilding.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)
from sobits_intball2_gnc.guidance.utils.model_kf_estimator import ModelKfEstimator
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector
from sobits_intball2_gnc.guidance.utils.quintic_hermite import (
    solve_quintic_hermite_coeffs,
)

DEFAULT_GLOBAL_REPLAN_PERIOD_S = 2.0
DEFAULT_T_LOCAL_S = 1.0
DEFAULT_DISTANCE_FALLBACK_M = 0.3
DEFAULT_GUARD_THRESHOLD = 0.05
DEFAULT_Q_ACCEL_STD = 0.01
DEFAULT_R_STD = 0.001


class ReplanningMincoV2Tracker:
    """See module docstring.

    Args:
        p0: initial position, shape ``(3,)`` -- used to seed the estimators
            and perform the first global solve synchronously in the
            constructor (mirrors ``MincoTrajectory``'s own constructor,
            which also solves eagerly and raises on failure, rather than
            ``ReplanningTrajectoryTracker``'s "caller already built the
            initial trajectory" convention -- there is no external trajectory
            type this class could be handed here).
        p_target: target position, shape ``(3,)``.
        pose_fn: callable ``() -> (pos, quat, stamp) | None`` (same contract
            as ``ReplanningTrajectoryTracker``).
        tf_fresh_fn: callable ``(stamp) -> bool`` (same contract).
        q0: reference attitude ``[x,y,z,w]`` (same convention as
            ``MincoTrajectory``); ``p0``'s attitude is assumed to equal
            ``q0`` exactly (rotvec 0), matching ``MincoTrajectory``'s own
            head-attitude convention.
        target_speed, max_accel: forwarded to every global replan's
            ``MincoTrajectory(..., target_speed=..., max_accel=...)``.
            Both non-None selects ``plan_minco_heuristic_time``; both
            ``None`` selects ``MincoTrajectory``'s free-time ``plan_minco``
            path instead (must be given together, same rule as
            ``MincoTrajectory`` itself).
        route_waypoints: optional ordered interior via points, shape
            ``(N, 3)`` -- same one-way retirement semantics as
            ``ReplanningTrajectoryTracker`` (``_next_idx`` reused verbatim).
        global_replan_period: seconds between global replans.
        t_local: local-segment look-ahead horizon, seconds. Must satisfy
            ``0 < t_local < global_replan_period`` (追試15's discovered v4
            design constraint -- violating it causes the look-ahead point to
            sample past the global trajectory's own horizon before the next
            replan arrives, checked at construction).
        distance_fallback_m: remaining-distance threshold below which
            global replanning permanently stops (module docstring).
        via_half_width, wrench_safety_margin, attitude_resample_spacing_m:
            forwarded to every global replan's ``MincoTrajectory(...)``
            (same meaning as ``ReplanningTrajectoryTracker``'s
            ``minco_*`` constructor args).
        q_accel_std_pos, r_std_pos, q_accel_std_rot, r_std_rot: MODEL_KF
            tuning (``ModelKfEstimator``'s ``q_accel_std``/``r_std``), one
            pair for translation, one for rotation. Defaults match the
            offline-verified value (0.01) and a representative assumed
            observation noise (0.001) -- both verified robust across
            several orders of magnitude of mistuning, not knife-edge
            (module docstring / archived robustness doc).
        guard_threshold: ``guard_global_replan_v0``'s velocity-jump
            threshold, m/s (追試16, verified robust across 0.02-0.2).
        initial_v0, initial_w0: initial velocity/angular-velocity, shape
            ``(3,)`` each, default zero (tracker assumed to start at rest,
            matching ``ReplanningTrajectoryTracker``'s convention).

    Raises:
        ValueError: if ``t_local`` is not strictly between 0 and
            ``global_replan_period``.
        MincoInfeasibleError: if the initial global solve fails.
    """

    def __init__(self, p0, p_target, pose_fn, tf_fresh_fn, q0,
                 target_speed, max_accel, route_waypoints=None,
                 global_replan_period=DEFAULT_GLOBAL_REPLAN_PERIOD_S,
                 t_local=DEFAULT_T_LOCAL_S,
                 distance_fallback_m=DEFAULT_DISTANCE_FALLBACK_M,
                 via_half_width=0.3, wrench_safety_margin=1.0,
                 attitude_resample_spacing_m=None,
                 q_accel_std_pos=DEFAULT_Q_ACCEL_STD, r_std_pos=DEFAULT_R_STD,
                 q_accel_std_rot=DEFAULT_Q_ACCEL_STD, r_std_rot=DEFAULT_R_STD,
                 guard_threshold=DEFAULT_GUARD_THRESHOLD,
                 initial_v0=None, initial_w0=None):
        if not (0.0 < t_local < global_replan_period):
            raise ValueError(
                "t_local must satisfy 0 < t_local < global_replan_period "
                "(got t_local=%r, global_replan_period=%r) -- violating "
                "this causes the look-ahead point to outrun the global "
                "trajectory before the next replan (追試15)"
                % (t_local, global_replan_period)
            )
        if (target_speed is None) != (max_accel is None):
            raise ValueError(
                "target_speed and max_accel must be given together (both "
                "None selects MincoTrajectory's free-time plan_minco path, "
                "both non-None selects plan_minco_heuristic_time)"
            )

        self._p_target = np.asarray(p_target, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._q0 = np.asarray(q0, dtype=float)
        self._target_speed = None if target_speed is None else float(target_speed)
        self._max_accel = None if max_accel is None else float(max_accel)
        self._global_replan_period = float(global_replan_period)
        self._t_local = float(t_local)
        self._distance_fallback_m = float(distance_fallback_m)
        self._via_half_width = float(via_half_width)
        self._wrench_safety_margin = float(wrench_safety_margin)
        self._attitude_resample_spacing_m = attitude_resample_spacing_m
        self._guard_threshold = float(guard_threshold)

        self._route_waypoints = (
            np.zeros((0, 3)) if route_waypoints is None
            else np.asarray(route_waypoints, dtype=float).reshape(-1, 3)
        )
        self._next_idx = 0
        self._route_target_dists = np.linalg.norm(
            self._route_waypoints - self._p_target, axis=1
        )

        v0 = np.zeros(3) if initial_v0 is None else np.asarray(initial_v0, dtype=float)
        w0 = np.zeros(3) if initial_w0 is None else np.asarray(initial_w0, dtype=float)
        self._pos_kf = ModelKfEstimator(3, q_accel_std_pos, r_std_pos)
        self._pos_kf.reset(np.asarray(p0, dtype=float), v0)
        self._rot_kf = ModelKfEstimator(3, q_accel_std_rot, r_std_rot)
        self._rot_kf.reset(np.zeros(3), w0)  # p0's attitude == q0 => rotvec 0
        self._prev_rotvec_obs = np.zeros(3)
        self._prev_v0_for_guard = v0.copy()

        self._stop_global_replan = False
        self._fallen_back = False
        self.last_fallback_reason = None
        self.last_replan_occurred = False
        self.last_local_fallback = False

        self._global_trajectory = None
        self._global_elapsed = 0.0
        self._rebuild_global(np.asarray(p0, dtype=float), v0, w0)

        self._local_pos_coeffs = None
        self._local_rot_coeffs = None
        self._local_elapsed = 0.0

        # Terminal (post-latch) local segment: solved ONCE when
        # _stop_global_replan first trips, then only ever sampled further
        # along -- never rebuilt from a fresh (p_now, vel_est) every tick
        # like the pre-latch branch does. A once-per-tick rebuild here would
        # close a recursive loop between the KF's own velocity estimate and
        # this segment's boundary conditions (both always targeting the same
        # fixed (p_target, 0, 0) endpoint); offline verification (docs/
        # 2026-09-18_minco_stretch_fix_sim_verification_facts.md facts 23-24)
        # showed this reproduces the live goal-arrival oscillation even with
        # a lag-free, noise-free self-referential pose_fn -- i.e. the
        # rebuild-every-tick itself is the resonance's proximate cause, not
        # real vehicle dynamics. This matches the module docstring's
        # already-stated intent ("local layer switches to connecting
        # directly to p_target ... same 'final leg is static' outcome") and
        # ``ReplanningTrajectoryTracker``'s condition-1 handling (a single
        # final re-plan, then _fallen_back freezes further replanning and
        # sample() just keeps sampling that one trajectory).
        self._terminal_pos_coeffs = None
        self._terminal_rot_coeffs = None
        self._terminal_duration = None
        self._terminal_elapsed = 0.0

        self._prev_t = 0.0
        p0 = np.asarray(p0, dtype=float)
        self._last_output = (p0, v0, np.zeros(3), self._q0.copy())
        self._last_a_cmd_pos = np.zeros(3)
        self._last_a_cmd_rot = np.zeros(3)

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

        p_now, quat, _stamp = pose
        p_now = np.asarray(p_now, dtype=float)
        quat = np.asarray(quat, dtype=float)

        dt = float(t) - self._prev_t
        if dt <= 0.0:
            # No new information since the last call (e.g. GuidanceExecutor.
            # _run_trajectory clamps its `t` argument at/after
            # total_duration once this tracker enters its final-approach
            # phase, see that property's docstring) -- hold the last output
            # rather than force a bogus tiny dt through the estimator/local
            # segment (which would corrupt the KF's integration if `t`
            # repeats many times while real wall/sim time keeps advancing).
            return self._last_output
        self._prev_t = float(t)

        pos_est, vel_est = self._pos_kf.step(p_now, self._last_a_cmd_pos, dt)
        rotvec_obs = unwrap_rotvec(
            quat_log(quat_mul(quat_conj(self._q0), quat)), self._prev_rotvec_obs
        )
        self._prev_rotvec_obs = rotvec_obs
        rot_est, angvel_est = self._rot_kf.step(rotvec_obs, self._last_a_cmd_rot, dt)

        distance = float(np.linalg.norm(self._p_target - p_now))
        if not self._stop_global_replan:
            while (self._next_idx < len(self._route_waypoints)
                   and distance < self._route_target_dists[self._next_idx]):
                self._next_idx += 1

            if distance < self._distance_fallback_m:
                self._stop_global_replan = True
            else:
                self._global_elapsed += dt
                if self._global_elapsed >= self._global_replan_period:
                    v0_prev_tick = self._prev_v0_for_guard
                    v0_for_global = vel_est
                    if np.linalg.norm(vel_est - v0_prev_tick) > self._guard_threshold:
                        v0_for_global = v0_prev_tick
                    try:
                        self._rebuild_global(p_now, v0_for_global, angvel_est)
                        self.last_replan_occurred = True
                    except MincoInfeasibleError:
                        self.last_fallback_reason = "minco_infeasible"
                    self._global_elapsed = 0.0
        self._prev_v0_for_guard = vel_est

        if self._stop_global_replan:
            if self._terminal_pos_coeffs is None:
                tail_t = self._global_trajectory.global_total_duration
                tgt_rv, _tail_rv_vel, _tail_rv_accel = (
                    self._global_trajectory.sample_rotvec_derivatives(tail_t)
                )
                remaining = float(np.linalg.norm(self._p_target - p_now))
                # Floor at `dt` (not `t_local`): sizing this from actual
                # remaining distance/target_speed keeps convergence timing
                # close to the pre-fix per-tick behavior. A `t_local` floor
                # here was tried first and regressed test_guidance_executor.
                # py::test_execute_replanning_minco_v2_mode_reaches_target
                # (it forced a needless >=1s terminal duration even when
                # remaining distance was small, so the setpoint hadn't
                # finished converging by the time the real TF -- which
                # doesn't follow this setpoint in that test -- already had).
                # Free-time path (target_speed=None) has no target_speed to
                # size this from -- use the last global trajectory's own
                # average speed instead (see _rebuild_global).
                speed_for_terminal = (
                    self._target_speed if self._target_speed is not None
                    else self._freetime_avg_speed
                )
                terminal_duration = max(
                    remaining / max(speed_for_terminal, 1e-6), dt
                )
                try:
                    self._terminal_pos_coeffs = solve_quintic_hermite_coeffs(
                        p_now, vel_est, np.zeros(3),
                        self._p_target, np.zeros(3), np.zeros(3),
                        terminal_duration,
                    )
                    self._terminal_rot_coeffs = solve_quintic_hermite_coeffs(
                        rotvec_obs, angvel_est, np.zeros(3),
                        tgt_rv, np.zeros(3), np.zeros(3),
                        terminal_duration,
                    )
                    self._terminal_duration = terminal_duration
                    self._terminal_elapsed = 0.0
                except ValueError:
                    if self._local_pos_coeffs is None:
                        raise
                    self.last_local_fallback = True
            else:
                self._terminal_elapsed += dt
            if self._terminal_pos_coeffs is not None:
                local_pos_coeffs = self._terminal_pos_coeffs
                local_rot_coeffs = self._terminal_rot_coeffs
                tau = min(self._terminal_elapsed, self._terminal_duration)
            else:
                local_pos_coeffs = self._local_pos_coeffs
                local_rot_coeffs = self._local_rot_coeffs
                tau = self._local_elapsed
        else:
            lookahead_t = self._global_elapsed + self._t_local
            tgt_p, tgt_v, tgt_a, _q = self._global_trajectory.sample(lookahead_t)
            tgt_rv, tgt_rv_vel, tgt_rv_accel = (
                self._global_trajectory.sample_rotvec_derivatives(lookahead_t)
            )

            try:
                local_pos_coeffs = solve_quintic_hermite_coeffs(
                    p_now, vel_est, np.zeros(3), tgt_p, tgt_v, tgt_a, self._t_local
                )
                local_rot_coeffs = solve_quintic_hermite_coeffs(
                    rotvec_obs, angvel_est, np.zeros(3), tgt_rv, tgt_rv_vel, tgt_rv_accel,
                    self._t_local,
                )
                self._local_pos_coeffs = local_pos_coeffs
                self._local_rot_coeffs = local_rot_coeffs
                self._local_elapsed = dt
            except ValueError:
                if self._local_pos_coeffs is None:
                    raise
                self.last_local_fallback = True
                self._local_elapsed += dt
                local_pos_coeffs = self._local_pos_coeffs
                local_rot_coeffs = self._local_rot_coeffs

            tau = self._local_elapsed
        p_out = evaluate_vector(local_pos_coeffs, tau, order=0)
        v_out = evaluate_vector(local_pos_coeffs, tau, order=1)
        a_out = evaluate_vector(local_pos_coeffs, tau, order=2)
        rv_out = evaluate_vector(local_rot_coeffs, tau, order=0)
        angvel_out = evaluate_vector(local_rot_coeffs, tau, order=1)
        angaccel_out = evaluate_vector(local_rot_coeffs, tau, order=2)

        self._last_a_cmd_pos = a_out
        self._last_a_cmd_rot = angaccel_out
        q_out = quat_mul(self._q0, quat_exp(rv_out))
        self._last_output = (p_out, v_out, a_out, q_out)
        return self._last_output

    @property
    def total_duration(self):
        """Analogous to ``ReplanningTrajectoryTracker.total_duration`` --
        read by ``GuidanceExecutor._run_trajectory`` both to clamp its ``t``
        argument to ``sample()`` and to decide when to start polling for
        real TF convergence (once ``elapsed >= total_duration``). Unlike
        the single-layer tracker (a genuine trajectory horizon), this
        design has no single fixed horizon -- the local layer is rebuilt
        every tick indefinitely.

        A first version of this simply froze at ``self._prev_t`` (no
        margin) once global replanning permanently stopped
        (``distance_fallback_m`` latch) -- this was wrong: it made
        ``_run_trajectory`` clamp its ``t`` argument to that same frozen
        value on every subsequent tick, which (via ``sample()``'s ``dt<=0``
        no-op handling) froze this tracker's *own commanded output* at
        whatever it was the instant the latch tripped, even though the
        vehicle had not actually arrived yet -- in real operation (not just
        a test where a TF fake advances independently of what's commanded)
        this would stop advancing the setpoint before reaching the target.

        Instead this returns a genuine estimated time-to-arrival (remaining
        distance from the KF's own position state to ``p_target``, over
        ``target_speed``), added to the last ``t`` seen. This value shrinks
        toward ``self._prev_t`` exactly as the vehicle actually nears the
        target (not on an arbitrary latch), so ``sample()`` keeps receiving
        undistorted, un-clamped real elapsed time for as long as there is
        genuinely still ground to cover, and only lets the clamp
        (harmlessly, by then) bite once the estimated remaining time drops
        below a single tick -- which is also exactly when
        ``_run_trajectory``'s real-TF convergence poll should reasonably
        start."""
        remaining = float(np.linalg.norm(self._p_target - self._pos_kf.pos))
        # Free-time path (target_speed=None) has no target_speed to size
        # this from -- use the same fallback as sample()'s terminal-duration
        # branch above (_rebuild_global's route-average speed).
        speed_for_eta = (
            self._target_speed if self._target_speed is not None
            else self._freetime_avg_speed
        )
        eta = remaining / max(speed_for_eta, 1e-6)
        return self._prev_t + eta

    @property
    def trajectory(self):
        """The current global trajectory (read-only, e.g. for an RViz
        preview) -- analogous to ``ReplanningTrajectoryTracker.trajectory``,
        but this is *only* the global layer; the local layer is rebuilt
        every tick and not exposed as a standalone object."""
        return self._global_trajectory

    def _rebuild_global(self, p_now, v0, w0):
        pending = self._route_waypoints[self._next_idx:]
        if len(pending):
            waypoints = np.vstack([p_now, pending, self._p_target])
        else:
            waypoints = np.array([p_now, self._p_target])
        self._global_trajectory = MincoTrajectory(
            waypoints, self._q0, v0=v0, w0=w0,
            via_half_width=self._via_half_width,
            attitude_resample_spacing_m=self._attitude_resample_spacing_m,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=self._target_speed, max_accel=self._max_accel,
        )
        # Free-time path (target_speed=None) has no fixed cruise speed to
        # size the terminal-approach/total_duration ETA estimates from
        # (module docstring's speed_for_terminal/speed_for_eta) -- using the
        # live velocity estimate there instead breaks whenever it is
        # transiently near zero (e.g. right at start-of-motion), producing a
        # divide-by-near-zero ETA. Use this trajectory's own average speed
        # (route length / duration) instead: well-defined immediately after
        # every rebuild, never collapses to zero unless start==target.
        route_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        self._freetime_avg_speed = route_length / self._global_trajectory.global_total_duration
