"""Prototype of B-1 (stop without waiting for the replan), B-2 (judge a landed solve on the latest
grid instead of re-solving a pre-collision one) and B-4 (keep replanning while braking and switch to
a collision-free local instead of stopping) on top of ReplanMincoTracker, for offline checks only.
See docs/2026-09-26_obstacle_emergency_stop_and_recovery.md section 5.
"""
import collections
import threading

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


class EarlyStopTracker(ReplanMincoTracker):
    def __init__(self, *args, wait_fallback_s=1.0, wait_margin=1.2, wait_window=10,
                 use_b1=True, use_b2=True, use_b4=True, **kwargs):
        self._recent_waits = collections.deque(maxlen=wait_window)
        self._wait_fallback_s = wait_fallback_s
        self._wait_margin = wait_margin
        self._use_b1, self._use_b2, self._use_b4 = use_b1, use_b2, use_b4
        self._brake_solve = False
        self._brake_solve_t = 0.0
        self._brake_switch = None
        self.early_stops = 0
        self.brake_resumes = 0
        self.brake_solves = 0
        self.brake_outcomes = collections.Counter()
        super().__init__(*args, **kwargs)

    def expected_wait_s(self):
        if not self._recent_waits:
            return self._wait_fallback_s
        return self._wait_margin * max(self._recent_waits)

    def _adopt_local(self, result, lag):
        if self._async_replan and lag > 0.0:
            self._recent_waits.append(lag)
        super()._adopt_local(result, lag)

    # ---- B-1 ----
    def _check_collision(self):
        super()._check_collision()
        if (self._use_b1 and self._async_replan and self._stop_profile is None
                and self.last_collision_ahead_s is not None and self._stop_profile_fn is not None
                and self._braking_after_wait_collides()):
            self._stop_now()

    def _braking_after_wait_collides(self):
        local, grid = self._local_trajectory, self._planner.obstacle_grid
        t_wait = self.expected_wait_s()
        t_brake = min(self._local_elapsed + t_wait, local.global_total_duration)
        for t in np.arange(self._local_elapsed, t_brake, 0.05):
            if grid.inflated_occupied(list(local.sample(t)[0])):
                return True
        p, v, _a, q = local.sample(t_brake)
        omega, _alpha = local.sample_body_angular(t_brake)
        profile = self._stop_profile_fn(p, v, q, omega)
        for t in np.append(np.arange(0.0, profile.duration, 0.1), profile.duration):
            if grid.inflated_occupied(list(profile.sample(t)[0])):
                return True
        return False

    def _stop_now(self):
        p_ref, v_ref, _a, q_ref = self._local_trajectory.sample(self._local_elapsed)
        omega_ref, _alpha = self._local_trajectory.sample_body_angular(self._local_elapsed)
        self._stop_profile = self._stop_profile_fn(p_ref, v_ref, q_ref, omega_ref)
        self._stop_elapsed = 0.0
        self.emergency_stops += 1
        self.early_stops += 1
        self.last_fallback_reason = "emergency_stop"

    # ---- B-2 ----
    def _resolve_collision_replan(self):
        if not self._use_b2:
            return super()._resolve_collision_replan()
        self._collision_replan_pending = False
        self.last_collision_ahead_s = self._collision_ahead_s()
        if self.last_collision_ahead_s is None:
            return
        self._emergency_stop_if_collision_close()
        if self._stop_profile is None:
            self._collision_replan_pending = True
            self._start_background_replan()

    # ---- B-4 (and dropping stale solves while braking) ----
    def _local_is_free(self, local, touches_goal):
        grid = self._planner.obstacle_grid
        end = local.global_total_duration * (1.0 if touches_goal else 0.75)
        speed = self._planner.local_max_vel or self._planner.global_avg_speed
        t_step = grid.resolution / 2.0 / max(speed, 1e-6)
        t = 0.0
        while t <= end:
            if grid.inflated_occupied(list(local.sample(t)[0])):
                return False
            t += t_step
        return True

    def _judge_brake_solve(self, result, t_now):
        if result is None or isinstance(result, Exception):
            return "failed"
        if t_now > self._brake_solve_t + 1e-9:
            return "late"
        if not self._local_is_free(*result):
            return "collides"
        return "switch"

    def _start_brake_solve(self, t_now):
        profile = self._stop_profile
        t_switch = min(t_now + self.expected_wait_s(), profile.duration)
        p, v, a, q, _omega, _alpha = profile.sample(t_switch)
        rv = quat_log(quat_mul(quat_conj(self._q0), np.asarray(q, dtype=float)))
        state = (np.asarray(p), np.asarray(v), np.asarray(a), rv, np.zeros(3), np.zeros(3), None, 0.0)
        self._brake_solve = True
        self._brake_solve_t = t_switch
        self._pending_lag = 0.0
        self._pending_is_collision_replan = False
        self._pending_thread = threading.Thread(
            target=self._solve_local_in_background, args=(state,), daemon=True)
        self._pending_thread.start()
        self.brake_solves += 1

    def _sample_emergency_stop(self, dt):
        t_now = self._stop_elapsed + dt
        if self._pending_thread is not None:
            if self._pending_thread.is_alive():
                # build_local mutates planner state: no rest replan while a solve is running.
                self._since_replan_attempt = 0.0
            else:
                result, was_brake = self._pending_result, self._brake_solve
                self._pending_thread = None
                self._pending_result = None
                self._collision_replan_pending = False
                self._brake_solve = False
                if was_brake and self._use_b4:
                    outcome = self._judge_brake_solve(result, t_now)
                    self.brake_outcomes[outcome] += 1
                    if outcome == "switch":
                        self._brake_switch = (self._brake_solve_t, result)
        if self._brake_switch is not None and t_now >= self._brake_switch[0] - 1e-9:
            t_switch, result = self._brake_switch
            self._brake_switch = None
            self._stop_profile = None
            self._rest_replan_failures = 0
            ReplanMincoTracker._adopt_local(self, result, t_now - t_switch)
            self.brake_resumes += 1
            p, v, a, q = self._local_trajectory.sample(self._local_elapsed)
            self.last_body_angular = self._local_trajectory.sample_body_angular(self._local_elapsed)
            self._last_output = (p, v, a, q)
            return self._last_output
        if (self._use_b4 and self._pending_thread is None and self._brake_switch is None
                and t_now < self._stop_profile.duration):
            self._start_brake_solve(t_now)
            self._since_replan_attempt = 0.0
        return super()._sample_emergency_stop(dt)
