#!/usr/bin/env python3
"""Tracking-point tracker of the JAXA Int-Ball2 baseline (ROS-agnostic).

Drives :mod:`~sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner`
(IAC-22-D1.6.2 3.1-3.2) behind the ``BaseTrajectoryTracker`` contract:
``sample(t)`` returns the paper's tracking point as ``p_des`` with
``v_des = a_des = 0`` (feedback only), so there is no timed trajectory.

- The global path is the straight line from the start to the goal (no via
  points). It is followed as is while it is free; the local planner runs only
  when it is blocked (paper Sec. 3: a local path is generated "when obstacles
  exist on the predefined global path"). The local path rejoins the global
  path (paper Fig. 2(a)): RRT* runs from the vehicle to the point
  ``_REJOIN_MARGIN_M`` past the last blocked stretch of the global path ahead
  of the vehicle (the goal if that is beyond it), and
  the global path continues from there to the goal. The rejoin rule is ours;
  with nothing blocked on the global path ahead (only the current path
  collides) the local path goes to the goal.
- Replans when the remaining path collides (checked every
  ``collision_check_period``). With ``async_replan`` the solve runs in a
  background thread and the old path is followed until it finishes, since a
  solve must not stall the 50 Hz setpoint loop. Native planning releases the
  GIL and reads a snapshot so depth integration can continue.
- Attitude: the paper's shot orientation planner points the camera at a target
  independently of the path, keeping the image level; with no shooting target
  the goal is the target. The camera axis is pointed with ``compute_q_des``,
  which keeps the current roll (as pre_align and face travel do), so a level
  start stays level whichever way up the vehicle is. Within
  ``goal_facing_hold_m`` of the goal the last attitude is held, since the
  direction flips when the vehicle passes the goal (ours; not in the paper).
- ``total_duration`` is ``inf`` until the tracking point has reached the goal,
  then that time, so the executor's position-convergence check starts there.
- A failed plan sets ``failed`` (reason); the executor ends the goal.

Design: docs/jaxa_baseline_gazebo_port_plan.md.
"""
import threading
import time

import numpy as np

from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import (
    JaxaPlanError,
    path_is_free,
    plan_local_path,
    tracking_point,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

_ZERO3 = np.zeros(3)
# Rejoin this far past the last blocked sample of the global path: with depth only the seen
# face of an obstacle is occupied, so the first free sample can lie inside it (ours, not in
# the paper; 0.6 m box depth + 2 x 0.2 m inflation).
_REJOIN_MARGIN_M = 1.0
# Each replan draws fresh RRT* seeds (attempts use seed .. seed + max_attempts - 1).
_SEED_STRIDE = 100


class JaxaTrackingPointTracker:
    """See module docstring.

    Args:
        p0, p_target: start and goal positions.
        pose_fn: ``() -> (pos, quat, stamp)`` or ``None`` (TF).
        tf_fresh_fn: ``(stamp) -> bool``; stale poses hold the last setpoint.
        q0: current attitude (its roll is kept).
        forward_axis: body camera axis pointed at the goal.
        obstacle_grid: inflated occupancy grid (``inflated_occupied``).
        bounds: ``(lower, upper)`` RRT* sampling box.
        lookahead_m: the paper's ``d``.
        config: :class:`JaxaPlannerConfig`.
        collision_check_period, goal_facing_hold_m: see module docstring.
        async_replan: solve replans in a background thread.
        seed: first RRT* seed.

    Raises :class:`JaxaPlanError` if the initial plan fails.
    """

    def __init__(self, p0, p_target, pose_fn, tf_fresh_fn, q0, obstacle_grid, bounds,
                 lookahead_m, config, collision_check_period, goal_facing_hold_m,
                 forward_axis=(1.0, 0.0, 0.0), async_replan=True, seed=0):
        self._goal = np.asarray(p_target, dtype=float)
        self._pose_fn = pose_fn
        self._tf_fresh_fn = tf_fresh_fn
        self._grid = obstacle_grid
        self._bounds = bounds
        self._lookahead = float(lookahead_m)
        self._config = config
        self._check_period = float(collision_check_period)
        self._hold_m = float(goal_facing_hold_m)
        self._async = bool(async_replan)
        self._seed = int(seed)
        self._forward_axis = np.asarray(forward_axis, dtype=float)
        self._q = self._facing_goal(np.asarray(p0, dtype=float), np.asarray(q0, dtype=float))
        self._p_hold = np.asarray(p0, dtype=float)
        self._last_t = None
        self._since_check = np.inf
        self._goal_reached_t = None
        self._pending_thread = None
        self._pending_result = None
        # Sim time waited on the running background solve (name shared with
        # ReplanMincoTracker so offline harnesses can pace both the same way).
        self._pending_lag = 0.0
        self.failed = None
        self.last_body_angular = (_ZERO3.copy(), _ZERO3.copy())
        self.last_replan_occurred = False
        self.last_replan_solve_seconds = None
        self.last_replan_lag_seconds = None
        self.last_replan_source = None
        self.last_replan_attempts = None
        self.plans = []
        self._global_path = np.array([p0, self._goal], dtype=float)
        if path_is_free(self._global_path, self._grid, self._bounds):
            path, info = self._global_path.copy(), {"plan_s": 0.0, "attempts": 0}
        else:
            path, info = self._solve(p0)
        self._path = path
        self.plans.append(info)
        self.last_replan_solve_seconds = info["plan_s"]

    @property
    def total_duration(self):
        return np.inf if self._goal_reached_t is None else self._goal_reached_t

    @property
    def path(self):
        return self._path

    @property
    def goal_position(self):
        return self._goal

    @property
    def replanning_stopped(self):
        return self.failed is not None

    def _facing_goal(self, p, q_prev):
        return np.asarray(compute_q_des(self._goal - p, q_prev, 0.0, self._forward_axis), dtype=float)

    def set_obstacle_grid(self, grid):
        self._grid = grid
        self._since_check = np.inf

    def _solve(self, start):
        self._seed += _SEED_STRIDE
        start = np.asarray(start, dtype=float)
        rejoin = self._rejoin_point(start)
        path, info = plan_local_path(start, rejoin, self._grid, self._bounds, self._seed, self._config)
        if np.linalg.norm(rejoin - self._goal) > 1e-9:
            path = np.vstack((path, self._goal))
        return path, info

    def _rejoin_point(self, p):
        """Point ``_REJOIN_MARGIN_M`` past the last blocked stretch of the global path ahead of
        ``p`` (sampled at half a voxel, as path_is_free), or the goal when none is blocked, the
        goal itself is (the planner then reports it) or the margin reaches past it."""
        origin, goal = self._global_path
        length = float(np.linalg.norm(goal - origin))
        if length < 1e-9:
            return goal
        u = (goal - origin) / length
        s0 = float(np.clip(np.dot(p - origin, u), 0.0, length))
        ss = np.append(np.arange(s0, length, 0.5 * float(self._grid.resolution)), length)
        lo, hi = (np.asarray(b, dtype=float) for b in self._bounds)
        blocked = [bool(np.any(q < lo) or np.any(q > hi) or self._grid.inflated_occupied(q.tolist()))
                   for q in (origin + u * s for s in ss)]
        if not any(blocked) or blocked[-1]:
            return goal
        last = len(blocked) - 1 - blocked[::-1].index(True)
        s_rejoin = ss[last] + _REJOIN_MARGIN_M
        if s_rejoin >= length:
            return goal
        for s_free, b in zip(ss[last + 1:], blocked[last + 1:]):
            if s_free >= s_rejoin and not b:
                return origin + u * s_free
        return goal

    def _solve_in_background(self, start):
        try:
            self._pending_result = self._solve(start)
        except JaxaPlanError as error:
            self._pending_result = error

    def _start_replan(self, p):
        if not self._async:
            t0 = time.perf_counter()
            try:
                result = self._solve(p)
            except JaxaPlanError as error:
                result = error
            self._adopt(result, 0.0, "sync", time.perf_counter() - t0)
            return
        self._pending_lag = 0.0
        self._pending_result = None
        self._pending_thread = threading.Thread(
            target=self._solve_in_background, args=(np.asarray(p, dtype=float),), daemon=True)
        self._pending_thread.start()

    def _adopt(self, result, lag, source, wall_s=None):
        if isinstance(result, JaxaPlanError):
            self.failed = str(result)
            self.plans.append({"plan_s": float("nan") if wall_s is None else wall_s, "attempts": 0})
            return
        path, info = result
        self._path = path
        self.plans.append(info)
        self.last_replan_occurred = True
        self.last_replan_solve_seconds = info["plan_s"]
        self.last_replan_lag_seconds = lag
        self.last_replan_source = source
        self.last_replan_attempts = info["attempts"]
        self._since_check = np.inf

    def sample(self, t):
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self.last_replan_occurred = False
        self._since_check += dt
        if self._pending_thread is not None:
            self._pending_lag += dt
            if not self._pending_thread.is_alive():
                self._pending_thread = None
                self._adopt(self._pending_result, self._pending_lag, "async")
        pose = self._pose_fn()
        fresh = pose is not None and self._tf_fresh_fn(pose[2])
        if self.failed is not None or not fresh:
            return self._p_hold.copy(), _ZERO3.copy(), _ZERO3.copy(), self._q.copy()
        p = np.asarray(pose[0], dtype=float)
        target, i = tracking_point(self._path, p, self._lookahead)
        if self._pending_thread is None and self._since_check >= self._check_period - 1e-9:
            self._since_check = 0.0
            remaining = self._path[i:]
            if len(remaining) >= 2 and not path_is_free(remaining, self._grid, self._bounds):
                self._start_replan(p)
                if self.failed is not None:
                    self._p_hold = p
                    return self._p_hold.copy(), _ZERO3.copy(), _ZERO3.copy(), self._q.copy()
                target, i = tracking_point(self._path, p, self._lookahead)
        if np.linalg.norm(self._goal - p) > self._hold_m:
            self._q = self._facing_goal(p, self._q)
        if self._goal_reached_t is None and np.linalg.norm(target - self._goal) < 1e-9:
            self._goal_reached_t = t
        self._p_hold = target
        return target.copy(), _ZERO3.copy(), _ZERO3.copy(), self._q.copy()
