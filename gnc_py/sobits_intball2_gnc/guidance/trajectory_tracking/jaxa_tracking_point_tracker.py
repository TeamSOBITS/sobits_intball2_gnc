#!/usr/bin/env python3
"""Tracking-point tracker of the JAXA Int-Ball2 baseline (ROS-agnostic).

Drives :mod:`~sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner`
(IAC-22-D1.6.2 3.1-3.2) behind the ``BaseTrajectoryTracker`` contract:
``sample(t)`` returns the paper's tracking point as ``p_des`` with
``v_des = a_des = 0`` (feedback only), so there is no timed trajectory.

- The global path is the straight line from the start to the goal, or, with
  ``reference_route``, the shared pre-departure route (A* shortcut polyline,
  the one the MINCO tracker gets) with its ends set to the start and the goal
  and densified to ``REFERENCE_SPACING_M``: the tracking point takes the
  nearest vertex's segment, so a sparse polyline cuts every corner from far
  before it (docs/2026-10-05_jaxa_astar_global_corner_offline.md). The shape is
  not changed and corners are not rounded. It is followed as is while it is
  free; the local planner runs only
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
  the goal is the target (``attitude_mode="goal"``, the default). The camera axis
  is pointed with ``compute_q_des``, which keeps the current roll (as pre_align
  and face travel do), so a level start stays level whichever way up the vehicle
  is. ``attitude_mode="path"`` (ours; not in the paper) points it at the point
  ``path_facing_ahead_m`` along the current path instead, as replan_minco's face
  travel does, turning at most ``path_facing_max_rate_deg`` per second. Within
  ``goal_facing_hold_m`` of the goal the last attitude is held in both modes,
  since the direction flips when the vehicle passes the goal (ours; not in the
  paper).
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
from sobits_intball2_gnc.guidance.search.reference_route import densify
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, slerp
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

_ZERO3 = np.zeros(3)
# Same spacing the MINCO tracker densifies the shared route to.
REFERENCE_SPACING_M = 0.25
# Rejoin this far past the last blocked sample of the global path: with depth only the seen
# face of an obstacle is occupied, so the first free sample can lie inside it (ours, not in
# the paper; 0.6 m box depth + 2 x 0.2 m inflation).
_REJOIN_MARGIN_M = 1.0
# Each replan draws fresh RRT* seeds (attempts use seed .. seed + max_attempts - 1).
_SEED_STRIDE = 100


ATTITUDE_MODES = ("goal", "path")


def point_ahead(path, p, ahead):
    """Point ``ahead`` metres along ``path`` past the point of it nearest to ``p``."""
    best, k_best, t_best = np.inf, 0, 0.0
    for k, (a, b) in enumerate(zip(path[:-1], path[1:])):
        d = b - a
        sq = float(np.dot(d, d))
        t = 0.0 if sq < 1e-12 else float(np.clip(np.dot(p - a, d) / sq, 0.0, 1.0))
        dist = float(np.linalg.norm(a + t * d - p))
        if dist < best:
            best, k_best, t_best = dist, k, t
    q = path[k_best] + t_best * (path[k_best + 1] - path[k_best])
    left = float(ahead)
    for k in range(k_best, len(path) - 1):
        seg = path[k + 1] - q
        n = float(np.linalg.norm(seg))
        if n >= left:
            return q + seg * left / n
        left -= n
        q = path[k + 1]
    return np.asarray(path[-1], dtype=float)


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
        reference_route: shared route polyline (>= 2 points) used as the global
            path in place of the start-goal line; its ends are replaced by
            ``p0`` and ``p_target``, as the MINCO tracker does.
        seed: first RRT* seed.
        attitude_mode: ``"goal"`` faces the goal, ``"path"`` faces the path ahead
            (see the module docstring).
        path_facing_ahead_m, path_facing_max_rate_deg: ``"path"`` only; how far
            along the path to look and the turn rate limit [deg/s].

    Raises :class:`JaxaPlanError` if the initial plan fails.
    """

    def __init__(self, p0, p_target, pose_fn, tf_fresh_fn, q0, obstacle_grid, bounds,
                 lookahead_m, config, collision_check_period, goal_facing_hold_m,
                 forward_axis=(1.0, 0.0, 0.0), async_replan=True, seed=0, reference_route=None,
                 attitude_mode="goal", path_facing_ahead_m=0.5, path_facing_max_rate_deg=20.0):
        if attitude_mode not in ATTITUDE_MODES:
            raise ValueError("attitude_mode must be one of %s, got %r" % (ATTITUDE_MODES, attitude_mode))
        if path_facing_ahead_m <= 0.0 or path_facing_max_rate_deg <= 0.0:
            raise ValueError("path_facing_ahead_m and path_facing_max_rate_deg must be positive")
        self._attitude_mode = attitude_mode
        self._facing_ahead = float(path_facing_ahead_m)
        self._facing_max_rate = np.radians(float(path_facing_max_rate_deg))
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
        if reference_route is None:
            self._global_path = np.array([p0, self._goal], dtype=float)
        else:
            route = np.asarray(reference_route, dtype=float).reshape(-1, 3)
            if len(route) < 2:
                raise ValueError("reference_route needs at least a start and a goal")
            route = np.vstack((np.asarray(p0, dtype=float), route[1:-1], self._goal))
            self._global_path = np.asarray(densify(route, REFERENCE_SPACING_M), dtype=float)
        legs = np.linalg.norm(np.diff(self._global_path, axis=0), axis=1)
        self._global_s = np.concatenate(([0.0], np.cumsum(legs)))
        if path_is_free(self._global_path, self._grid, self._bounds):
            path, info = self._global_path.copy(), {"plan_s": 0.0, "attempts": 0}
        else:
            path, info = self._solve(p0)
        self._path = path
        if self._attitude_mode == "path":
            self._q = self._facing_path(np.asarray(p0, dtype=float), self._q, rate_limited=False)
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

    def _facing_path(self, p, q_prev, rate_limited, dt=0.0):
        direction = point_ahead(self._path, p, self._facing_ahead) - p
        if np.linalg.norm(direction) < 1e-6:
            return np.asarray(q_prev, dtype=float)
        q_new = np.asarray(compute_q_des(direction, q_prev, 0.0, self._forward_axis), dtype=float)
        if not rate_limited:
            return q_new
        step = self._facing_max_rate * dt
        angle = geodesic_angle(q_prev, q_new)
        if angle <= step or angle < 1e-9:
            return q_new
        return np.asarray(slerp(q_prev, q_new, step / angle), dtype=float)

    def _next_attitude(self, p, dt):
        if np.linalg.norm(self._goal - p) <= self._hold_m:
            return self._q
        if self._attitude_mode == "goal":
            return self._facing_goal(p, self._q)
        return self._facing_path(p, self._q, rate_limited=True, dt=dt)

    def set_obstacle_grid(self, grid):
        self._grid = grid
        self._since_check = np.inf

    def _solve(self, start):
        self._seed += _SEED_STRIDE
        start = np.asarray(start, dtype=float)
        rejoin, s_rejoin = self._rejoin_point(start)
        path, info = plan_local_path(start, rejoin, self._grid, self._bounds, self._seed, self._config)
        if s_rejoin < self._global_s[-1]:
            path = np.vstack((path, self._global_path[self._global_s > s_rejoin + 1e-9]))
        return path, info

    def _global_at(self, s):
        k = min(int(np.searchsorted(self._global_s, s, side="right")) - 1, len(self._global_path) - 2)
        a, b = self._global_path[k], self._global_path[k + 1]
        leg = self._global_s[k + 1] - self._global_s[k]
        return a if leg < 1e-12 else a + (b - a) * (s - self._global_s[k]) / leg

    def _global_arc_length(self, p):
        """Arc length of the point of the global path nearest to ``p``."""
        best, s_best = np.inf, 0.0
        for k, (a, b) in enumerate(zip(self._global_path[:-1], self._global_path[1:])):
            d = b - a
            sq = float(np.dot(d, d))
            t = 0.0 if sq < 1e-12 else float(np.clip(np.dot(p - a, d) / sq, 0.0, 1.0))
            dist = float(np.linalg.norm(a + t * d - p))
            if dist < best:
                best, s_best = dist, self._global_s[k] + t * (self._global_s[k + 1] - self._global_s[k])
        return s_best

    def _rejoin_point(self, p):
        """``(point, arc length)`` ``_REJOIN_MARGIN_M`` past the last blocked stretch of the
        global path ahead of ``p`` (sampled at half a voxel, as path_is_free), or the goal when
        none is blocked, the goal itself is (the planner then reports it) or the margin reaches
        past it."""
        goal, total = self._global_path[-1], float(self._global_s[-1])
        if total < 1e-9:
            return goal, total
        s0 = self._global_arc_length(p)
        ss = np.append(np.arange(s0, total, 0.5 * float(self._grid.resolution)), total)
        lo, hi = (np.asarray(b, dtype=float) for b in self._bounds)
        blocked = [bool(np.any(q < lo) or np.any(q > hi) or self._grid.inflated_occupied(q.tolist()))
                   for q in (self._global_at(s) for s in ss)]
        if not any(blocked) or blocked[-1]:
            return goal, total
        last = len(blocked) - 1 - blocked[::-1].index(True)
        s_rejoin = ss[last] + _REJOIN_MARGIN_M
        if s_rejoin >= total:
            return goal, total
        for s_free, b in zip(ss[last + 1:], blocked[last + 1:]):
            if s_free >= s_rejoin and not b:
                return self._global_at(s_free), float(s_free)
        return goal, total

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
        self._q = self._next_attitude(p, dt)
        if self._goal_reached_t is None and np.linalg.norm(target - self._goal) < 1e-9:
            self._goal_reached_t = t
        self._p_hold = target
        return target.copy(), _ZERO3.copy(), _ZERO3.copy(), self._q.copy()
