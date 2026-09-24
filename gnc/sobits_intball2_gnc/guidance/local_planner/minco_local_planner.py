#!/usr/bin/env python3
"""MINCO global/local planner behind ``ReplanningMincoV3Tracker`` (ROS-agnostic, pure).

EGO-Planner v2 splits replanning into ``ego_replan_fsm`` (when to replan, collision
checks, emergency stop) and ``planner_manager`` (how to build the trajectories).
This is the latter; ``trajectory_tracking/replanning_minco_v3_tracker.py`` is the
former.

- **Global**: built **once**, at construction, from ``p0`` + ``route_waypoints``
  + ``p_target`` via :class:`~sobits_intball2_gnc.guidance.trajectory.
  minco_trajectory.MincoTrajectory` (heuristic-time if ``target_speed``/
  ``max_accel`` given, free-time ``plan_minco`` otherwise). Only rebuilt when the
  goal is moved out of an obstacle (:meth:`move_goal_out_of_obstacle`) -- no
  via-waypoint retirement logic is needed because there is no periodic rebuild to
  retire waypoints *for* (offline verification 追記5: reasoned to be unnecessary
  for a build-once global, not independently implementation-verified with
  multiple via points).
- **Local** (:meth:`build_local`): a fresh free-time ``MincoTrajectory`` from the
  caller's head state to a look-ahead point ``planning_horizon_m`` ahead along the
  global trajectory, ending at the global trajectory's velocity there
  (EGO-Planner v2 ``getLocalTarget``; zero at the goal or once inside braking
  distance of it). Starting from the measured state with zero acceleration and
  ending at rest instead -- the original port -- made the vehicle crawl, overshoot
  the goal, and absorb disturbance-induced velocity into every plan as lateral
  drift (docs/2026-09-23_replanning_minco_v3_sim_verification_result_and_root_cause.md).
  (:meth:`_get_local_target`, ported from ``prototype_ego_v2_style_local_replan.py``'s
  ``get_local_target``.) Without ``face_travel`` it is a single segment (K=1),
  the offline-verified baseline; K>1 was offline-verified to help (~29.7% faster
  convergence, offline verification fact 51) but combining it with warm start
  showed no measurable benefit (fact 54).

Attitude: with ``face_travel=False`` (default) every ``MincoTrajectory`` built
here is fixed at ``q0`` and ``w0=0``. With ``face_travel=True`` the global faces
travel, and each local is solved twice: once for its own path (optionally
multi-piece, EGO-Planner v2 ``polyTraj_piece_length``/``computeInitState``, with
``obstacle_grid`` rebound), then re-solved through points on that path with
attitude waypoints facing its own tangent (not the global attitude, so it stays
valid once the local deviates, e.g. for obstacles), carrying the head rotvec
rate/accel. It also caps local speed (the split local otherwise accelerates to its
braking limit every period) and checks the wrench envelope in the body frame
(``docs/archive/achieved/2026-09-23_replanning_minco_v3_face_travel_handoff.md``).
``attitude_resample_spacing_m`` is still forwarded to the *global* build for
spatial densify (the global route's own smoothness, independent of whether
attitude is tracked) -- this only works because of the ``face_travel``/densify
decoupling fix in ``minco_trajectory.py`` (2026-09-23, ``docs/
2026-09-20_ego_v2_style_replan_migration_plan.md`` "やるべき内容1"); before that
fix, ``face_travel=False`` silently disabled densify.
"""
import math

import numpy as np

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

# Resolution for the internal forward-scan of the (already-solved, analytic)
# global trajectory in _get_local_target -- not a control-loop timing
# decision (no clock/sleep involved), just how finely that scan samples the
# trajectory function, so CLAUDE.mdのreal-time禁止の対象外。
_LOCAL_TARGET_SEARCH_DT = 0.05
DEFAULT_LOCAL_ATTITUDE_SPACING_M = 0.3
_SELF_PATH_SAMPLES = 400


class MincoLocalPlanner:
    """See module docstring. Arguments are those of ``ReplanningMincoV3Tracker``
    with the same names (see its docstring); ``obstacle_grid`` can be swapped
    later through the attribute of the same name."""

    def __init__(self, p0, v0, p_target, q0, target_speed, max_accel, route_waypoints,
                 planning_horizon_m, via_half_width, wrench_safety_margin,
                 attitude_resample_spacing_m, face_travel, forward_axis, local_max_vel,
                 local_piece_length_m, obstacle_grid, obstacle_clearance_soft):
        self.p_target = np.asarray(p_target, dtype=float)
        self._q0 = np.asarray(q0, dtype=float)
        self._target_speed = None if target_speed is None else float(target_speed)
        self._max_accel = None if max_accel is None else float(max_accel)
        self._planning_horizon_m = float(planning_horizon_m)
        self._via_half_width = float(via_half_width)
        self._wrench_safety_margin = float(wrench_safety_margin)
        self._attitude_resample_spacing_m = attitude_resample_spacing_m
        self._face_travel = bool(face_travel)
        self._forward_axis = np.asarray(forward_axis, dtype=float)
        self.local_max_vel = None if local_max_vel is None else float(local_max_vel)
        self._local_piece_length_m = (
            None if local_piece_length_m is None else float(local_piece_length_m))
        self.obstacle_grid = obstacle_grid
        self._obstacle_clearance_soft = float(obstacle_clearance_soft)
        self._rng = np.random.default_rng(0)

        route_waypoints = (
            np.zeros((0, 3)) if route_waypoints is None
            else np.asarray(route_waypoints, dtype=float).reshape(-1, 3)
        )
        if len(route_waypoints):
            waypoints = np.vstack([p0, route_waypoints, self.p_target])
        else:
            waypoints = np.array([p0, self.p_target])
        self.global_trajectory = self._build_global(waypoints, v0)
        route_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        # Always sized from the global trajectory's own average speed (not
        # target_speed directly), matching v2's _freetime_avg_speed -- well
        # defined immediately, never collapses near zero the way a live
        # velocity estimate transiently can right at start-of-motion, and
        # works identically whether the global build used the heuristic-time
        # or free-time path.
        self.global_avg_speed = route_length / self.global_trajectory.global_total_duration
        self._global_search_t = 0.0

    def _build_global(self, waypoints, v0):
        return MincoTrajectory(
            waypoints, self._q0, v0=v0, w0=np.zeros(3), face_travel=self._face_travel,
            forward_axis=self._forward_axis, body_frame_wrench=self._face_travel,
            via_half_width=self._via_half_width,
            attitude_resample_spacing_m=self._attitude_resample_spacing_m,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=self._target_speed, max_accel=self._max_accel,
        )

    def goal_in_obstacle(self):
        return (self.obstacle_grid is not None
                and self.obstacle_grid.inflated_occupied(list(self.p_target)))

    def move_goal_out_of_obstacle(self, p_ref, v_ref):
        """EGO-Planner v2 ``mondifyInCollisionFinalGoal``: walk the global back from its
        end to the first free point and re-plan the global from the given reference
        state to it (``planNextWaypoint``). Remaining route waypoints are dropped."""
        speed = self.local_max_vel or self.global_avg_speed
        t_step = self.obstacle_grid.resolution / max(speed, 1e-6)
        t = self.global_trajectory.global_total_duration
        while t > 0.0:
            pt = self.global_trajectory.sample(t)[0]
            if not self.obstacle_grid.inflated_occupied(list(pt)):
                self.global_trajectory = self._build_global(np.array([p_ref, pt]), v_ref)
                self.p_target = np.asarray(pt, dtype=float)
                self._global_search_t = 0.0
                return True
            t -= t_step
        return False

    def build_local(self, p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local, prev_elapsed,
                    rest_failures=0):
        """Solve a fresh free-time local segment from the head state (position
        ``p0``/``v0``/``a0``, ``q0``-relative rotvec ``rv0`` and its rate/accel)
        to the look-ahead target, ending at the global trajectory's velocity
        there (zero at the goal or inside braking distance of it, EGO-Planner
        v2 ``getLocalTarget``). K=1 without ``face_travel``; see the module
        docstring for the ``face_travel`` two-solve. ``prev_local`` (sampled at
        ``prev_elapsed`` for the head state, ``None`` for the first plan and from
        rest) seeds the multi-piece shape solve; ``rest_failures`` (consecutive
        failed replans from rest) widens the random seed. Returns
        ``(local, touch_goal)``."""
        prev_target_global_t = self._global_search_t
        target_pos, target_vel, touch_goal = self._get_local_target(p0)
        inside_braking = (
            self._max_accel is not None
            and np.linalg.norm(self.p_target - target_pos)
            < float(target_vel @ target_vel) / (2.0 * self._max_accel)
        )
        v_tail = np.zeros(3) if (touch_goal or inside_braking) else target_vel
        if self._face_travel:
            shape_seed = self._shape_seed(p0, target_pos, prev_local, prev_elapsed,
                                          prev_target_global_t, rest_failures)
            local = self.build_face_travel_local(
                p0, v0, a0, rv0, rv_rate0, rv_accel0, target_pos, v_tail, shape_seed, touch_goal)
            return local, touch_goal
        local = MincoTrajectory(
            [p0, target_pos], self._q0, v0=v0, w0=np.zeros(3),
            face_travel=False, via_half_width=self._via_half_width,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=None, max_accel=None, a0=a0, v_tail=v_tail,
        )
        return local, touch_goal

    def _shape_seed(self, p0, target_pos, prev_local, prev_elapsed, prev_target_global_t,
                    rest_failures):
        """Interior points, their directions and uniform piece times for the
        multi-piece shape solve (EGO-Planner v2 ``computeInitState`` case 2),
        or ``None`` for the single-piece solve."""
        if self._local_piece_length_m is None:
            return None
        n_pieces = max(2, math.ceil(np.linalg.norm(target_pos - p0) / self._local_piece_length_m))
        t_to_prev_end = 0.0 if prev_local is None else max(
            prev_local.global_total_duration - prev_elapsed, 0.0)
        t_to_target = t_to_prev_end + (self._global_search_t - prev_target_global_t)
        if t_to_target <= 0.0:
            return self._rest_shape_seed(p0, target_pos, n_pieces, rest_failures)
        piece_time = t_to_target / n_pieces
        points, directions = [], []
        for i in range(1, n_pieces):
            t = i * piece_time
            if t < t_to_prev_end:
                pos, vel, _a, _q = prev_local.sample(prev_elapsed + t)
            else:
                pos, vel, _a, _q = self.global_trajectory.sample(
                    prev_target_global_t + t - t_to_prev_end)
            points.append(pos)
            directions.append(vel)
        return np.array(points), np.array(directions), [piece_time] * n_pieces

    def _rest_shape_seed(self, p0, target_pos, n_pieces, rest_failures):
        """computeInitState case 1 (replanning from rest, e.g. after an emergency stop):
        through the midpoint, shifted at random across the chord once replans keep failing
        (flag_randomPolyTraj, spread growing with the failure count). The interior points
        are taken at equal arc length on the polyline through that midpoint instead of on
        EGO's min-jerk initial trajectory through it."""
        chord = target_pos - p0
        mid = (p0 + target_pos) / 2.0
        if rest_failures > 0:
            horizontal = np.cross(-chord, [0.0, 0.0, 1.0])
            if np.linalg.norm(horizontal) < 1e-9:
                horizontal = np.cross(-chord, [1.0, 0.0, 0.0])
            horizontal /= np.linalg.norm(horizontal)
            vertical = np.cross(-chord, horizontal)
            vertical /= np.linalg.norm(vertical)
            spread = -0.978 / (rest_failures + 0.989) + 0.989
            length = np.linalg.norm(chord)
            mid = (mid + (self._rng.random() - 0.5) * length * horizontal * 0.8 * spread
                   + (self._rng.random() - 0.5) * length * vertical * 0.4 * spread)
        legs = np.linalg.norm(mid - p0), np.linalg.norm(target_pos - mid)
        total = legs[0] + legs[1]
        points = []
        for i in range(1, n_pieces):
            s = total * i / n_pieces
            points.append(p0 + (mid - p0) * s / legs[0] if s <= legs[0]
                          else mid + (target_pos - mid) * (s - legs[0]) / legs[1])
        points = np.array(points)
        directions = np.diff(np.vstack([p0, points, target_pos]), axis=0)[1:]
        speed = self.local_max_vel or self.global_avg_speed
        piece_time = total / max(speed, 1e-6) / n_pieces
        return points, directions, [piece_time] * n_pieces

    def build_face_travel_local(self, p0, v0, a0, rv0, rv_rate0, rv_accel0,
                                target_pos, v_tail, shape_seed=None, touch_goal=False):
        def solve(points, directions, warm_start_segment_times, via_half_width=0.0,
                  obstacle_grid=None):
            rotvecs = MincoTrajectory._rotvecs_from_directions(
                directions, self._q0, self._forward_axis, rv_head=rv0)
            return MincoTrajectory.from_rotvec_waypoints(
                points, rotvecs, self._q0, v0, rv_rate0, a0, rv_accel0, v_tail=v_tail,
                via_half_width=via_half_width,
                wrench_safety_margin=self._wrench_safety_margin,
                max_vel=self.local_max_vel,
                warm_start_segment_times=warm_start_segment_times,
                body_frame_wrench=True, obstacle_grid=obstacle_grid,
                obstacle_touch_goal=touch_goal,
                obstacle_clearance_soft=self._obstacle_clearance_soft)

        obstacle_grid = self.obstacle_grid
        if shape_seed is None:
            shape = solve(np.array([p0, target_pos]), np.array([np.zeros(3), target_pos - p0]), None,
                          obstacle_grid=obstacle_grid)
        else:
            inner_points, inner_directions, piece_times = shape_seed
            shape = solve(np.vstack([p0, inner_points, target_pos]),
                          np.vstack([np.zeros(3), inner_directions, target_pos - p0]),
                          piece_times, via_half_width=math.inf, obstacle_grid=obstacle_grid)
        # Sample resolution only (the solved path is analytic), not a clock/timing decision.
        ts = np.linspace(0.0, shape.global_total_duration, _SELF_PATH_SAMPLES)
        path = np.array([shape.sample(t)[0] for t in ts])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        spacing = self._attitude_resample_spacing_m or DEFAULT_LOCAL_ATTITUDE_SPACING_M
        cuts = np.searchsorted(arc, np.arange(spacing, arc[-1] - spacing / 2.0, spacing))
        point_times = np.concatenate([[0.0], ts[cuts], [shape.global_total_duration]])
        samples = [shape.sample(t) for t in point_times]
        local = solve(np.array([smp[0] for smp in samples]), np.array([smp[1] for smp in samples]),
                      np.diff(point_times))
        local.solve_wall_seconds += shape.solve_wall_seconds
        return local

    def _get_local_target(self, p_from):
        """Walk the global trajectory forward from the last search cursor
        until its sampled position is ``planning_horizon_m`` away from
        ``p_from``, or the global trajectory's own end is reached (in which
        case the target is ``p_target`` directly and ``touch_goal=True``).
        Returns ``(pos, vel, touch_goal)``. Ported from
        ``prototype_ego_v2_style_local_replan.py``'s ``get_local_target``."""
        total_dur = self.global_trajectory.global_total_duration
        t_step = max(
            self._planning_horizon_m / 20.0 / max(self.global_avg_speed, 1e-6),
            _LOCAL_TARGET_SEARCH_DT,
        )
        t = self._global_search_t
        while t < total_dur:
            pos_t, vel_t, _a, _q = self.global_trajectory.sample(t)
            if np.linalg.norm(pos_t - p_from) >= self._planning_horizon_m:
                self._global_search_t = t
                return pos_t, vel_t, False
            t += t_step
        self._global_search_t = total_dur
        return self.p_target.copy(), np.zeros(3), True
