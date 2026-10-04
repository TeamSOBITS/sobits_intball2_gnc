#!/usr/bin/env python3
"""MINCO global/local planner behind ``ReplanMincoTracker`` (ROS-agnostic, pure).

EGO-Planner v2 splits replanning into ``ego_replan_fsm`` (when to replan, collision
checks, emergency stop) and ``planner_manager`` (how to build the trajectories).
This is the latter; ``trajectory_tracking/replan_minco_tracker.py`` is the
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
  (zero at the goal or once inside braking distance of it). Starting from the measured state with zero acceleration and
  ending at rest instead -- the original port -- made the vehicle crawl, overshoot
  the goal, and absorb disturbance-induced velocity into every plan as lateral
  drift (docs/archive/achieved/2026-09-23_replanning_minco_v3_sim_verification_result_and_root_cause.md).
  (:meth:`_get_local_target`.) Without ``face_travel`` it is a single segment (K=1),
  the offline-verified baseline; K>1 was offline-verified to help (~29.7% faster
  convergence, offline verification fact 51) but combining it with warm start
  showed no measurable benefit (fact 54).

Attitude: with ``face_travel=False`` (default) every ``MincoTrajectory`` built
here is fixed at ``q0`` and ``w0=0``. With ``face_travel=True`` the global faces
travel, and each local is solved twice: once for its own path (optionally
multi-piece (``local_piece_length_m``, seeded by :meth:`_shape_seed`), with
``obstacle_grid`` rebound), then re-solved through points on that path with
attitude waypoints facing its own tangent (not the global attitude, so it stays
valid once the local deviates, e.g. for obstacles), carrying the head rotvec
rate/accel. It also caps local speed (the split local otherwise accelerates to its
braking limit every period) and checks the wrench envelope in the body frame
(``docs/archive/achieved/2026-09-23_replanning_minco_replan_face_travel_handoff.md``).
``attitude_resample_spacing_m`` is still forwarded to the *global* build for
spatial densify (the global route's own smoothness, independent of whether
attitude is tracked) -- this only works because of the ``face_travel``/densify
decoupling fix in ``minco_trajectory.py`` (2026-09-23, ``docs/
2026-09-20_ego_v2_style_replan_migration_plan.md`` "やるべき内容1"); before that
fix, ``face_travel=False`` silently disabled densify.
"""
import math

import numpy as np

from sobits_intball2_gnc.guidance.global_planner.astar_planner import (
    AStarPlanner,
    AStarPlanningError,
)
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import shortcut_path
from sobits_intball2_gnc.guidance.global_planner.reference_route import densify
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)
from sobits_intball2_gnc.guidance.trajectory.reference_polynomial import MinJerkReference

# Resolution for the internal forward-scan of the (already-solved, analytic)
# global trajectory in _get_local_target -- not a control-loop timing
# decision (no clock/sleep involved), just how finely that scan samples the
# trajectory function, so CLAUDE.mdのreal-time禁止の対象外。
_LOCAL_TARGET_SEARCH_DT = 0.05
DEFAULT_LOCAL_ATTITUDE_SPACING_M = 0.3
_SELF_PATH_SAMPLES = 400
# A* seed search box: both ends plus this margin; the static map bounds it further.
_ASTAR_SEED_MARGIN_M = 1.0
# A local target inside the inflated grid moves forward along the global trajectory to a free
# point at least this far past the last occupied sample (None: no move). The rebound check
# fails outright on an occupied target, and with depth the first free sample past a seen face
# can be inside the unseen obstacle (docs/jaxa_baseline_ompl_reproduction.md 9.4).
_LOCAL_TARGET_OBSTACLE_MARGIN_M = 1.0
# How an occupied local target is replaced: "forward" along the global trajectory (above) or
# "astar": ``planning_horizon_m`` along an A*6 path from the vehicle to the goal.
_LOCAL_TARGET_IN_OBSTACLE = "forward"
# Spacing of the points the reference global spline is pinned to. At the 0.5 m
# of SCAN-Planner's reference mode the spline still rounds a corner into the
# inflated grid; 0.25 m keeps every layout clear of the vehicle radius
# (docs/minco_astar_reference_global.md 1).
_REFERENCE_SPACING_M = 0.25


class MincoLocalPlanner:
    """See module docstring. Arguments are those of ``ReplanMincoTracker``
    with the same names (see its docstring); ``obstacle_grid`` can be swapped
    later through the attribute of the same name."""

    def __init__(self, p0, v0, p_target, q0, target_speed, max_accel, route_waypoints,
                 planning_horizon_m, via_half_width, wrench_safety_margin,
                 attitude_resample_spacing_m, face_travel, forward_axis, local_max_vel,
                 local_piece_length_m, obstacle_grid, obstacle_clearance_soft, corridor_planes=None,
                 reference_route=None, obstacle_clearance=None):
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
        self._obstacle_clearance = None if obstacle_clearance is None else float(obstacle_clearance)
        if self._obstacle_clearance is not None:
            if not self._obstacle_clearance > 0.0:
                raise ValueError("obstacle_clearance must be positive")
            if self._obstacle_clearance >= self._obstacle_clearance_soft:
                # penalties.cpp's `useSoft = clearanceSoft > clearance` drops the
                # soft term entirely once the hard one catches up with it.
                raise ValueError(
                    "obstacle_clearance (%.3f) must stay below obstacle_clearance_soft "
                    "(%.3f), which would otherwise be ignored"
                    % (self._obstacle_clearance, self._obstacle_clearance_soft))
        self._corridor_planes = corridor_planes
        self._rng = np.random.default_rng(0)
        self._replan_failures = 0

        route_waypoints = (
            np.zeros((0, 3)) if route_waypoints is None
            else np.asarray(route_waypoints, dtype=float).reshape(-1, 3)
        )
        self._reference_route = reference_route is not None
        if self._reference_route:
            if len(route_waypoints):
                raise ValueError("reference_route and route_waypoints are exclusive")
            route = np.asarray(reference_route, dtype=float).reshape(-1, 3)
            if len(route) < 2:
                raise ValueError("reference_route needs at least a start and a goal")
            # The ends come from p0/p_target: the route was planned before the
            # final TF read, so its own ends can be centimetres away by now.
            route_waypoints = route[1:-1]
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
        if self._reference_route:
            # Position-only, so the global carries no attitude or wrench limits
            # and its time does not blow up as the waypoint count grows; the
            # local solves own both (docs/minco_astar_reference_global.md 1).
            speed = self.local_max_vel or self._target_speed
            return MinJerkReference(densify(waypoints, _REFERENCE_SPACING_M), speed, self._q0)
        return MincoTrajectory(
            waypoints, self._q0, v0=v0, w0=np.zeros(3), face_travel=self._face_travel,
            forward_axis=self._forward_axis, body_frame_wrench=self._face_travel,
            via_half_width=self._via_half_width,
            attitude_resample_spacing_m=self._attitude_resample_spacing_m,
            wrench_safety_margin=self._wrench_safety_margin,
            target_speed=self._target_speed, max_accel=self._max_accel,
            corridor_planes=self._corridor_planes,
        )

    def goal_in_obstacle(self):
        return (self.obstacle_grid is not None
                and self.obstacle_grid.inflated_occupied(list(self.p_target)))

    def move_goal_out_of_obstacle(self, p_ref, v_ref):
        """If the goal lies in the inflated grid, walk the global trajectory back from its end
        to the first free point and rebuild the global from ``p_ref``/``v_ref`` straight to it
        (via waypoints dropped). Returns whether the goal was moved; ``False`` also when the
        whole global is occupied, which is left to the emergency stop."""
        if not self.goal_in_obstacle():
            return False
        grid = self.obstacle_grid
        duration = self.global_trajectory.global_total_duration
        walk_speed = self.local_max_vel if self.local_max_vel else self.global_avg_speed
        dt = grid.resolution / walk_speed if walk_speed and walk_speed > 0.0 else _LOCAL_TARGET_SEARCH_DT
        free_point = None
        for t in np.append(np.arange(duration, 0.0, -dt), 0.0):
            p = np.asarray(self.global_trajectory.sample(t)[0], dtype=float)
            if not grid.inflated_occupied(list(p)):
                free_point = p
                break
        if free_point is None:
            return False
        p_ref = np.asarray(p_ref, dtype=float)
        self.p_target = free_point
        self.global_trajectory = self._build_global(np.array([p_ref, free_point]), v_ref)
        self.global_avg_speed = (float(np.linalg.norm(free_point - p_ref))
                                 / self.global_trajectory.global_total_duration)
        self._global_search_t = 0.0
        return True

    def build_local(self, p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local, prev_elapsed,
                    rest_failures=0):
        """:meth:`_build_local_seeded` with EGO-Planner v2 ``planFromLocalTraj``'s fallback:
        with obstacles, a failed warm-started solve is retried from a straight seed, then from
        a random one widening with consecutive failures, so a local minimum is not retried with
        the same seed forever. Before the random seed an A* seed is tried (ours, not in EGO v2):
        EGO's random midpoint scales with the chord, so it cannot reach a goal right behind an
        obstacle (docs/jaxa_baseline_ompl_reproduction.md 9.3). Raises
        :class:`MincoInfeasibleError` when every tier fails."""
        tiers = ("warm",)
        if (self.obstacle_grid is not None and self._face_travel
                and self._local_piece_length_m is not None and self._corridor_planes is None):
            tiers = ("warm", "straight", "astar", "random")
        cursor = self._global_search_t
        for k, seed_mode in enumerate(tiers):
            self._global_search_t = cursor  # every tier aims at the same local target
            try:
                result = self._build_local_seeded(
                    p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local, prev_elapsed,
                    max(rest_failures, self._replan_failures) + 1 if seed_mode == "random"
                    else rest_failures, seed_mode)
            except MincoInfeasibleError:
                if k == len(tiers) - 1:
                    self._replan_failures += 1
                    raise
                continue
            self._replan_failures = 0
            return result

    def _build_local_seeded(self, p0, v0, a0, rv0, rv_rate0, rv_accel0, prev_local, prev_elapsed,
                            rest_failures, seed_mode):
        """Solve a fresh free-time local segment from the head state (position
        ``p0``/``v0``/``a0``, ``q0``-relative rotvec ``rv0`` and its rate/accel)
        to the look-ahead target, ending at the global trajectory's velocity
        there (zero at the goal or inside braking distance of it). K=1 without ``face_travel``; see the module
        docstring for the ``face_travel`` two-solve. ``prev_local`` (sampled at
        ``prev_elapsed`` for the head state, ``None`` for the first plan and from
        rest) seeds the multi-piece shape solve; ``rest_failures`` (consecutive
        failed replans from rest) widens the random seed. ``seed_mode`` (``face_travel`` with
        ``local_piece_length_m`` only): ``"warm"`` as above, ``"straight"`` seeds from the
        chord to the target, ``"random"`` from the chord midpoint pushed sideways by
        ``rest_failures`` (EGO-Planner v2 ``planFromLocalTraj``'s three tiers). Returns
        ``(local, touch_goal)``."""
        prev_target_global_t = self._global_search_t
        target_pos, target_vel, touch_goal = self._get_local_target(p0)
        v_tail = np.zeros(3) if touch_goal else target_vel
        if self._face_travel:
            if seed_mode == "astar":
                shape_seed = self._astar_shape_seed(p0, target_pos)
            elif seed_mode != "warm" and self._local_piece_length_m is not None:
                n_pieces = max(2, math.ceil(np.linalg.norm(np.asarray(target_pos) - np.asarray(p0))
                                            / self._local_piece_length_m))
                shape_seed = self._rest_shape_seed(
                    p0, target_pos, n_pieces, rest_failures if seed_mode == "random" else 0)
            elif self._corridor_planes is not None and prev_local is None:
                shape_seed = self._global_shape_seed(p0, target_pos)
            else:
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

    def _global_shape_seed(self, p0, target_pos):
        """Seed the first long local solve from the already-safe global curve."""
        duration = max(self._global_search_t, 1e-3)
        n_pieces = max(2, math.ceil(self._planning_horizon_m / self._local_piece_length_m))
        samples = [self.global_trajectory.sample(duration * k / n_pieces)
                   for k in range(1, n_pieces)]
        return (np.asarray([sample[0] for sample in samples]),
                np.asarray([sample[1] for sample in samples]),
                np.full(n_pieces, duration / n_pieces))

    def _shape_seed(self, p0, target_pos, prev_local, prev_elapsed, prev_target_global_t,
                    rest_failures):
        """Initial ``(inner_points, inner_directions, piece_times)`` of the multi-piece shape
        solve, or ``None`` without ``local_piece_length_m``: the rest of ``prev_local`` (if any)
        joined with the global trajectory up to the new target, cut into equal times."""
        if self._local_piece_length_m is None:
            return None
        p0 = np.asarray(p0, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        n_pieces = max(2, math.ceil(np.linalg.norm(target_pos - p0) / self._local_piece_length_m))

        prev_span = 0.0 if prev_local is None else max(
            prev_local.global_total_duration - prev_elapsed, 0.0)
        global_span = max(self._global_search_t - prev_target_global_t, 0.0)
        total_span = prev_span + global_span
        if total_span <= 0.0:
            return self._rest_shape_seed(p0, target_pos, n_pieces, rest_failures)
        inner_points, inner_directions = [], []
        for k in range(1, n_pieces):
            t = total_span * k / n_pieces
            if t <= prev_span:
                p, v, _a, _q = prev_local.sample(prev_elapsed + t)
            else:
                p, v, _a, _q = self.global_trajectory.sample(prev_target_global_t + t - prev_span)
            inner_points.append(p)
            inner_directions.append(v)
        return (np.asarray(inner_points, dtype=float), np.asarray(inner_directions, dtype=float),
                np.full(n_pieces, total_span / n_pieces))

    def _astar_shape_seed(self, p0, target_pos):
        """Seed along the shortcut A*6 path to the target, in a box around both ends."""
        p0 = np.asarray(p0, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        grid = self.obstacle_grid
        if hasattr(grid, "snapshot"):
            grid = grid.snapshot()
        bounds = (np.minimum(p0, target_pos) - _ASTAR_SEED_MARGIN_M,
                  np.maximum(p0, target_pos) + _ASTAR_SEED_MARGIN_M)
        try:
            path = np.asarray(shortcut_path(
                AStarPlanner(grid.resolution, grid=grid, search_bounds=bounds,
                             connectivity=6).plan(p0, target_pos), grid, bounds), dtype=float)
        except (AStarPlanningError, ValueError) as exc:
            raise MincoInfeasibleError("A* seed: %s" % exc) from exc
        # Two pieces per A* leg so the seed keeps the detour's corners; piece length alone
        # (1.5 m) can leave a U-turn around a box with a single inner point.
        n_pieces = max(2 * (len(path) - 1),
                       math.ceil(_polyline_length(path) / self._local_piece_length_m))
        cuts = _equal_arc_cuts(path, n_pieces)
        if cuts is None:
            raise MincoInfeasibleError("A* seed: degenerate path")
        length, inner_points, segments = cuts
        inner_directions = np.array([_unit(path[seg + 1] - path[seg]) for seg in segments])
        return (inner_points, inner_directions,
                np.full(n_pieces, length / self.global_avg_speed / n_pieces))

    def _rest_shape_seed(self, p0, target_pos, n_pieces, rest_failures):
        """Seed for replanning from rest (e.g. after an emergency stop): a two-leg polyline
        through the chord midpoint, pushed sideways by a random offset that widens with the
        number of consecutive failures from rest."""
        p0 = np.asarray(p0, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        chord = target_pos - p0
        chord_length = float(np.linalg.norm(chord))
        midpoint = 0.5 * (p0 + target_pos)
        if rest_failures > 0:
            horizontal = np.cross(chord, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(horizontal) < 1e-6:
                horizontal = np.array([1.0, 0.0, 0.0])
            horizontal = _unit(horizontal)
            vertical = _unit(np.cross(chord, horizontal))
            spread = min(0.99, 1.0 - 0.5 ** rest_failures)
            xi_h, xi_v = self._rng.uniform(-1.0, 1.0, size=2)
            midpoint = midpoint + chord_length * spread * (0.8 * xi_h * horizontal + 0.4 * xi_v * vertical)
        polyline = np.array([p0, midpoint, target_pos])
        leg_directions = [_unit(polyline[1] - polyline[0]), _unit(polyline[2] - polyline[1])]
        cuts = _equal_arc_cuts(polyline, n_pieces)
        if cuts is None:
            inner = np.array([p0 + chord * k / n_pieces for k in range(1, n_pieces)])
            return inner, np.tile(chord, (n_pieces - 1, 1)), np.full(n_pieces, 1e-3)
        length, inner_points, segments = cuts
        inner_directions = []
        for point, seg in zip(inner_points, segments):
            if seg == 1 and np.linalg.norm(point - polyline[1]) < 1e-9:
                inner_directions.append(_unit(leg_directions[0] + leg_directions[1]))
            else:
                inner_directions.append(leg_directions[min(seg, 1)])
        return (inner_points, np.array(inner_directions),
                np.full(n_pieces, length / self.global_avg_speed / n_pieces))

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
                obstacle_clearance_soft=self._obstacle_clearance_soft,
                obstacle_clearance=self._obstacle_clearance)

        obstacle_grid = self.obstacle_grid
        if obstacle_grid is not None and hasattr(obstacle_grid, "snapshot"):
            # A solve can take seconds; reading a copy keeps depth integration from waiting on it.
            obstacle_grid = obstacle_grid.snapshot()
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
        """Walk the global trajectory forward from the last search cursor to the first point
        ``planning_horizon_m`` (straight-line) from ``p_from``; returns ``(pos, vel,
        touch_goal)`` with the global velocity there, zeroed once that target is within braking
        distance of the goal (with ``max_accel`` only). A target in the inflated grid moves
        forward (:meth:`_free_target_from`). Past the end: the goal at rest."""
        p_from = np.asarray(p_from, dtype=float)
        duration = self.global_trajectory.global_total_duration
        t = self._global_search_t
        while t < duration:
            p, v, _a, _q = self.global_trajectory.sample(t)
            if np.linalg.norm(np.asarray(p) - p_from) >= self._planning_horizon_m:
                if _LOCAL_TARGET_IN_OBSTACLE == "astar":
                    found = self._astar_target(p_from, t)
                    if found is not None:
                        self._global_search_t = t
                        return found
                moved = self._free_target_from(t)
                if moved is None:
                    break
                t, p, v = moved
                self._global_search_t = t
                vel = np.asarray(v, dtype=float)
                if self._max_accel is not None and self._max_accel > 0.0:
                    braking_distance = float(vel @ vel) / (2.0 * self._max_accel)
                    if np.linalg.norm(self.p_target - np.asarray(p, dtype=float)) < braking_distance:
                        vel = np.zeros(3)
                return np.asarray(p, dtype=float), vel, False
            t += _LOCAL_TARGET_SEARCH_DT
        self._global_search_t = duration
        return self.p_target.copy(), np.zeros(3), True

    def _astar_target(self, p_from, t):
        """``(pos, vel, touch_goal)`` ``planning_horizon_m`` along an A*6 path from ``p_from``
        to the goal when the global trajectory at ``t`` is in the inflated grid, else ``None``
        (also when A* fails). The velocity is the global speed at ``t`` (capped at
        ``local_max_vel``) along the path."""
        grid = self.obstacle_grid
        p, v, _a, _q = self.global_trajectory.sample(t)
        if grid is None or not grid.inflated_occupied(list(np.asarray(p, dtype=float))):
            return None
        if hasattr(grid, "snapshot"):
            grid = grid.snapshot()
        goal = np.asarray(self.p_target, dtype=float)
        bounds = (np.minimum(p_from, goal) - _ASTAR_SEED_MARGIN_M,
                  np.maximum(p_from, goal) + _ASTAR_SEED_MARGIN_M)
        try:
            path = np.asarray(shortcut_path(
                AStarPlanner(grid.resolution, grid=grid, search_bounds=bounds,
                             connectivity=6).plan(p_from, goal), grid, bounds), dtype=float)
        except (AStarPlanningError, ValueError):
            return None
        remaining = self._planning_horizon_m
        for a, b in zip(path[:-1], path[1:]):
            length = float(np.linalg.norm(b - a))
            if length >= remaining:
                speed = float(np.linalg.norm(v))
                if self.local_max_vel:
                    # The global speed can exceed the local cap; an end velocity above it is
                    # infeasible for the local solve.
                    speed = min(speed, float(self.local_max_vel))
                return a + (b - a) * (remaining / length), _unit(b - a) * speed, False
            remaining -= length
        return goal.copy(), np.zeros(3), True

    def _free_target_from(self, t):
        """``(t, pos, vel)`` of the global trajectory at ``t``, or past it at the first free
        sample ``_LOCAL_TARGET_OBSTACLE_MARGIN_M`` beyond the last occupied one when ``t`` is in
        the inflated grid; ``None`` when no such point comes before the goal."""
        grid = self.obstacle_grid
        p, v, _a, _q = self.global_trajectory.sample(t)
        if (grid is None or _LOCAL_TARGET_OBSTACLE_MARGIN_M is None
                or not grid.inflated_occupied(list(np.asarray(p, dtype=float)))):
            return t, p, v
        duration = self.global_trajectory.global_total_duration
        last_occupied = np.asarray(p, dtype=float)
        while t < duration:
            t += _LOCAL_TARGET_SEARCH_DT
            p, v, _a, _q = self.global_trajectory.sample(min(t, duration))
            p = np.asarray(p, dtype=float)
            if grid.inflated_occupied(list(p)):
                last_occupied = p
            elif np.linalg.norm(p - last_occupied) >= _LOCAL_TARGET_OBSTACLE_MARGIN_M:
                return t, p, v
        return None


def _polyline_length(points):
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _unit(v):
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else np.zeros(3)


def _equal_arc_cuts(points, n_pieces):
    """``(length, inner points, segment index of each)`` cutting the polyline into
    ``n_pieces`` equal arc lengths, or ``None`` for a degenerate polyline."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    length = float(arc[-1])
    if length < 1e-6:
        return None
    inner, segments = [], []
    for k in range(1, n_pieces):
        s = length * k / n_pieces
        seg = min(int(np.searchsorted(arc, s, side="right")) - 1, len(points) - 2)
        seg_length = arc[seg + 1] - arc[seg]
        frac = 0.0 if seg_length <= 0.0 else (s - arc[seg]) / seg_length
        inner.append(points[seg] + frac * (points[seg + 1] - points[seg]))
        segments.append(seg if frac < 1.0 - 1e-12 else seg + 1)
    return length, np.array(inner), segments
