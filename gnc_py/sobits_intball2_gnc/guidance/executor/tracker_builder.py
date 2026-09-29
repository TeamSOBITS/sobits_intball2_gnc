#!/usr/bin/env python3
"""Builds the tracker for one move_to goal's ``trajectory_tracking_mode`` (ROS-agnostic).

Split out of ``GuidanceExecutor.execute()`` (``guidance/executor/guidance_executor.py``),
which calls :meth:`TrackerBuilder.build` once per goal. ``replan_minco``
falls back to ``static_toppra`` on failure; any other failure raises
:class:`TrajectoryBuildError` and the goal is aborted.
"""
import time

import numpy as np

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    DEFAULT_LOCAL_REPLAN_PERIOD_S,
    DEFAULT_PLANNING_HORIZON_M,
    ReplanMincoTracker,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)

TRAJECTORY_TRACKING_MODES = frozenset(
    {"static_toppra", "static_minco", "replan_minco"}
)


class TrajectoryBuildError(RuntimeError):
    pass


class TrackerBuilder:
    """Args mirror ``GuidanceExecutor``'s of the same names; ``tf_fresh_fn`` is its
    TF-liveness check, ``stop_profile_fn(p, v, q, omega)`` the emergency-stop
    profile for obstacle avoidance (``None`` = no emergency stop). While an
    obstacle-avoiding ``replan_minco`` goal runs, grids rebuilt by
    ``obstacle_map`` are handed to its tracker."""

    def __init__(self, tf_client, tf_fresh_fn, logger, target_speed, max_accel,
                 max_angular_rate, wrench_envelope, mass, inertia, obstacle_map=None,
                 stop_profile_fn=None):
        self._tf = tf_client
        self._tf_fresh_fn = tf_fresh_fn
        self._log = logger
        self._target_speed = target_speed
        self._max_accel = max_accel
        self._max_angular_rate = max_angular_rate
        self._wrench_envelope = wrench_envelope
        self._mass = mass
        self._inertia = inertia
        self._obstacle_map = obstacle_map
        self._stop_profile_fn = stop_profile_fn
        self._obstacle_tracker = None
        if obstacle_map is not None:
            obstacle_map.add_listener(self._on_obstacle_grid)

    def build(self, p0, q0, p_target, via_waypoints, trajectory_tracking_mode, forward_axis,
              face_travel, minco_via_half_width=0.3, minco_attitude_resample_spacing_m=None,
              minco_wrench_safety_margin=1.0, minco_freetime=False,
              minco_local_replan_period=DEFAULT_LOCAL_REPLAN_PERIOD_S,
              minco_planning_horizon_m=DEFAULT_PLANNING_HORIZON_M,
              minco_replan_face_travel=False, minco_local_max_vel=None,
              minco_async_replan=False, minco_obstacle_avoidance=False,
              minco_local_piece_length_m=None, minco_obstacle_clearance_soft=0.2):
        """Returns ``(tracker, traj)``: ``traj`` is the trajectory to preview (the
        tracked one, or the global one for ``replan_minco``).
        ``forward_axis`` must already be resolved (never ``None``).
        Raises :class:`TrajectoryBuildError` if no trajectory can be built."""
        waypoints = [p0, *via_waypoints, p_target]

        mode = trajectory_tracking_mode
        if mode not in TRAJECTORY_TRACKING_MODES:
            raise TrajectoryBuildError(
                "unknown trajectory_tracking_mode=%r" % trajectory_tracking_mode)
        if mode == "replan_minco" and self._max_accel is None and not minco_freetime:
            # Only the (one-time) global build's heuristic-time path needs
            # max_accel; minco_freetime doesn't.
            self._log.warn(
                "[TrackerBuilder] trajectory_tracking_mode='replan_minco' "
                "requires max_accel to be configured (unless minco_freetime=True) "
                "-- falling back to 'static_toppra'"
            )
            mode = "static_toppra"

        traj = None
        replan_tracker = None
        self._obstacle_tracker = None
        obstacle_kwargs = {}
        if mode == "replan_minco" and minco_obstacle_avoidance:
            obstacle_kwargs = self._obstacle_tracker_kwargs(
                minco_local_piece_length_m, minco_obstacle_clearance_soft)
        if mode == "replan_minco":
            # Builds its own trajectory internally; `traj` is set to its
            # global MincoTrajectory only for the speed path preview.
            #
            # Must run before the `mode == "static_toppra"` branch below, which
            # also serves its downgrade on infeasibility.
            try:
                replan_tracker = ReplanMincoTracker(
                    p0, p_target, pose_fn=self._tf.get_pose,
                    tf_fresh_fn=self._tf_fresh_fn, q0=q0,
                    target_speed=(None if minco_freetime else self._target_speed),
                    max_accel=(None if minco_freetime else self._max_accel),
                    route_waypoints=via_waypoints,
                    via_half_width=minco_via_half_width,
                    wrench_safety_margin=minco_wrench_safety_margin,
                    attitude_resample_spacing_m=minco_attitude_resample_spacing_m,
                    local_replan_period=minco_local_replan_period,
                    planning_horizon_m=minco_planning_horizon_m,
                    face_travel=face_travel and minco_replan_face_travel,
                    forward_axis=forward_axis,
                    local_max_vel=minco_local_max_vel,
                    async_replan=minco_async_replan,
                    **obstacle_kwargs,
                )
                if obstacle_kwargs:
                    self._obstacle_tracker = replan_tracker
                    if self._obstacle_map.grid is not obstacle_kwargs["obstacle_grid"]:
                        replan_tracker.set_obstacle_grid(self._obstacle_map.grid)
                traj = replan_tracker.trajectory
                self._log.info(
                    "[TrackerBuilder] initial replan_minco global "
                    "solve took %.2fs (%d waypoints)"
                    % (traj.solve_wall_seconds, traj.num_waypoints)
                )
            except (MincoInfeasibleError, ValueError) as exc:
                self._log.warn(
                    "[TrackerBuilder] replan_minco tracker "
                    "construction failed (%s) -- falling back to 'static_toppra'" % exc
                )
                replan_tracker = None
                traj = None
                mode = "static_toppra"

        if mode == "static_toppra":
            traj = self._build_toppra(waypoints, q0, forward_axis, face_travel)

        if mode == "static_minco":
            # 再計画は一切しない: MINCOをgoal受付時に一度だけ解いて、
            # StaticTrajectoryTrackerでそのまま完走させる。
            # plan_mincoのブロッキング解決レイテンシ（実測3〜5秒）はgoal受付が
            # 一度遅れるだけで済む（docs/archive/achieved/
            # 2026-08-30_minco_replanning_blocking_latency_incident.md）。
            try:
                traj = MincoTrajectory(
                    waypoints, q0, v0=np.zeros(3), w0=np.zeros(3),
                    forward_axis=forward_axis,
                    face_travel=face_travel,
                    via_half_width=minco_via_half_width,
                    attitude_resample_spacing_m=minco_attitude_resample_spacing_m,
                    wrench_safety_margin=minco_wrench_safety_margin,
                )
            except MincoInfeasibleError as exc:
                raise TrajectoryBuildError(
                    "static MINCO trajectory infeasible (%s)" % exc) from exc
            self._log.info(
                "[TrackerBuilder] static MINCO solve took %.2fs "
                "(%d waypoints, via_half_width=%.2f, "
                "attitude_resample_spacing_m=%s, wrench_safety_margin=%.2f)"
                % (traj.solve_wall_seconds, traj.num_waypoints,
                   minco_via_half_width, minco_attitude_resample_spacing_m,
                   minco_wrench_safety_margin)
            )

        if mode == "replan_minco":
            return replan_tracker, traj
        return StaticTrajectoryTracker(traj), traj

    def _build_toppra(self, waypoints, q0, forward_axis, face_travel):
        missing = [name for name, value in (
            ("wrench_envelope", self._wrench_envelope), ("mass", self._mass),
            ("inertia", self._inertia), ("max_angular_rate", self._max_angular_rate),
        ) if value is None]
        if missing:
            raise TrajectoryBuildError(
                "TOPP-RA needs %s configured" % ", ".join(missing))
        try:
            build_t0 = time.perf_counter()
            traj = ToppraTrajectory(
                waypoints, q0,
                max_vel=self._target_speed,
                mass=self._mass,
                inertia=self._inertia,
                wrench_envelope=self._wrench_envelope,
                max_angular_rate=self._max_angular_rate,
                forward_axis=forward_axis,
                face_travel=face_travel,
            )
            build_wall_seconds = time.perf_counter() - build_t0
        except TrajectoryInfeasibleError as exc:
            raise TrajectoryBuildError(
                "TOPP-RA time-parameterization infeasible (%s)" % exc) from exc
        self._log.info(
            "[TrackerBuilder] TOPP-RA trajectory used (%d waypoints, "
            "duration=%.2fs, build_time=%.3fs)"
            % (len(waypoints), traj.global_total_duration, build_wall_seconds)
        )
        return traj

    def _obstacle_tracker_kwargs(self, local_piece_length_m, obstacle_clearance_soft):
        if self._obstacle_map is None:
            self._log.warn("[TrackerBuilder] minco_obstacle_avoidance needs an "
                           "obstacle map -- planning without obstacles")
            return {}
        kwargs = {
            "obstacle_grid": self._obstacle_map.grid,
            "local_piece_length_m": local_piece_length_m,
            "obstacle_clearance_soft": obstacle_clearance_soft,
        }
        if self._stop_profile_fn is not None:
            kwargs["stop_profile_fn"] = self._stop_profile_fn
        else:
            self._log.warn("[TrackerBuilder] no allocator/mass/inertia -- obstacle "
                           "avoidance replans without an emergency stop")
        return kwargs

    def _on_obstacle_grid(self, grid):
        tracker = self._obstacle_tracker
        if tracker is not None:
            tracker.set_obstacle_grid(grid)
