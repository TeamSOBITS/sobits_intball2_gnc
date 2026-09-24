#!/usr/bin/env python3
"""Builds the tracker for one move_to goal's ``trajectory_tracking_mode`` (ROS-agnostic).

Split out of ``GuidanceExecutor.execute()`` (``guidance/utils/guidance_executor.py``),
which calls :meth:`TrackerBuilder.build` once per goal. The fallback chain is
unchanged: ``replanning_minco_v3`` -> ``static`` on failure; ``static`` tries the
force/torque-aware TOPP-RA path first; ``static_minco`` solves MINCO once; anything
left without a trajectory ends on the legacy Hermite path.
"""
import time

import numpy as np

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
)
from sobits_intball2_gnc.guidance.trajectory.trajectory import Trajectory
from sobits_intball2_gnc.guidance.trajectory_generation import (
    hermite_spline_trajectory_generator as _hermite,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import (
    DEFAULT_LOCAL_REPLAN_PERIOD_S,
    DEFAULT_PLANNING_HORIZON_M,
    ReplanningMincoV3Tracker,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)

TRAJECTORY_TRACKING_MODES = frozenset(
    {"static", "static_minco", "replanning_minco_v3"}
)


class TrackerBuilder:
    """Args mirror ``GuidanceExecutor``'s of the same names; ``tf_fresh_fn`` is its
    TF-liveness check, ``stop_profile_fn(p, v, q, omega)`` the emergency-stop
    profile for obstacle avoidance (``None`` = no emergency stop). While an
    obstacle-avoiding ``replanning_minco_v3`` goal runs, grids rebuilt by
    ``obstacle_map`` are handed to its tracker."""

    def __init__(self, tf_client, tf_fresh_fn, logger, target_speed, max_accel,
                 attitude_speed_threshold, max_angular_rate, wrench_envelope, mass,
                 inertia, obstacle_map=None, stop_profile_fn=None):
        self._tf = tf_client
        self._tf_fresh_fn = tf_fresh_fn
        self._log = logger
        self._target_speed = target_speed
        self._max_accel = max_accel
        self._attitude_speed_threshold = attitude_speed_threshold
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
              minco_v3_face_travel=False, minco_local_max_vel=None,
              minco_v3_async_replan=False, minco_obstacle_avoidance=False,
              minco_local_piece_length_m=None, minco_obstacle_clearance_soft=0.2):
        """Returns ``(tracker, traj)``: ``traj`` is the trajectory to preview (the
        tracked one, or the global one for ``replanning_minco_v3``).
        ``forward_axis`` must already be resolved (never ``None``)."""
        waypoints = [p0, *via_waypoints, p_target]

        mode = trajectory_tracking_mode
        if mode not in TRAJECTORY_TRACKING_MODES:
            self._log.warn(
                "[TrackerBuilder] unknown trajectory_tracking_mode=%r, "
                "falling back to 'static'" % trajectory_tracking_mode
            )
            mode = "static"
        if mode == "replanning_minco_v3" and self._max_accel is None and not minco_freetime:
            # Only the (one-time) global build's heuristic-time path needs
            # max_accel; minco_freetime doesn't.
            self._log.warn(
                "[TrackerBuilder] trajectory_tracking_mode='replanning_minco_v3' "
                "requires max_accel to be configured (unless minco_freetime=True) "
                "-- falling back to 'static'"
            )
            mode = "static"

        traj = None
        v3_tracker = None
        self._obstacle_tracker = None
        obstacle_kwargs = {}
        if mode == "replanning_minco_v3" and minco_obstacle_avoidance:
            obstacle_kwargs = self._obstacle_tracker_kwargs(
                minco_local_piece_length_m, minco_obstacle_clearance_soft)
        if mode == "replanning_minco_v3":
            # Builds its own trajectory internally; `traj` is set to its
            # global MincoTrajectory only for the speed path preview.
            #
            # This must run BEFORE toppra_ready is evaluated below: on
            # infeasibility it downgrades `mode` to "static", and
            # toppra_ready's `mode == "static"` check needs to see that
            # downgraded value -- otherwise the fallback silently skips
            # TOPP-RA and lands on the envelope-unaware legacy Hermite path
            # (docs/archive/achieved/2026-09-18_replanning_minco_v2_zeno_route_rediagnosis_toppra_ready_bug.md).
            try:
                v3_tracker = ReplanningMincoV3Tracker(
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
                    face_travel=face_travel and minco_v3_face_travel,
                    forward_axis=forward_axis,
                    local_max_vel=minco_local_max_vel,
                    async_replan=minco_v3_async_replan,
                    **obstacle_kwargs,
                )
                if obstacle_kwargs:
                    self._obstacle_tracker = v3_tracker
                    if self._obstacle_map.grid is not obstacle_kwargs["obstacle_grid"]:
                        v3_tracker.set_obstacle_grid(self._obstacle_map.grid)
                traj = v3_tracker.trajectory
                self._log.info(
                    "[TrackerBuilder] initial replanning_minco_v3 global "
                    "solve took %.2fs (%d waypoints)"
                    % (traj.solve_wall_seconds, traj.num_waypoints)
                )
            except (MincoInfeasibleError, ValueError) as exc:
                self._log.warn(
                    "[TrackerBuilder] replanning_minco_v3 tracker "
                    "construction failed (%s) -- falling back to 'static'" % exc
                )
                v3_tracker = None
                traj = None
                mode = "static"

        # static mode tries the force/torque-aware TOPP-RA path first (docs/
        # 2026-08-28_constrained_trajectory_generation_research.md).
        toppra_ready = (
            mode == "static"
            and self._wrench_envelope is not None
            and self._mass is not None
            and self._inertia is not None
            and self._max_angular_rate is not None
        )
        if toppra_ready:
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
                self._log.info(
                    "[TrackerBuilder] TOPP-RA trajectory used (%d waypoints, "
                    "duration=%.2fs, build_time=%.3fs)"
                    % (len(waypoints), traj.global_total_duration, build_wall_seconds)
                )
            except TrajectoryInfeasibleError as exc:
                self._log.warn(
                    "[TrackerBuilder] TOPP-RA time-parameterization "
                    "infeasible (%s) -- falling back to the Hermite static "
                    "path" % exc
                )
                traj = None

        if mode == "static_minco":
            # 再計画は一切しない: MINCOをgoal受付時に一度だけ解いて、
            # StaticTrajectoryTracker（下のelse節）でそのまま完走させる。
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
                self._log.info(
                    "[TrackerBuilder] static MINCO solve took %.2fs "
                    "(%d waypoints, via_half_width=%.2f, "
                    "attitude_resample_spacing_m=%s, wrench_safety_margin=%.2f)"
                    % (traj.solve_wall_seconds, traj.num_waypoints,
                       minco_via_half_width, minco_attitude_resample_spacing_m,
                       minco_wrench_safety_margin)
                )
            except MincoInfeasibleError as exc:
                self._log.warn(
                    "[TrackerBuilder] static MINCO trajectory infeasible "
                    "(%s) -- falling back to the Hermite static path" % exc
                )
                traj = None

        if traj is None:
            segment_times = HeuristicSegmentTimeAllocator(
                target_speed=self._target_speed, max_accel=self._max_accel
            ).allocate(waypoints)
            coeffs = _hermite.HermiteSplineTrajectoryGenerator().generate(
                waypoints, segment_times
            )
            traj = Trajectory(
                waypoints, segment_times, coeffs,
                attitude_speed_threshold=self._attitude_speed_threshold,
                forward_axis=forward_axis,
                initial_q_des=q0, face_travel=face_travel,
                max_angular_rate=self._max_angular_rate,
            )

        if mode == "replanning_minco_v3":
            return v3_tracker, traj
        return StaticTrajectoryTracker(traj), traj

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
