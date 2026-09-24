#!/usr/bin/env python3
"""guidance_node parameters: in-code defaults, the read-only set, and the per-goal
read that feeds ``GuidanceExecutor.execute()``.

Split out of ``guidance.py``. The values normally come from
``config/gnc_params.yaml`` (``launch/guidance.launch.py``); a bare ``ros2 run``
uses the defaults here.
"""
GUIDANCE_PARAM_DEFAULTS = {
    "guidance.target_speed": 0.5,
    "guidance.attitude_speed_threshold": 0.02,
    "guidance.align_tolerance_deg": 3.0,
    "guidance.align_timeout": 60.0,
    "guidance.align_settle_time": 0.5,
    "guidance.align_pos_tolerance_m": 0.05,
    "guidance.align_pos_settle_time": 0.5,
    "guidance.align_pos_timeout": 10.0,
    "guidance.tf_staleness_timeout": 1.0,
    "guidance.rate": 50.0,
    "guidance.velocity_estimate_rate": 10.0,
    "guidance.velocity_estimate_alpha": 0.3,
    "guidance.camera_forward_axis.main": [1.0, 0.0, 0.0],
    "guidance.camera_forward_axis.stereo": [0.0, 1.0, 0.0],
    "guidance.attitude_reference_mode": "face_travel",
    "guidance.pre_align": True,
    "guidance.align_at_arrival": True,
    "guidance.look_at_target_frame": "",
    # Optional ordered interior relay points (docs/
    # 2026-08-25_guidance_waypoint_insertion_curve_verification.md,
    # generalized from a single point to a list 2026-08-31, docs/
    # 2026-08-31_curve_aware_realtime_replanning_design_discussion.md): TF
    # frame names (e.g. maps/iss_location.yaml entries), resolved in order
    # via self._tf at goal receipt in _execute_fn, same Category B latching
    # as the other per-goal options here. [] (default) means no via
    # waypoints -- unchanged prior 2-waypoint behavior.
    "guidance.via_waypoints": [""],
    # "static_minco"/"replanning_minco_v3" only: MincoTrajectory's via-point
    # free-variable box half-width [m] (docs/
    # 2026-08-30_static_minco_face_travel_gap.md 追記3 -- was a hardcoded
    # C++ constant in minco_solver.cpp, now tunable without a rebuild). 0.0
    # pins every via waypoint exactly (TOPPRA-style hard pass-through) --
    # default (2026-09-01, prior default was 0.3): via waypoints should be
    # hit exactly unless a goal explicitly opts into slack. Same Category B
    # per-goal latching as via_waypoints above.
    "guidance.minco_via_half_width": 0.0,
    # "static_minco"/"replanning_minco_v3" only:
    # MincoTrajectory's attitude_resample_spacing_m (docs/
    # 2026-08-30_static_minco_face_travel_gap.md 追記4). 0.0 means "off"
    # (None -- attitude only seeded at the given waypoints, prior behavior);
    # a positive value densifies face-travel attitude seeding every that
    # many meters along each segment without changing the position path
    # shape. Default 0.3 (2026-09-01, prior default was 0.0): attitude
    # tracking observed to lag badly over long legs (e.g. nav_entry ->
    # inspection_entry_1, ~4.7m) with seeding only at waypoints. Same
    # Category B per-goal latching as via_waypoints above.
    "guidance.minco_attitude_resample_spacing_m": 0.3,
    "guidance.face_travel_camera": "main",
    "guidance.align_at_arrival_camera": "main",
    # "static" (default), "static_minco" or "replanning_minco_v3". Category
    # B, like attitude_reference_mode -- latched at goal receipt in
    # goal_execute_kwargs below, not applied mid-trajectory.
    "guidance.trajectory_tracking_mode": "static",
    # q_des rate limit (docs/archive/achieved/
    # 2026-08-24_trajectory_state_carryover_design.md 3-4節). First-cut
    # default, not yet tuned against real tracking performance.
    "guidance.max_angular_rate_deg": 90.0,
    # SLERP+trapezoid align ramp (docs/2026-08-27_align_slerp_trapezoid_
    # next_steps.md): _align_to() feeds the checkpoint a moving intermediate
    # target along this profile instead of stepping straight to the goal
    # attitude, removing composite-axis overshoot (docs/
    # 2026-08-27_composite_axis_overshoot_summary_and_plan.md). Read-only
    # (see STATIC_PARAMS below): only read at GuidanceExecutor construction,
    # and align_angular_accel_deg specifically does not auto-track
    # control_node gain changes -- re-derive by hand if those gains change.
    "guidance.align_angular_speed_deg": 15.0,
    "guidance.align_angular_accel_deg": 2.4,
    "guidance.align_traj_publish_rate_hz": 20.0,
    # static mode: shrinks wrench_envelope_halfspaces (see that function's
    # docstring and docs/2026-08-28_toppra_static_path_attitude_overshoot_
    # incident.md "追記（2026-08-28 その5/6）"). Only read once at
    # wrench_envelope construction in guidance.py -- static like the fan geometry it's
    # paired with. static_minco/replanning_minco_v3 reuse this same value,
    # forwarded to MincoTrajectory's wrench_safety_margin each goal (docs/
    # 2026-08-30_static_minco_face_travel_gap.md 追記2) -- one physical
    # meaning (feedback headroom against the fan envelope), one parameter,
    # rather than a second minco-specific margin that could drift from this
    # one.
    "guidance.wrench_envelope_safety_margin": 0.7,
    # Post-cancel brake (docs/archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md),
    # JAXA ctl.yaml values. eta reuses wrench_envelope_safety_margin.
    "guidance.stopping.x_threshold": 0.05,
    "guidance.stopping.theta_threshold": 0.017453292519943,
    "guidance.stopping.f_max": 181.0032e-3,
    "guidance.stopping.t_max": 8.1904e-3,
    "guidance.stopping.tolerance_pos": 0.30,
    "guidance.stopping.tolerance_att": 1.0,
    "guidance.stopping.duration_goal": 3.0,
    "guidance.stopping.wait_cancel": 10.0,
    # "replanning_minco_v3" only: forwarded to
    # GuidanceExecutor.execute()'s minco_freetime (see that method's
    # docstring). False (default, unchanged prior behavior) uses this
    # mode's configured target_speed/max_accel (plan_minco_heuristic_time)
    # for the global build; True switches the global build to
    # MincoTrajectory's free-time plan_minco path instead
    # (docs/2026-09-20_minco_global_replan_freetime_switch_offline_
    # investigation.md).
    "guidance.minco_freetime": False,
    # "replanning_minco_v3" only: forwarded to the tracker's
    # local_replan_period/planning_horizon_m. Category B, latched per goal.
    "guidance.minco_local_replan_period": 1.0,
    "guidance.minco_planning_horizon_m": 4.0,
    "guidance.minco_v3_face_travel": True,
    "guidance.minco_local_max_vel": 0.15,
    "guidance.minco_v3_async_replan": True,
    # replanning_minco_v3 obstacle avoidance (docs/2026-09-24_obstacle_avoidance_
    # production_integration_plan.md). The map/grid ones are read once at startup;
    # the minco_* ones latch per goal like the rest.
    "guidance.obstacle_map_file": "jem_octomap.bt",  # relative = share/maps/; "" = no static map
    "guidance.obstacle_grid_resolution": 0.1,
    # Robot radius 0.1 m + 0.1 m margin, applied in whole cells (0.15 = 0.2 at 0.1 m resolution).
    "guidance.obstacle_grid_inflation": 0.2,
    "guidance.minco_obstacle_avoidance": False,
    "guidance.minco_local_piece_length_m": 1.5,
    "guidance.minco_obstacle_clearance_soft": 0.2,
}

# Timer periods are only ever read at node construction (guidance.py), so
# changing them at runtime would silently have no effect -- read-only
# like guidance.rate (docs/guidance_velocity_estimator_design.md 5 節).
STATIC_PARAMS = frozenset({
    "guidance.rate", "guidance.velocity_estimate_rate",
    # max_angular_rate_deg: no dynamic-reconfigure design has been
    # done for it yet (docs/archive/achieved/
    # 2026-08-24_trajectory_state_carryover_design.md only decided the
    # value/semantics, not a Category-A wiring) -- read-only until
    # that's explicitly designed.
    "guidance.max_angular_rate_deg",
    # Only read at GuidanceExecutor construction (see the ramp's own
    # comment above); no Category-A wiring exists for these either.
    "guidance.align_angular_speed_deg", "guidance.align_angular_accel_deg",
    "guidance.align_traj_publish_rate_hz",
    "guidance.wrench_envelope_safety_margin",
    "guidance.stopping.x_threshold", "guidance.stopping.theta_threshold",
    "guidance.stopping.f_max", "guidance.stopping.t_max",
    "guidance.stopping.tolerance_pos", "guidance.stopping.tolerance_att",
    "guidance.stopping.duration_goal", "guidance.stopping.wait_cancel",
    "guidance.obstacle_map_file", "guidance.obstacle_grid_resolution",
    "guidance.obstacle_grid_inflation",
})

ATTITUDE_REFERENCE_MODES = frozenset({"fixed", "face_travel", "look_at"})
CAMERA_NAMES = frozenset({"main", "stereo"})

# execute() keyword -> (guidance.* parameter, type), read at goal receipt so each
# goal latches the values in effect at that moment (Category B,
# docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md).
_GOAL_EXECUTE_PARAMS = {
    "minco_via_half_width": ("minco_via_half_width", float),
    "minco_wrench_safety_margin": ("wrench_envelope_safety_margin", float),
    "face_travel_camera": ("face_travel_camera", str),
    "pre_align": ("pre_align", bool),
    "align_at_arrival": ("align_at_arrival", bool),
    "look_at_target_frame": ("look_at_target_frame", str),
    "align_at_arrival_camera": ("align_at_arrival_camera", str),
    "trajectory_tracking_mode": ("trajectory_tracking_mode", str),
    "minco_freetime": ("minco_freetime", bool),
    "minco_local_replan_period": ("minco_local_replan_period", float),
    "minco_planning_horizon_m": ("minco_planning_horizon_m", float),
    "minco_v3_face_travel": ("minco_v3_face_travel", bool),
    "minco_local_max_vel": ("minco_local_max_vel", float),
    "minco_v3_async_replan": ("minco_v3_async_replan", bool),
    "minco_obstacle_avoidance": ("minco_obstacle_avoidance", bool),
    "minco_local_piece_length_m": ("minco_local_piece_length_m", float),
    "minco_obstacle_clearance_soft": ("minco_obstacle_clearance_soft", float),
}


def declare_guidance_params(node, static_descriptor):
    for name, default in GUIDANCE_PARAM_DEFAULTS.items():
        descriptor = static_descriptor if name in STATIC_PARAMS else None
        node.declare_parameter(name, default, descriptor)


def goal_execute_kwargs(node):
    """``execute()`` keyword arguments from the current ``guidance.*`` parameters
    (everything except ``via_waypoints`` and ``face_travel``, which need TF and the
    attitude-reference fallback)."""
    kwargs = {key: kind(node.get_parameter("guidance." + name).value)
              for key, (name, kind) in _GOAL_EXECUTE_PARAMS.items()}
    spacing = float(node.get_parameter("guidance.minco_attitude_resample_spacing_m").value)
    kwargs["minco_attitude_resample_spacing_m"] = spacing if spacing > 0.0 else None
    return kwargs
