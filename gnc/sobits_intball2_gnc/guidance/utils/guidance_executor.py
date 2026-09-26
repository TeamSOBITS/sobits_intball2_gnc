#!/usr/bin/env python3
"""Guidance move-to-target executor (ROS-agnostic orchestration logic).

Implements the ``execute_fn`` body for
:class:`~sobits_intball2_gnc.guidance.ros.ctl_command_action_server.CtlCommandActionServer`
(``docs/guidance_node_implementation_plan.md``): current pose (TF) -> target
pose -> a trajectory tracker (:class:`~sobits_intball2_gnc.guidance.
trajectory_tracking.tracker_builder.TrackerBuilder`) -> a sim-clock-paced publish loop onto
``/gnc/trajectory_setpoint``, folding in the pre-/post-alignment steps that
``test/manual/send_curve_via_naventry_to_*_facing_direction.py`` scripts have
so far done ad hoc per-script.

Like :class:`~sobits_intball2_gnc.control.utils.hover_controller.HoverController`,
this class does not subclass ``Node`` or import ``rclpy``: it is driven by
already-constructed ROS I/O wrappers (``tf_client``, ``setpoint_publisher``,
``checkpoint_publisher``) and two injected callables (``clock_seconds_fn``,
``spin_fn``) so it is directly unit-testable with fakes, following this
package's DI convention (see ``docs/architecture_guidelines.md``).

Only a ``(current_pose, [via_waypoints...], p_target)`` trajectory is
generated per call -- no global path planning is invoked (``docs/
guidance_node_implementation_plan.md`` decision 1). ``via_waypoints`` is an
optional ordered list of interior relay points (``docs/
2026-08-25_guidance_waypoint_insertion_curve_verification.md``, generalized
from a single point 2026-08-31), passed through this goal's ``execute()``
call as already TF-resolved positions -- resolving a further chain of
several *separate goals* (a client streaming several ``CtlCommand`` goals in
a row stops at each one, same granularity as ``PoseCorrector``'s existing
checkpoint chaining) is still out of scope here.
"""
import numpy as np

from sobits_intball2_gnc.guidance.align.attitude_aligner import AttitudeAligner
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    DEFAULT_LOCAL_REPLAN_PERIOD_S,
    DEFAULT_PLANNING_HORIZON_M,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.tracker_builder import (
    TrackerBuilder,
    TrajectoryBuildError,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    compute_camera_relative_quat,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.cancel_brake import CancelBrake

STATUS_SUCCESS = "success"
STATUS_ABORTED = "aborted"
STATUS_CANCELED = "canceled"
STATUS_PLANNING_FAILED = "planning_failed"

DEFAULT_CAMERA_FORWARD_AXIS = {
    "main": (1.0, 0.0, 0.0),
    "stereo": (0.0, 1.0, 0.0),
}


def _goal_position(tracker, p_target):
    """The tracker's own goal when it can move it (an obstacle over the requested one)."""
    goal = getattr(tracker, "goal_position", None)
    return np.asarray(p_target if goal is None else goal, dtype=float)


class GuidanceExecutor:
    """Drive one ``CtlCommand`` move-to-target goal to completion.

    Args:
        tf_client: current-pose source (``get_pose() -> (pos, quat, stamp)``
            or ``None``).
        setpoint_publisher: ``MultiDOFJointTrajectoryPublisher``-shaped
            object (``publish(p_des, v_des, a_des, q_des)``).
        checkpoint_publisher: ``CheckpointPublisher``-shaped object
            (``publish(pos, quat)``, ``wait_for_subscriber(timeout_sec)``).
        clock_seconds_fn: callable returning the current sim-clock time [s]
            (e.g. ``lambda: node.get_clock().now().nanoseconds * 1e-9``) --
            sim time, not wall-clock, for the reason documented in
            ``docs/recording_cpu_load_control_degradation.md``.
        spin_fn: callable ``spin_fn(seconds)`` invoked once per loop tick to
            let ROS callbacks run and pace the loop (e.g. ``rclpy.spin_once``
            + a short sleep); this class never touches ``rclpy`` directly.
        logger: object with ``.info``/``.warn`` (e.g. a node's logger).
        target_speed/attitude_speed_threshold/align_tolerance_deg/
            align_timeout/rate/tf_staleness_timeout: see
            ``config/gnc_params.yaml``'s ``guidance`` section.
        camera_forward_axis: ``{camera_name: [x, y, z]}`` body-frame forward
            axis per ``face_travel_camera`` option (default:
            ``DEFAULT_CAMERA_FORWARD_AXIS``, TF-measured 2026-08-20).
        velocity_fn: optional callable returning the latest
            ``VelocityEstimator.get()`` result (``docs/
            guidance_velocity_estimator_design.md``); :meth:`brake`'s v0.
        angular_velocity_fn: optional callable returning the latest
            body-frame gyro ``[wx, wy, wz]`` or ``None``; :meth:`brake`'s w0.
        allocator: ``ThrustAllocator`` configured like control's; required
            by :meth:`brake`.
        stopping_profile_kwargs: extra ``StoppingProfile`` keyword args
            (``eta``, ``f_max``, ``x_threshold``, ``max_axis_force`` ...).
        stopping_tolerance_pos/att, stopping_duration_goal,
            stopping_wait_cancel: :meth:`brake`'s end conditions (JAXA
            ``ctl.tolerance_pos_stop``/``tolerance_att_stop``/
            ``duration_goal``/``wait_cancel``).
        obstacle_map: optional ``ObstacleMap``; ``replan_minco`` goals
            with ``minco_obstacle_avoidance`` avoid its grid, and grids it
            rebuilds mid-goal are handed to the running tracker.
    """

    def __init__(self, tf_client, setpoint_publisher, checkpoint_publisher,
                 clock_seconds_fn, spin_fn, logger,
                 target_speed=0.5, attitude_speed_threshold=0.02,
                 align_tolerance_deg=3.0, align_timeout=60.0,
                 align_settle_time=0.5, rate=50.0,
                 camera_forward_axis=None, speed_path_publisher=None,
                 local_speed_path_publisher=None,
                 path_preview_points=20, max_accel=None,
                 align_pos_tolerance_m=0.05, align_pos_settle_time=0.5,
                 align_pos_timeout=10.0, tf_staleness_timeout=1.0,
                 velocity_fn=None, max_angular_rate=None,
                 align_angular_speed_deg=None, align_angular_accel_deg=None,
                 align_traj_publish_rate_hz=20.0,
                 wrench_envelope=None, mass=None, inertia=None,
                 angular_velocity_fn=None, allocator=None,
                 stopping_profile_kwargs=None, stopping_tolerance_pos=0.30,
                 stopping_tolerance_att=1.0, stopping_duration_goal=3.0,
                 stopping_wait_cancel=10.0, obstacle_map=None):
        self._tf = tf_client
        self._setpoint_pub = setpoint_publisher
        self._checkpoint_pub = checkpoint_publisher
        self._clock_seconds = clock_seconds_fn
        self._spin = spin_fn
        self._log = logger
        self._target_speed = float(target_speed)
        # Vehicle's achievable acceleration [m/s^2], e.g.
        # trajectory_controller.max_force / mass -- replan_minco's
        # heuristic segment times must allow a rest-to-rest profile at it.
        self._max_accel = None if max_accel is None else float(max_accel)
        self._attitude_speed_threshold = float(attitude_speed_threshold)
        # Position-error counterpart of AttitudeAligner's align_tolerance_rad/
        # align_settle_time/align_timeout trio, but kept as separate
        # parameters (not shared) -- position and attitude correction have
        # different loop dynamics (mass vs. inertia, kp_pos vs. kp_att), same
        # reasoning PoseCorrector already applies by keeping its own
        # align_tolerance_deg/align_settle_time independent of this class's.
        # See docs/align_at_arrival_position_based_plan.md.
        self._align_pos_tolerance_m = float(align_pos_tolerance_m)
        self._align_pos_settle_time = float(align_pos_settle_time)
        # Deliberately much shorter than align_timeout (60s default): this
        # only waits out the residual translation error left after
        # _run_trajectory's planned duration elapses (expected: a few cm,
        # settled by the existing P+D trajectory_controller in a few
        # seconds), not a whole goal's worth of safety margin.
        self._align_pos_timeout = float(align_pos_timeout)
        # TF liveness (docs/guidance_realtime_replanning_design.md 6-8/7-4):
        # a TF lookup is a pull from tf2's buffer and keeps succeeding after
        # the publisher stops, so liveness is judged by whether the stamp
        # itself advances, timed on this class's own sim clock -- mirrors
        # PoseCorrector._classify's convention (control/utils/
        # pose_corrector.py). _tf_last_stamp/_tf_last_advance_t persist for
        # this executor's whole lifetime (spanning multiple goals), same as
        # PoseCorrector's equivalent state.
        self._tf_staleness_timeout = float(tf_staleness_timeout)
        self._tf_last_stamp = None
        self._tf_last_advance_t = None
        self._dt = 1.0 / float(rate)
        # q_des rate limit for TOPP-RA's face-travel attitude (docs/archive/
        # achieved/2026-08-24_trajectory_state_carryover_design.md 3-4節).
        self._max_angular_rate = (
            None if max_angular_rate is None else float(max_angular_rate)
        )
        self._camera_forward_axis = dict(
            camera_forward_axis or DEFAULT_CAMERA_FORWARD_AXIS
        )
        # Optional: a SpeedPathPublisher (guidance/ros/speed_path_publisher.py)
        # for RViz-only visualization of the planned trajectory, colored by
        # speed -- has no bearing on control behavior, see that module's
        # docstring.
        self._speed_path_pub = speed_path_publisher
        self._local_speed_path_pub = local_speed_path_publisher
        self._path_preview_points = int(path_preview_points)
        # Real actuator-derived budget for the static/TOPP-RA path (docs/
        # 2026-08-28_constrained_trajectory_generation_research.md,
        # docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md
        # "追記（2026-08-28 その2）") -- None (any of the three, or
        # max_angular_rate) makes static-mode goals abort.
        # wrench_envelope is a static (F, g) half-space pair from
        # actuation_envelope.wrench_envelope_halfspaces -- passed straight
        # through to ToppraTrajectory, not touched here.
        self._wrench_envelope = wrench_envelope
        self._mass = None if mass is None else float(mass)
        self._inertia = None if inertia is None else float(inertia)
        # Attitude-only alignment (pre_align/align_at_arrival), extracted out
        # of this class (docs/2026-08-29_guidance_dir_and_dead_code_survey.md)
        # -- constructed last since it needs self._dt and self._tf_pose_fresh,
        # both set above.
        # velocity_fn: Guidance-side TF velocity estimate (docs/
        # guidance_velocity_estimator_design.md), independent of
        # TrajectoryController's own estimator (one-way Guidance -> Control
        # data flow). Fed by a low-rate timer on another thread.
        self._brake = CancelBrake(
            tf_client, self._tf_pose_fresh, setpoint_publisher, checkpoint_publisher,
            clock_seconds_fn, spin_fn, logger, self._dt, allocator, self._mass,
            self._inertia, stopping_profile_kwargs, stopping_tolerance_pos,
            stopping_tolerance_att, stopping_duration_goal, stopping_wait_cancel,
            velocity_fn=velocity_fn, angular_velocity_fn=angular_velocity_fn,
        )
        self._tracker_builder = TrackerBuilder(
            tf_client, self._tf_pose_fresh, logger, self._target_speed, self._max_accel,
            self._max_angular_rate, self._wrench_envelope,
            self._mass, self._inertia, obstacle_map=obstacle_map,
            stop_profile_fn=self._brake.profile_from if self._brake.available else None,
        )
        self._aligner = AttitudeAligner(
            tf_client, checkpoint_publisher, spin_fn, clock_seconds_fn, logger,
            tf_fresh_fn=self._tf_pose_fresh, dt=self._dt,
            align_tolerance_deg=align_tolerance_deg, align_timeout=align_timeout,
            align_settle_time=align_settle_time,
            align_angular_speed_deg=align_angular_speed_deg,
            align_angular_accel_deg=align_angular_accel_deg,
            align_traj_publish_rate_hz=align_traj_publish_rate_hz,
        )

    def set_gains(self, align_tolerance_deg=None, align_timeout=None,
                  align_settle_time=None, align_pos_tolerance_m=None,
                  align_pos_settle_time=None, align_pos_timeout=None,
                  tf_staleness_timeout=None) -> None:
        """Update the align convergence threshold/timeout in place (dynamic
        reconfiguration, see docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md
        category A). The align-wait loop in :meth:`execute` reads these
        instance attributes every iteration, so a change here is picked up
        mid-align on the very next loop tick -- it is not restricted to
        goal boundaries, unlike ``target_speed``/``attitude_speed_threshold``
        (category B, latched via a fresh trajectory generation instead).
        """
        self._aligner.set_gains(
            align_tolerance_deg=align_tolerance_deg, align_timeout=align_timeout,
            align_settle_time=align_settle_time,
        )
        if align_pos_tolerance_m is not None:
            self._align_pos_tolerance_m = float(align_pos_tolerance_m)
        if align_pos_settle_time is not None:
            self._align_pos_settle_time = float(align_pos_settle_time)
        if align_pos_timeout is not None:
            self._align_pos_timeout = float(align_pos_timeout)
        if tf_staleness_timeout is not None:
            self._tf_staleness_timeout = float(tf_staleness_timeout)

    def _tf_pose_fresh(self, stamp) -> bool:
        """Return whether ``stamp`` (get_pose()'s third element) indicates a
        live TF stream, updating this executor's liveness state as a side
        effect.

        Mirrors ``PoseCorrector._classify``'s convention: a TF lookup pulls
        from tf2's buffer and keeps succeeding after the publisher stops, so
        liveness is judged by whether the stamp *advances*, timed on this
        class's own sim clock -- never by comparing the stamp directly
        against that clock (the stamp may be on an unrelated clock epoch,
        see ``TfClient.get_pose()``'s docstring).
        """
        stamp = float(stamp)
        if stamp == 0.0:
            # A zero stamp means "unset" in ROS, not "time zero" -- tf2 can
            # hand one back for an early lookup while the listener is still
            # filling in the transform chain (see tf_race_investigation.md).
            return False
        t = self._clock_seconds()
        if stamp != self._tf_last_stamp:
            # Either the stamp is genuinely advancing (the normal case), or
            # it went backwards (a simulator restart) -- either way, adopt
            # it as the new reference point rather than reporting a stall.
            self._tf_last_stamp = stamp
            self._tf_last_advance_t = t
            return True
        return (t - self._tf_last_advance_t) <= self._tf_staleness_timeout

    def execute(self, p_target, q_target, feedback_cb, is_cancel_requested,
                face_travel=True, face_travel_camera="main",
                align_at_arrival=True, pre_align=True, look_at_target_frame="",
                align_at_arrival_camera="main", trajectory_tracking_mode="static_toppra",
                via_waypoints=None, minco_via_half_width=0.3,
                minco_attitude_resample_spacing_m=None,
                minco_wrench_safety_margin=1.0, minco_freetime=False,
                minco_local_replan_period=DEFAULT_LOCAL_REPLAN_PERIOD_S,
                minco_planning_horizon_m=DEFAULT_PLANNING_HORIZON_M,
                minco_replan_face_travel=False, minco_local_max_vel=None,
                minco_async_replan=False, minco_obstacle_avoidance=False,
                minco_local_piece_length_m=None, minco_obstacle_clearance_soft=0.2):
        """Run one move-to-target goal; returns a ``STATUS_*`` constant.

        ``via_waypoints``: an optional ordered list of interior relay points
        (already TF-resolved to ``[x, y, z]`` positions by the caller, e.g.
        ``guidance.py``'s ``guidance.via_waypoints`` param), inserted between
        ``p0`` and ``p_target`` in order (``docs/
        2026-08-25_guidance_waypoint_insertion_curve_verification.md``,
        generalized from a single point 2026-08-31). Their own orientation
        is not used for anything -- only their positions feed the
        ``(2 + len(via_waypoints))``-waypoint trajectory and ``pre_align``'s
        initial facing direction (the first via point). ``None``/``[]``
        (default) reproduces the prior 2-waypoint behavior exactly.

        ``minco_via_half_width``: ``"static_minco"``/``"replan_minco"``
        only -- forwarded to ``MincoTrajectory``'s ``via_half_width`` (see
        that class's docstring and ``docs/
        2026-08-30_static_minco_face_travel_gap.md`` 追記3). ``0.0`` pins
        each via point exactly (TOPPRA-style hard pass-through); the
        default ``0.3`` matches the box MINCO originally hardcoded. Ignored
        by every other mode.

        ``minco_attitude_resample_spacing_m``: ``"static_minco"``/
        ``"replan_minco"`` only -- forwarded to ``MincoTrajectory``'s
        ``attitude_resample_spacing_m`` (docs/
        2026-08-30_static_minco_face_travel_gap.md 追記4). ``None``
        (default) reproduces prior behavior (attitude only seeded at the
        given waypoints). Ignored by every other mode.

        ``minco_wrench_safety_margin``: ``"static_minco"``/
        ``"replan_minco"`` only -- forwarded to ``MincoTrajectory``'s
        ``wrench_safety_margin`` (docs/
        2026-08-30_static_minco_face_travel_gap.md 追記2). ``1.0`` (default)
        reproduces prior behavior (envelope unshrunk). Ignored by every
        other mode.

        ``minco_freetime``: ``"replan_minco"`` only -- if ``True``,
        the (one-time) global build solves with
        ``target_speed=None, max_accel=None``
        (``MincoTrajectory``'s free-time ``plan_minco`` path -- segment
        times optimized jointly with waypoints, not fixed by a heuristic +
        analytic stretch) instead of this executor's configured
        ``target_speed``/``max_accel``. Bypasses the mode's own
        ``max_accel``-required fallback-to-``"static_toppra"`` check (that check
        exists for the heuristic-time path, which needs ``max_accel`` to
        size its initial guess -- free-time doesn't). Ignored by every
        other mode. ``False`` (default) reproduces prior behavior.

        ``minco_local_replan_period``/``minco_planning_horizon_m``:
        ``"replan_minco"`` only -- forwarded to
        ``ReplanMincoTracker``'s ``local_replan_period``/
        ``planning_horizon_m``. Non-positive values fall back to
        ``"static_toppra"``. Ignored by every other mode.

        ``minco_replan_face_travel``/``minco_local_max_vel``: ``"replan_minco"``
        only -- with ``face_travel`` also set, the tracker faces travel
        (``ReplanMincoTracker``'s ``face_travel``/``local_max_vel``).
        ``False`` (default) keeps the fixed-``q0`` behavior.

        ``minco_async_replan``: ``"replan_minco"`` only --
        ``ReplanMincoTracker``'s ``async_replan`` (local solve off the
        setpoint loop). ``False`` (default) solves inside ``sample()``.

        ``minco_obstacle_avoidance``/``minco_local_piece_length_m``/
        ``minco_obstacle_clearance_soft``: ``"replan_minco"`` with face
        travel only -- the tracker avoids the ``obstacle_map`` grid (EGO-Planner
        v2 rebound, multi-piece local of this piece length) and emergency-stops
        along a ``StoppingProfile`` when a replan cannot clear a collision.
        Ignored without an ``obstacle_map``.

        ``look_at_target_frame`` is accepted but currently unused -- reserved
        for the future ``look_at`` attitude-reference mode (docs/
        movement_mode_design.md), which still needs a per-tick TF lookup this
        class does not yet perform.

        ``align_at_arrival_camera``: on arrival, align to whatever
        orientation the goal's ``q_target`` implies for this camera's axis
        -- ``"main"`` aligns to ``q_target`` as-is; any other camera (e.g.
        ``"stereo"``) aligns so that camera's axis ends up facing where the
        main camera would have under ``q_target`` (:func:`compute_
        camera_relative_quat`), i.e. "show me through a different camera
        whatever the main camera would have seen".

        ``trajectory_tracking_mode``: ``"static_toppra"`` (default) samples a single
        open-loop TOPP-RA trajectory generated at goal start (needs
        ``wrench_envelope``/``mass``/``inertia``/``max_angular_rate``).

        ``"static_minco"`` is like ``"static_toppra"`` (single open-loop trajectory,
        no TF-driven re-planning) but backed by ``MincoTrajectory`` instead
        of TOPP-RA.

        ``"replan_minco"`` (``docs/
        2026-09-20_ego_v2_style_replan_migration_plan.md``): an
        EGO-Planner-v2-style global/local tracker (:class:`~sobits_
        intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker.
        ReplanMincoTracker`) where the global layer is solved **once**
        and the local layer is a periodically (every ``local_replan_period``
        seconds) re-solved free-time segment, started from the previous
        local trajectory's reference state, that *is* the tracked
        trajectory. Attitude is out of scope for this mode (always commands
        ``q0``). Needs no ``velocity_fn``; needs ``max_accel`` unless
        ``minco_freetime=True``. Falls back to ``"static_toppra"`` if ``max_accel``
        is missing or the initial global/local solve is infeasible.

        Returns ``STATUS_PLANNING_FAILED`` (the caller brakes) when no
        trajectory can be built, including an unrecognized mode.
        """
        pose = self._tf.get_pose()
        if pose is None:
            self._log.warn("[GuidanceExecutor] no TF pose available, aborting")
            return STATUS_ABORTED
        p0, q0, stamp = pose
        if not self._tf_pose_fresh(stamp):
            self._log.warn(
                "[GuidanceExecutor] TF pose is stale (stamp not advancing "
                "for >%.1fs), aborting" % self._tf_staleness_timeout
            )
            return STATUS_ABORTED
        via_waypoints = [] if via_waypoints is None else list(via_waypoints)

        forward_axis = self._camera_forward_axis.get(face_travel_camera)
        if face_travel and forward_axis is None:
            self._log.warn(
                "[GuidanceExecutor] unknown face_travel_camera=%r, "
                "falling back to 'main'" % face_travel_camera
            )
            forward_axis = self._camera_forward_axis["main"]

        if face_travel and pre_align:
            # Use the chord to the target, not a sampled trajectory
            # velocity: v(0) == 0 by construction (see
            # HermiteSplineTrajectoryGenerator), so an early-in-the-segment
            # velocity sample can land below attitude_speed_threshold and
            # get silently treated as "no direction change needed" by
            # compute_q_des's low-speed guard -- this previously made
            # pre_align skip entirely for a target straight behind the
            # vehicle (see
            # docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md).
            # This is exactly the direction v_early would have pointed
            # anyway (HermiteSplineTrajectoryGenerator interpolates a
            # 2-waypoint, at-rest-at-both-ends trajectory as a straight
            # line, so every non-zero v(t) already points p0 -> p_target).
            #
            # With via_waypoints, face the first leg's chord (p0 -> first via)
            # since that's the direction of imminent travel -- unlike the
            # 2-point case this is only an approximation once the via
            # points' interior tangents are nonzero (docs/
            # 2026-08-25_guidance_waypoint_insertion_curve_verification.md),
            # but it's a reasonable initial facing since face_travel's
            # continuous per-tick compute_q_des (below) takes over once
            # translation starts.
            #
            # Pass q0 (not None) so compute_q_des fills the pointing task's
            # free roll DOF to match the vehicle's current roll instead of
            # the shortest-arc convention's incidental value -- otherwise
            # pre_align chases a functionally unnecessary roll change (see
            # docs/2026-08-21_tf_correction_align_optimization.md 8節).
            pre_align_target = via_waypoints[0] if via_waypoints else p_target
            q_align = compute_q_des(
                np.asarray(pre_align_target, dtype=float) - np.asarray(p0, dtype=float),
                q0, self._attitude_speed_threshold, forward_axis
            )
            if self._aligner.needs_align(q0, q_align):
                self._log.info(
                    "[GuidanceExecutor] pre-aligning to initial tangent "
                    "direction before departure"
                )
                status = self._aligner.align_to(p0, q_align, is_cancel_requested)
                if status != STATUS_SUCCESS:
                    return status

            # Re-read TF: pre_align may have just rotated the vehicle to
            # q_align, but (p0, q0) above are from BEFORE that rotation.
            # Without this re-read, the tracker's q0 below seeds
            # face-travel's reference with the STALE pre-pre_align attitude
            # -- once translation speed crosses attitude_speed_threshold,
            # compute_q_des then has to visibly (rate-limited) catch up from
            # that stale value to q_align, commanding a large, functionally
            # unnecessary re-orientation away from where the vehicle
            # actually already is. Confirmed in sim: pre_align correctly
            # converges to q_align, yet the very first in-flight q_des
            # differs from the vehicle's actual (already-aligned) attitude
            # by ~127 degrees -- see docs/
            # guidance_attitude_saturation_investigation.md 7節.
            pose = self._tf.get_pose()
            if pose is None:
                self._log.warn(
                    "[GuidanceExecutor] no TF pose available after "
                    "pre_align, aborting"
                )
                return STATUS_ABORTED
            p0, q0, stamp = pose
            if not self._tf_pose_fresh(stamp):
                self._log.warn(
                    "[GuidanceExecutor] TF pose is stale after pre_align "
                    "(stamp not advancing for >%.1fs), aborting"
                    % self._tf_staleness_timeout
                )
                return STATUS_ABORTED

        try:
            tracker, traj = self._tracker_builder.build(
                p0, q0, p_target, via_waypoints, trajectory_tracking_mode,
                forward_axis or DEFAULT_CAMERA_FORWARD_AXIS["main"], face_travel,
                minco_via_half_width=minco_via_half_width,
                minco_attitude_resample_spacing_m=minco_attitude_resample_spacing_m,
                minco_wrench_safety_margin=minco_wrench_safety_margin,
                minco_freetime=minco_freetime,
                minco_local_replan_period=minco_local_replan_period,
                minco_planning_horizon_m=minco_planning_horizon_m,
                minco_replan_face_travel=minco_replan_face_travel,
                minco_local_max_vel=minco_local_max_vel,
                minco_async_replan=minco_async_replan,
                minco_obstacle_avoidance=minco_obstacle_avoidance,
                minco_local_piece_length_m=minco_local_piece_length_m,
                minco_obstacle_clearance_soft=minco_obstacle_clearance_soft,
            )
        except TrajectoryBuildError as exc:
            self._log.error("[GuidanceExecutor] %s, aborting" % exc)
            return STATUS_PLANNING_FAILED

        if self._speed_path_pub is not None:
            self._publish_speed_path_preview(traj)
        self._publish_local_speed_path_preview(tracker)

        status = self._run_trajectory(tracker, p_target, feedback_cb, is_cancel_requested)
        if status != STATUS_SUCCESS:
            return status
        p_arrival = _goal_position(tracker, p_target)

        if align_at_arrival:
            current = self._tf.get_pose()
            cur_quat = (
                current[1]
                if current is not None and self._tf_pose_fresh(current[2])
                else q_target
            )
            arrival_target_quat = self._resolve_arrival_target_quat(
                q_target, align_at_arrival_camera,
            )
            if self._aligner.needs_align(cur_quat, arrival_target_quat):
                self._log.info(
                    "[GuidanceExecutor] aligning to target attitude on arrival"
                )
                status = self._aligner.align_to(
                    p_arrival, arrival_target_quat, is_cancel_requested
                )
                if status != STATUS_SUCCESS:
                    return status

        return STATUS_SUCCESS

    def brake(self):
        """Stop along a JAXA ``stoppingProfile`` from the measured state, then
        hold its end pose (:class:`~sobits_intball2_gnc.guidance.utils.
        cancel_brake.CancelBrake`). Returns a ``STATUS_*``."""
        return STATUS_SUCCESS if self._brake.run() else STATUS_ABORTED

    def _resolve_arrival_target_quat(self, q_target, camera):
        """Return the quaternion ``align_at_arrival`` should converge to.

        ``camera == "main"`` (default) is just ``q_target``. Any other known
        camera name aligns so that camera's axis faces where the main
        camera's axis would have under ``q_target`` (:func:`compute_
        camera_relative_quat`); falls back to ``q_target`` with a warning
        if the camera name is unknown.
        """
        main_axis = self._camera_forward_axis.get("main",
                                                    DEFAULT_CAMERA_FORWARD_AXIS["main"])
        forward_axis = self._camera_forward_axis.get(camera)
        if forward_axis is None:
            self._log.warn(
                "[GuidanceExecutor] unknown align_at_arrival_camera=%r, "
                "falling back to 'main'" % camera
            )
            return q_target
        return compute_camera_relative_quat(q_target, main_axis, forward_axis)

    def _publish_local_speed_path_preview(self, tracker):
        local_traj = getattr(tracker, "local_trajectory", None)
        if self._local_speed_path_pub is None or local_traj is None:
            return
        self._publish_speed_path_preview(local_traj, self._local_speed_path_pub)

    def _publish_speed_path_preview(self, traj, publisher=None):
        """Publish ``traj``'s current shape to ``self._speed_path_pub``, for
        RViz-only visualization -- called once at goal start, and again on
        every re-plan (see ``_run_trajectory``).

        ``ToppraTrajectory``/``MincoTrajectory.sample()`` is a pure lookup
        (no per-call mutable state), so sampling ``traj`` out of order here
        is safe.
        """
        n = max(2, self._path_preview_points)
        duration = traj.global_total_duration
        samples = [traj.sample(duration * i / (n - 1)) for i in range(n)]
        (publisher or self._speed_path_pub).publish(
            [(p, np.linalg.norm(v)) for p, v, _a, _q in samples]
        )

    def _run_trajectory(self, tracker, p_target, feedback_cb, is_cancel_requested):
        t_start = self._clock_seconds()
        # Set once, the tick the planned duration first elapses -- kept
        # independent of tracker.total_duration itself so the replanning
        # tracker's repeatedly-recomputed total_duration (docs/
        # guidance_realtime_replanning_design.md 6-1) can't push this
        # deadline back out; only the position-convergence check below reads
        # live TF.
        pos_timeout_deadline = None
        # Time [s, sim clock] the TF position error first entered
        # align_pos_tolerance_m on this unbroken streak; None while out of
        # tolerance or no TF pose available. Same settle-time reasoning as
        # AttitudeAligner.align_to's in_tolerance_since (a single
        # in-tolerance sample isn't enough evidence of settling).
        in_pos_tolerance_since = None
        # Whether this tracker's fallback latch has already been logged
        # ("[C] Controller内部値の可観測性強化" task):
        # last_fallback_reason stays populated after the tick it trips on
        # (module docstring), so without this guard the same event would
        # otherwise appear to still be "happening" every remaining tick.
        fallback_logged = False
        while True:
            if is_cancel_requested():
                return STATUS_CANCELED
            elapsed = self._clock_seconds() - t_start
            sample_t = min(elapsed, tracker.total_duration)
            p, v, a, q = tracker.sample(sample_t)
            omega_des, alpha_des = tracker.last_body_angular
            self._setpoint_pub.publish(p, v, a, q, omega_des, alpha_des)
            if getattr(tracker, "last_replan_occurred", False):
                self._log.info(
                    "[GuidanceExecutor] replanning: re-planned trajectory at "
                    "t=%.2fs (solve=%.3fs, lag=%.3fs)"
                    % (sample_t, getattr(tracker, "last_replan_solve_seconds", None)
                       or float("nan"),
                       getattr(tracker, "last_replan_lag_seconds", None) or 0.0)
                )
                if self._speed_path_pub is not None:
                    self._publish_speed_path_preview(tracker.trajectory)
                self._publish_local_speed_path_preview(tracker)
            if not fallback_logged and getattr(
                tracker, "last_fallback_reason", None
            ) is not None:
                fallback_logged = True
                self._log.info(
                    "[GuidanceExecutor] replanning: stopped re-planning for "
                    "the rest of this goal (reason=%s) at t=%.2fs"
                    % (tracker.last_fallback_reason, sample_t)
                )

            time_to_go = max(0.0, tracker.total_duration - elapsed)
            p_goal = _goal_position(tracker, p_target)
            p_to_go = (p_goal - np.asarray(p)).tolist()
            feedback_cb(time_to_go, p_to_go, q.tolist())

            if elapsed >= tracker.total_duration:
                now = self._clock_seconds()
                if pos_timeout_deadline is None:
                    pos_timeout_deadline = now + self._align_pos_timeout
                pose = self._tf.get_pose()
                if pose is not None and self._tf_pose_fresh(pose[2]):
                    cur_pos, _quat, _stamp = pose
                    pos_error = np.linalg.norm(p_goal - np.asarray(cur_pos, dtype=float))
                    if pos_error <= self._align_pos_tolerance_m:
                        if in_pos_tolerance_since is None:
                            in_pos_tolerance_since = now
                        elif now - in_pos_tolerance_since >= self._align_pos_settle_time:
                            return STATUS_SUCCESS
                    else:
                        in_pos_tolerance_since = None
                if now >= pos_timeout_deadline:
                    self._log.warn(
                        "[GuidanceExecutor] position did not converge within "
                        "%.1fs of the planned duration elapsing -- "
                        "proceeding anyway" % self._align_pos_timeout
                    )
                    return STATUS_SUCCESS
            self._spin(self._dt)
