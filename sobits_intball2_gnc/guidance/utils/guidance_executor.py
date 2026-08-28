#!/usr/bin/env python3
"""Guidance move-to-target executor (ROS-agnostic orchestration logic).

Implements the ``execute_fn`` body for
:class:`~sobits_intball2_gnc.guidance.ros.ctl_command_action_server.CtlCommandActionServer`
(``docs/guidance_node_implementation_plan.md``): current pose (TF) -> target
pose -> ``HeuristicSegmentTimeAllocator`` -> ``HermiteSplineTrajectoryGenerator``
-> ``Trajectory`` -> a sim-clock-paced publish loop onto
``/gnc/trajectory_setpoint``, folding in the pre-/post-alignment steps that
``test/manual/send_curve_via_naventry_to_*_facing_direction.py`` scripts have
so far done ad hoc per-script.

Like :class:`~sobits_intball2_gnc.control.utils.hover_controller.HoverController`,
this class does not subclass ``Node`` or import ``rclpy``: it is driven by
already-constructed ROS I/O wrappers (``tf_client``, ``setpoint_publisher``,
``checkpoint_publisher``) and two injected callables (``clock_seconds_fn``,
``spin_fn``) so it is directly unit-testable with fakes, following this
package's DI convention (see ``docs/architecture_guidelines.md``).

Only a ``(current_pose, [via_waypoint], p_target)`` trajectory is generated
per call -- no global path planning is invoked (``docs/
guidance_node_implementation_plan.md`` decision 1). ``via_waypoint`` is an
optional single interior relay point (``docs/
2026-08-25_guidance_waypoint_insertion_curve_verification.md``), passed
through this goal's ``execute()`` call as an already TF-resolved position --
resolving a further chain of several waypoints (a client streaming several
``CtlCommand`` goals in a row stops at each one, same granularity as
``PoseCorrector``'s existing checkpoint chaining) is still out of scope here.
"""
import numpy as np

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation import (
    hermite_spline_trajectory_generator as _hermite,
)
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, slerp
from sobits_intball2_gnc.guidance.utils import angular_trajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_trajectory_tracker import (
    ReplanningTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    compute_camera_relative_quat,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
)
from sobits_intball2_gnc.guidance.utils.trajectory import Trajectory

TRAJECTORY_TRACKING_MODES = frozenset({"static", "replanning"})

STATUS_SUCCESS = "success"
STATUS_ABORTED = "aborted"
STATUS_CANCELED = "canceled"

DEFAULT_CAMERA_FORWARD_AXIS = {
    "main": (1.0, 0.0, 0.0),
    "stereo": (0.0, 1.0, 0.0),
}

_geodesic_angle = geodesic_angle


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
            guidance_velocity_estimator_design.md``). Not yet consumed by
            this class -- accepted for a future task (feeding v_now into
            ``HeuristicSegmentTimeAllocator``).
    """

    def __init__(self, tf_client, setpoint_publisher, checkpoint_publisher,
                 clock_seconds_fn, spin_fn, logger,
                 target_speed=0.5, attitude_speed_threshold=0.02,
                 align_tolerance_deg=3.0, align_timeout=60.0,
                 align_settle_time=0.5, rate=50.0,
                 camera_forward_axis=None, speed_path_publisher=None,
                 path_preview_points=20, max_accel=None,
                 align_pos_tolerance_m=0.05, align_pos_settle_time=0.5,
                 align_pos_timeout=10.0, tf_staleness_timeout=1.0,
                 velocity_fn=None, max_angular_rate=None,
                 distance_fallback_m=0.3, replan_rate_hz=10.0,
                 align_angular_speed_deg=None, align_angular_accel_deg=None,
                 align_traj_publish_rate_hz=20.0,
                 wrench_envelope=None, mass=None, inertia=None):
        self._tf = tf_client
        self._setpoint_pub = setpoint_publisher
        self._checkpoint_pub = checkpoint_publisher
        self._clock_seconds = clock_seconds_fn
        self._spin = spin_fn
        self._log = logger
        self._target_speed = float(target_speed)
        # Vehicle's achievable acceleration [m/s^2], e.g.
        # trajectory_controller.max_force / mass -- see
        # HeuristicSegmentTimeAllocator's docstring for why this matters
        # (segment_time must be long enough for a from-rest-to-rest profile
        # at this acceleration, not just distance/target_speed).
        self._max_accel = None if max_accel is None else float(max_accel)
        self._attitude_speed_threshold = float(attitude_speed_threshold)
        self._align_tolerance_rad = np.radians(float(align_tolerance_deg))
        self._align_timeout = float(align_timeout)
        # Minimum time [s] the geodesic angle to hold_quat must stay
        # continuously <= align_tolerance_rad before _align_to declares
        # convergence -- a single in-tolerance TF sample is not enough
        # evidence of settling: an underdamped attitude correction swings
        # through the target and this window keeps a mid-swing pass from
        # being mistaken for arrival (observed for pre_align: it "converged"
        # while still oscillating, then translation started ~40 deg off
        # target again once the swing continued past it, see
        # docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md).
        self._align_settle_time = float(align_settle_time)
        # SLERP+trapezoid align ramp (docs/2026-08-27_align_slerp_trapezoid_
        # next_steps.md): both None (default) keeps _align_to's prior
        # single-checkpoint-then-poll behavior byte-for-byte -- this is
        # opt-in so existing tests that don't pass these two params are
        # unaffected. guidance.py wires real values from config so production
        # gets the ramp; only a test that explicitly sets both exercises it.
        self._align_angular_speed_rad = (
            None if align_angular_speed_deg is None
            else np.radians(float(align_angular_speed_deg))
        )
        self._align_angular_accel_rad = (
            None if align_angular_accel_deg is None
            else np.radians(float(align_angular_accel_deg))
        )
        self._align_traj_dt = 1.0 / float(align_traj_publish_rate_hz)
        # Position-error counterpart of the attitude align_tolerance_rad/
        # align_settle_time/align_timeout trio above, but kept as separate
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
        # Guidance-side TF velocity estimate (docs/
        # guidance_velocity_estimator_design.md), independent of
        # TrajectoryController's own estimator (one-way Guidance -> Control
        # data flow, can't be reused -- see docs/
        # guidance_realtime_replanning_design.md 3-3 節). `velocity_fn` is a
        # callable returning the latest VelocityEstimator.get() result,
        # fed by a low-rate timer in guidance.py running on a different
        # thread than this class's own execute()/`_run_trajectory` loop --
        # never call it from inside a tight/latency-sensitive path without
        # accounting for that. Accepted here but not yet consumed: this is
        # only the DI plumbing for the next task in docs/main_plan.md
        # (passing v_now into HeuristicSegmentTimeAllocator).
        self._velocity_fn = velocity_fn
        self._dt = 1.0 / float(rate)
        # q_des rate limit (docs/archive/achieved/
        # 2026-08-24_trajectory_state_carryover_design.md 3-4節): always
        # passed to every Trajectory this class builds (static or
        # re-planning) -- None (default) reproduces the prior unlimited
        # behavior, but the node wires a real value so re-planning's noisy
        # v0 (docs/guidance_realtime_replanning_design.md 6-4節) can't
        # translate into an unbounded single-tick q_des jump.
        self._max_angular_rate = (
            None if max_angular_rate is None else float(max_angular_rate)
        )
        # Re-planning fallback threshold/cadence (docs/archive/achieved/
        # 2026-08-24_replanning_distance_fallback_decision.md,
        # 2026-08-24_replan_rate_design.md) -- only consumed when
        # execute()'s trajectory_tracking_mode == "replanning".
        self._distance_fallback_m = float(distance_fallback_m)
        # Re-plan every Nth _run_trajectory tick (currently `rate`, e.g.
        # 50Hz) rather than a separate rclpy timer/thread -- counter-based,
        # per the replan-rate decision doc's rationale (no new cross-thread
        # state beyond what VelocityEstimator already introduces).
        self._replan_every_n_ticks = max(1, round(float(rate) / float(replan_rate_hz)))
        self._camera_forward_axis = dict(
            camera_forward_axis or DEFAULT_CAMERA_FORWARD_AXIS
        )
        # Optional: a SpeedPathPublisher (guidance/ros/speed_path_publisher.py)
        # for RViz-only visualization of the planned trajectory, colored by
        # speed -- has no bearing on control behavior, see that module's
        # docstring.
        self._speed_path_pub = speed_path_publisher
        self._path_preview_points = int(path_preview_points)
        # Real actuator-derived budget for the static/TOPP-RA path (docs/
        # 2026-08-28_constrained_trajectory_generation_research.md,
        # docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md
        # "追記（2026-08-28 その2）") -- None (any of the three) disables
        # TOPP-RA and falls back to the legacy Hermite/
        # HeuristicSegmentTimeAllocator static path, same convention as
        # max_accel/velocity_fn gating "replanning" mode above.
        # wrench_envelope is a static (F, g) half-space pair from
        # actuation_envelope.wrench_envelope_halfspaces -- passed straight
        # through to ToppraTrajectory, not touched here.
        self._wrench_envelope = wrench_envelope
        self._mass = None if mass is None else float(mass)
        self._inertia = None if inertia is None else float(inertia)

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
        if align_tolerance_deg is not None:
            self._align_tolerance_rad = np.radians(float(align_tolerance_deg))
        if align_timeout is not None:
            self._align_timeout = float(align_timeout)
        if align_settle_time is not None:
            self._align_settle_time = float(align_settle_time)
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
                align_at_arrival_camera="main", trajectory_tracking_mode="static",
                via_waypoint=None):
        """Run one move-to-target goal; returns a ``STATUS_*`` constant.

        ``via_waypoint``: an optional single interior relay point (already
        TF-resolved to a ``[x, y, z]`` position by the caller, e.g.
        ``guidance.py``'s ``guidance.via_waypoint`` param), inserted between
        ``p0`` and ``p_target`` (``docs/
        2026-08-25_guidance_waypoint_insertion_curve_verification.md``). Its
        own orientation is not used for anything -- only its position feeds
        the 3-waypoint trajectory and ``pre_align``'s initial facing
        direction. ``None`` (default) reproduces the prior 2-waypoint
        behavior exactly.

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

        ``trajectory_tracking_mode``: ``"static"`` (default) samples a single
        open-loop trajectory generated at goal start, unchanged from prior
        behavior. ``"replanning"`` continuously re-plans the translation leg
        from live TF (``docs/guidance_realtime_replanning_design.md``); it
        falls back to ``"static"`` with a warning if ``max_accel`` or
        ``velocity_fn`` was not configured (both required by
        :class:`~sobits_intball2_gnc.guidance.trajectory_tracking.
        replanning_trajectory_tracker.ReplanningTrajectoryTracker`), or if
        given an unrecognized value.
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
            # With via_waypoint, face the first leg's chord (p0 -> via) since
            # that's the direction of imminent travel -- unlike the 2-point
            # case this is only an approximation once via_waypoint's
            # interior tangent is nonzero (docs/
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
            pre_align_target = p_target if via_waypoint is None else via_waypoint
            q_align = compute_q_des(
                np.asarray(pre_align_target, dtype=float) - np.asarray(p0, dtype=float),
                q0, self._attitude_speed_threshold, forward_axis
            )
            if _geodesic_angle(q0, q_align) > self._align_tolerance_rad:
                self._log.info(
                    "[GuidanceExecutor] pre-aligning to initial tangent "
                    "direction before departure"
                )
                status = self._align_to(p0, q_align, is_cancel_requested)
                if status != STATUS_SUCCESS:
                    return status

            # Re-read TF: pre_align may have just rotated the vehicle to
            # q_align, but (p0, q0) above are from BEFORE that rotation.
            # Without this re-read, Trajectory's initial_q_des below seeds
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

        if via_waypoint is None:
            waypoints = [p0, p_target]
        else:
            waypoints = [p0, via_waypoint, p_target]
            self._warn_if_sharp_turn(p0, via_waypoint, p_target)

        mode = trajectory_tracking_mode
        if mode not in TRAJECTORY_TRACKING_MODES:
            self._log.warn(
                "[GuidanceExecutor] unknown trajectory_tracking_mode=%r, "
                "falling back to 'static'" % trajectory_tracking_mode
            )
            mode = "static"
        if mode == "replanning" and (self._max_accel is None or self._velocity_fn is None):
            self._log.warn(
                "[GuidanceExecutor] trajectory_tracking_mode='replanning' "
                "requires max_accel and velocity_fn to be configured -- "
                "falling back to 'static'"
            )
            mode = "static"

        # static mode tries the force/torque-aware TOPP-RA path first (docs/
        # 2026-08-28_constrained_trajectory_generation_research.md); "replanning"
        # never does (toppra.compute_trajectory's sd_start is a scalar
        # path-tangent speed, which cannot express v0's perpendicular-to-path
        # residual that ReplanningTrajectoryTracker's exact bound handles --
        # see that doc's "trajectory_tracking_mode="replanning"との統合" 節).
        traj = None
        toppra_ready = (
            mode == "static"
            and self._wrench_envelope is not None
            and self._mass is not None
            and self._inertia is not None
            and self._max_angular_rate is not None
        )
        if toppra_ready:
            resolved_forward_axis = forward_axis or DEFAULT_CAMERA_FORWARD_AXIS["main"]
            try:
                traj = ToppraTrajectory(
                    waypoints, q0,
                    max_vel=self._target_speed,
                    mass=self._mass,
                    inertia=self._inertia,
                    wrench_envelope=self._wrench_envelope,
                    max_angular_rate=self._max_angular_rate,
                    forward_axis=resolved_forward_axis,
                    face_travel=face_travel,
                )
            except TrajectoryInfeasibleError as exc:
                self._log.warn(
                    "[GuidanceExecutor] TOPP-RA time-parameterization "
                    "infeasible (%s) -- falling back to the Hermite static "
                    "path" % exc
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
                forward_axis=forward_axis or DEFAULT_CAMERA_FORWARD_AXIS["main"],
                initial_q_des=q0, face_travel=face_travel,
                max_angular_rate=self._max_angular_rate,
            )

        if mode == "replanning":
            tracker = ReplanningTrajectoryTracker(
                traj, p_target, pose_fn=self._tf.get_pose,
                tf_fresh_fn=self._tf_pose_fresh, velocity_fn=self._velocity_fn,
                target_speed=self._target_speed, max_accel=self._max_accel,
                distance_fallback_m=self._distance_fallback_m,
                replan_every_n_ticks=self._replan_every_n_ticks,
                via_waypoint=via_waypoint,
            )
        else:
            tracker = StaticTrajectoryTracker(traj)

        if self._speed_path_pub is not None:
            self._publish_speed_path_preview(traj)

        status = self._run_trajectory(tracker, p_target, feedback_cb, is_cancel_requested)
        if status != STATUS_SUCCESS:
            return status

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
            if _geodesic_angle(cur_quat, arrival_target_quat) > self._align_tolerance_rad:
                self._log.info(
                    "[GuidanceExecutor] aligning to target attitude on arrival"
                )
                status = self._align_to(
                    p_target, arrival_target_quat, is_cancel_requested
                )
                if status != STATUS_SUCCESS:
                    return status

        return STATUS_SUCCESS

    def _warn_if_sharp_turn(self, p0, via_waypoint, p_target) -> None:
        """Log a warning if the route bends more than 90 deg at ``via_waypoint``.

        Static-geometry check only, computed once from the goal's
        ``(p0, via_waypoint, p_target)`` -- not the live re-planned ``v0``
        (that's a separate, already-existing concern:
        ``HeuristicSegmentTimeAllocator``'s ``v_perp``/``T_min_perp``
        handling). A turn this sharp cannot actually be flown without
        slowing to a stop first (``docs/
        2026-08-25_guidance_waypoint_insertion_curve_verification.md``); that
        stop-and-turn behavior is not implemented, so this is advisory only.
        """
        leg1 = np.asarray(via_waypoint, dtype=float) - np.asarray(p0, dtype=float)
        leg2 = np.asarray(p_target, dtype=float) - np.asarray(via_waypoint, dtype=float)
        n1 = np.linalg.norm(leg1)
        n2 = np.linalg.norm(leg2)
        if n1 < 1e-9 or n2 < 1e-9:
            return
        cos_angle = np.clip(np.dot(leg1, leg2) / (n1 * n2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg > 90.0:
            self._log.warn(
                "[GuidanceExecutor] via_waypoint turn angle is %.1f deg "
                "(>90 deg) -- the vehicle cannot fly this corner without "
                "slowing to a stop first; no such behavior is implemented, "
                "proceeding anyway" % angle_deg
            )

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

    def _publish_speed_path_preview(self, traj):
        """Publish ``traj``'s current shape to ``self._speed_path_pub``, for
        RViz-only visualization -- called once at goal start, and again on
        every re-plan (see ``_run_trajectory``) so the preview tracks
        ``trajectory_tracking_mode="replanning"``'s continuously-updated
        trajectory instead of going stale after the first plan.

        For a legacy ``Trajectory``, samples a throwaway instance built from
        ``traj``'s current ``waypoints``/``segment_times``/``coeffs``, not
        ``traj`` itself: ``Trajectory.sample()`` is stateful (face-travel
        rate-limiting via ``_last_sample_t``/``_last_q_des``) and sampling it
        out of order here would corrupt that state before the real,
        monotonically-increasing sampling in ``_run_trajectory``. The
        preview never reads ``q_des`` (``SpeedPathPublisher.publish()`` only
        wants position/speed), so ``face_travel=False`` skips the attitude
        computation entirely rather than reproducing the goal's actual
        attitude-reference settings here.

        For a ``ToppraTrajectory``, ``sample()`` is a pure lookup into an
        already-computed time-parameterized trajectory (no per-call mutable
        state), so it's safe to sample ``traj`` directly here.
        """
        n = max(2, self._path_preview_points)
        if isinstance(traj, ToppraTrajectory):
            duration = traj.global_total_duration
            samples = [traj.sample(duration * i / (n - 1)) for i in range(n)]
        else:
            preview_traj = Trajectory(
                traj.waypoints, traj.segment_times, traj.coeffs, face_travel=False,
            )
            samples = [preview_traj.sample(traj.total_duration * i / (n - 1))
                       for i in range(n)]
        self._speed_path_pub.publish(
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
        # _align_to's in_tolerance_since (a single in-tolerance sample isn't
        # enough evidence of settling).
        in_pos_tolerance_since = None
        # Whether this tracker's fallback latch has already been logged
        # (docs/main_plan.md "[C] Controller内部値の可観測性強化"):
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
            self._setpoint_pub.publish(p, v, a, q)
            if getattr(tracker, "last_replan_occurred", False):
                self._log.info(
                    "[GuidanceExecutor] replanning: re-planned trajectory at "
                    "t=%.2fs" % sample_t
                )
                if self._speed_path_pub is not None:
                    self._publish_speed_path_preview(tracker.trajectory)
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
            p_to_go = (np.asarray(p_target, dtype=float) - np.asarray(p)).tolist()
            feedback_cb(time_to_go, p_to_go, q.tolist())

            if elapsed >= tracker.total_duration:
                now = self._clock_seconds()
                if pos_timeout_deadline is None:
                    pos_timeout_deadline = now + self._align_pos_timeout
                pose = self._tf.get_pose()
                if pose is not None and self._tf_pose_fresh(pose[2]):
                    cur_pos, _quat, _stamp = pose
                    pos_error = np.linalg.norm(
                        np.asarray(p_target, dtype=float)
                        - np.asarray(cur_pos, dtype=float)
                    )
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

    def _align_to(self, hold_pos, hold_quat, is_cancel_requested):
        """Ramp a SLERP+trapezoid checkpoint toward ``hold_quat`` (if the
        align ramp is configured), then poll TF for convergence.

        docs/2026-08-27_align_slerp_trapezoid_next_steps.md: instead of
        stepping the checkpoint straight to ``hold_quat``, publish a moving
        intermediate target along the current-attitude -> hold_quat SLERP
        arc, paced by a rest-to-rest trapezoidal angular-speed profile, so
        the attitude controller only ever chases a small instantaneous
        error instead of a large step input -- this is what removes the
        composite-axis overshoot (docs/2026-08-27_composite_axis_overshoot_
        summary_and_plan.md).

        Opt-in: if ``align_angular_speed_deg``/``align_angular_accel_deg``
        were not configured (both None), ``theta_total`` is forced to 0.0
        below and this degrades to exactly the prior single-checkpoint
        behavior.
        """
        if not self._checkpoint_pub.wait_for_subscriber(
            timeout_sec=5.0, spin_fn=self._spin
        ):
            self._log.warn(
                "[GuidanceExecutor] no /gnc/checkpoints subscriber matched "
                "after 5s -- publishing anyway, alignment may not converge"
            )

        ramp_enabled = (
            self._align_angular_speed_rad is not None
            and self._align_angular_accel_rad is not None
        )
        theta_total = 0.0
        q_from = None
        if ramp_enabled:
            pose = self._tf.get_pose()
            if pose is not None:
                q_from = pose[1]
                theta_total = _geodesic_angle(q_from, hold_quat)
            # pose is None (no TF): theta_total stays 0.0, i.e. skip the
            # ramp and fall straight through to the plain publish below --
            # execute()'s callers already checked TF freshness earlier in
            # the goal, so this is just a defensive fallback, not the
            # expected path.

        duration = (
            angular_trajectory.trapezoid_duration(
                theta_total, self._align_angular_speed_rad,
                self._align_angular_accel_rad,
            ) if theta_total > 0.0 else 0.0
        )
        if duration > 0.0:
            q_from_arr = np.asarray(q_from, dtype=float)
            q_to_arr = np.asarray(hold_quat, dtype=float)
            t0 = self._clock_seconds()
            while True:
                if is_cancel_requested():
                    return STATUS_CANCELED
                t = self._clock_seconds() - t0
                if t >= duration:
                    break
                u = angular_trajectory.trapezoid_fraction(
                    t, theta_total, self._align_angular_speed_rad,
                    self._align_angular_accel_rad, duration,
                )
                self._checkpoint_pub.publish(
                    hold_pos, slerp(q_from_arr, q_to_arr, u)
                )
                self._spin(self._align_traj_dt)

        self._checkpoint_pub.publish(hold_pos, hold_quat)

        deadline = self._clock_seconds() + self._align_timeout
        # Time [s, sim clock] the geodesic angle first entered tolerance on
        # this unbroken in-tolerance streak; None while out of tolerance.
        # Requiring align_settle_time of unbroken dwell (not just one
        # in-tolerance sample) keeps an underdamped correction's mid-swing
        # pass through the target from being mistaken for having arrived --
        # see docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md.
        in_tolerance_since = None
        while self._clock_seconds() < deadline:
            if is_cancel_requested():
                return STATUS_CANCELED
            pose = self._tf.get_pose()
            if pose is not None and self._tf_pose_fresh(pose[2]):
                _pos, quat, _stamp = pose
                now = self._clock_seconds()
                if _geodesic_angle(quat, hold_quat) <= self._align_tolerance_rad:
                    if in_tolerance_since is None:
                        in_tolerance_since = now
                    elif now - in_tolerance_since >= self._align_settle_time:
                        return STATUS_SUCCESS
                else:
                    in_tolerance_since = None
            self._spin(self._dt)

        self._log.warn(
            "[GuidanceExecutor] alignment did not converge within %.1fs -- "
            "proceeding anyway" % self._align_timeout
        )
        return STATUS_SUCCESS
