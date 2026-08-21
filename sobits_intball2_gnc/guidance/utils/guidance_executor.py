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

Only a single ``(current_pose, p_target)`` 2-waypoint trajectory is generated
per call -- no global path planning is invoked (``docs/
guidance_node_implementation_plan.md`` decision 1). Genuine multi-waypoint
smooth transit (a client streaming several ``CtlCommand`` goals in a row
stops at each one, same granularity as ``PoseCorrector``'s existing
checkpoint chaining) is out of scope here.
"""
import numpy as np

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation import (
    hermite_spline_trajectory_generator as _hermite,
)
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    compute_camera_relative_quat,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.trajectory import Trajectory

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
            align_timeout/rate: see ``config/gnc_params.yaml``'s
            ``guidance`` section.
        camera_forward_axis: ``{camera_name: [x, y, z]}`` body-frame forward
            axis per ``face_travel_camera`` option (default:
            ``DEFAULT_CAMERA_FORWARD_AXIS``, TF-measured 2026-08-20).
    """

    def __init__(self, tf_client, setpoint_publisher, checkpoint_publisher,
                 clock_seconds_fn, spin_fn, logger,
                 target_speed=0.5, attitude_speed_threshold=0.02,
                 align_tolerance_deg=3.0, align_timeout=60.0,
                 align_settle_time=0.5, rate=50.0,
                 camera_forward_axis=None, speed_path_publisher=None,
                 path_preview_points=20, max_accel=None):
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
        self._dt = 1.0 / float(rate)
        self._camera_forward_axis = dict(
            camera_forward_axis or DEFAULT_CAMERA_FORWARD_AXIS
        )
        # Optional: a SpeedPathPublisher (guidance/ros/speed_path_publisher.py)
        # for RViz-only visualization of the planned trajectory, colored by
        # speed -- has no bearing on control behavior, see that module's
        # docstring.
        self._speed_path_pub = speed_path_publisher
        self._path_preview_points = int(path_preview_points)

    def set_gains(self, align_tolerance_deg=None, align_timeout=None,
                  align_settle_time=None) -> None:
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

    def execute(self, p_target, q_target, feedback_cb, is_cancel_requested,
                face_travel=True, face_travel_camera="main",
                align_at_arrival=True, pre_align=True, look_at_target_frame="",
                align_at_arrival_camera="main"):
        """Run one move-to-target goal; returns a ``STATUS_*`` constant.

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
        """
        pose = self._tf.get_pose()
        if pose is None:
            self._log.warn("[GuidanceExecutor] no TF pose available, aborting")
            return STATUS_ABORTED
        p0, q0, _stamp = pose

        forward_axis = self._camera_forward_axis.get(face_travel_camera)
        if face_travel and forward_axis is None:
            self._log.warn(
                "[GuidanceExecutor] unknown face_travel_camera=%r, "
                "falling back to 'main'" % face_travel_camera
            )
            forward_axis = self._camera_forward_axis["main"]

        waypoints = [p0, p_target]
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
        )

        if self._speed_path_pub is not None:
            # Sample a throwaway Trajectory instance, not `traj` itself:
            # Trajectory.sample() is stateful (face-travel rate-limiting
            # via _last_sample_t/_last_q_des) and sampling it out of order
            # here would corrupt that state before _run_trajectory's real,
            # monotonically-increasing sampling begins below.
            preview_traj = Trajectory(
                waypoints, segment_times, coeffs,
                attitude_speed_threshold=self._attitude_speed_threshold,
                forward_axis=forward_axis or DEFAULT_CAMERA_FORWARD_AXIS["main"],
                initial_q_des=q0, face_travel=face_travel,
            )
            n = max(2, self._path_preview_points)
            samples = [preview_traj.sample(traj.total_duration * i / (n - 1))
                       for i in range(n)]
            self._speed_path_pub.publish(
                [(p, np.linalg.norm(v)) for p, v, _a, _q in samples]
            )

        if face_travel and pre_align:
            # Use the chord to the next waypoint, not a sampled trajectory
            # velocity: v(0) == 0 by construction (see
            # HermiteSplineTrajectoryGenerator), so an early-in-the-segment
            # velocity sample can land below attitude_speed_threshold and
            # get silently treated as "no direction change needed" by
            # compute_q_des's low-speed guard -- this previously made
            # pre_align skip entirely for a target straight behind the
            # vehicle (see
            # docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md).
            # waypoints[1] (not p_target) so this keeps meaning "the next
            # leg's direction" if/when multi-waypoint paths (e.g. a future
            # obstacle-avoiding planner) replace today's 2-point
            # ``[p0, p_target]`` list; today waypoints[1] is p_target, so
            # this is exactly the direction v_early would have pointed
            # anyway (HermiteSplineTrajectoryGenerator interpolates a
            # 2-waypoint, at-rest-at-both-ends trajectory as a straight
            # line, so every non-zero v(t) already points p0 -> p_target).
            #
            # Pass q0 (not None) so compute_q_des fills the pointing task's
            # free roll DOF to match the vehicle's current roll instead of
            # the shortest-arc convention's incidental value -- otherwise
            # pre_align chases a functionally unnecessary roll change (see
            # docs/2026-08-21_tf_correction_align_optimization.md 8節).
            q_align = compute_q_des(
                np.asarray(waypoints[1], dtype=float)
                - np.asarray(waypoints[0], dtype=float),
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

        status = self._run_trajectory(traj, p_target, feedback_cb, is_cancel_requested)
        if status != STATUS_SUCCESS:
            return status

        if align_at_arrival:
            current = self._tf.get_pose()
            cur_quat = current[1] if current is not None else q_target
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

    def _run_trajectory(self, traj, p_target, feedback_cb, is_cancel_requested):
        t_start = self._clock_seconds()
        while True:
            if is_cancel_requested():
                return STATUS_CANCELED
            elapsed = self._clock_seconds() - t_start
            sample_t = min(elapsed, traj.total_duration)
            p, v, a, q = traj.sample(sample_t)
            self._setpoint_pub.publish(p, v, a, q)

            time_to_go = max(0.0, traj.total_duration - elapsed)
            p_to_go = (np.asarray(p_target, dtype=float) - np.asarray(p)).tolist()
            feedback_cb(time_to_go, p_to_go, q.tolist())

            if elapsed >= traj.total_duration:
                return STATUS_SUCCESS
            self._spin(self._dt)

    def _align_to(self, hold_pos, hold_quat, is_cancel_requested):
        """Publish a single static checkpoint and poll TF for convergence."""
        if not self._checkpoint_pub.wait_for_subscriber(
            timeout_sec=5.0, spin_fn=self._spin
        ):
            self._log.warn(
                "[GuidanceExecutor] no /gnc/checkpoints subscriber matched "
                "after 5s -- publishing anyway, alignment may not converge"
            )
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
            if pose is not None:
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
