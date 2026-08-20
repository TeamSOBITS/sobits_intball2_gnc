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
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector
from sobits_intball2_gnc.guidance.utils.trajectory import Trajectory

STATUS_SUCCESS = "success"
STATUS_ABORTED = "aborted"
STATUS_CANCELED = "canceled"

DEFAULT_CAMERA_FORWARD_AXIS = {
    "main": (1.0, 0.0, 0.0),
    "stereo": (0.0, 1.0, 0.0),
}

# How far into the first segment to sample v_des for the pre-alignment
# target: small enough to be "the initial tangent direction", far enough
# from 0 that it isn't swamped by the trajectory's start-at-rest zero
# velocity (see Trajectory/HermiteSplineTrajectoryGenerator: v(0) == 0 by
# construction, so v(0) itself can't be used to pick a facing direction).
_EARLY_TAU_FRACTION = 0.05


def _geodesic_angle(q_a, q_b):
    """Angle [rad] between two orientations, robust to the double cover."""
    dot = np.clip(abs(np.dot(np.asarray(q_a, dtype=float),
                             np.asarray(q_b, dtype=float))), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


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
                 align_tolerance_deg=3.0, align_timeout=60.0, rate=50.0,
                 camera_forward_axis=None, path_publisher=None,
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
        self._dt = 1.0 / float(rate)
        self._camera_forward_axis = dict(
            camera_forward_axis or DEFAULT_CAMERA_FORWARD_AXIS
        )
        # Optional: a PathPublisher (guidance/ros/path_publisher.py) for
        # RViz-only visualization of the planned trajectory -- has no
        # bearing on control behavior, see that module's docstring.
        self._path_pub = path_publisher
        self._path_preview_points = int(path_preview_points)

    def execute(self, p_target, q_target, feedback_cb, is_cancel_requested,
                face_travel=True, face_travel_camera="main",
                align_at_arrival=True):
        """Run one move-to-target goal; returns a ``STATUS_*`` constant."""
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

        if self._path_pub is not None:
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
            self._path_pub.publish([(p, q) for p, _v, _a, q in samples])

        if face_travel:
            tau_early = min(
                _EARLY_TAU_FRACTION * segment_times[0], segment_times[0]
            )
            v_early = evaluate_vector(coeffs[0], tau_early, order=1)
            q_align = compute_q_des(
                v_early, None, self._attitude_speed_threshold, forward_axis
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
            if _geodesic_angle(cur_quat, q_target) > self._align_tolerance_rad:
                self._log.info(
                    "[GuidanceExecutor] aligning to target attitude on arrival"
                )
                status = self._align_to(p_target, q_target, is_cancel_requested)
                if status != STATUS_SUCCESS:
                    return status

        return STATUS_SUCCESS

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
        while self._clock_seconds() < deadline:
            if is_cancel_requested():
                return STATUS_CANCELED
            pose = self._tf.get_pose()
            if pose is not None:
                _pos, quat, _stamp = pose
                if _geodesic_angle(quat, hold_quat) <= self._align_tolerance_rad:
                    return STATUS_SUCCESS
            self._spin(self._dt)

        self._log.warn(
            "[GuidanceExecutor] alignment did not converge within %.1fs -- "
            "proceeding anyway" % self._align_timeout
        )
        return STATUS_SUCCESS
