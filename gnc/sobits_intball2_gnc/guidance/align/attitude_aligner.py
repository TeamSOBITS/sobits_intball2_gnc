#!/usr/bin/env python3
"""Attitude-only alignment: ramp toward a target quaternion, then poll TF for
convergence (ROS-agnostic orchestration logic).

Extracted out of ``GuidanceExecutor`` (docs/
2026-08-29_guidance_dir_and_dead_code_survey.md): translation trajectory
generation/tracking and attitude alignment are independent concerns --
this class only ever receives a ``(hold_pos, hold_quat)`` pair to converge
on, with no knowledge of ``trajectory_tracking_mode`` or the TOPP-RA/Hermite
dispatch in ``GuidanceExecutor.execute()``. Used for both ``pre_align`` and
``align_at_arrival``.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, slerp
from sobits_intball2_gnc.guidance.align import angular_trajectory

# Mirrors guidance_executor.STATUS_SUCCESS/STATUS_CANCELED -- duplicated as
# plain string literals (not imported) to avoid a guidance_executor <->
# attitude_aligner circular import. Must stay in sync with that module's
# status vocabulary.
STATUS_SUCCESS = "success"
STATUS_CANCELED = "canceled"

_geodesic_angle = geodesic_angle


class AttitudeAligner:
    """Ramp-then-poll convergence to a held ``(hold_pos, hold_quat)``.

    Args:
        tf_client: current-pose source (``get_pose() -> (pos, quat, stamp)``
            or ``None``), same object ``GuidanceExecutor`` uses.
        checkpoint_pub: ``CheckpointPublisher``-shaped object
            (``publish(pos, quat)``, ``wait_for_subscriber(timeout_sec)``).
        spin_fn: callable ``spin_fn(seconds)``, see ``GuidanceExecutor``.
        clock_seconds_fn: callable returning the current sim-clock time [s].
        logger: object with ``.info``/``.warn``.
        tf_fresh_fn: callable ``tf_fresh_fn(stamp) -> bool``, e.g.
            ``GuidanceExecutor._tf_pose_fresh`` -- TF liveness is judged by
            whether the stamp advances, and that tracking state spans a
            whole goal (or the executor's lifetime), not just one align
            call, so it stays owned by ``GuidanceExecutor`` and is only
            injected here.
        dt: main loop period [s] (``GuidanceExecutor``'s own ``1/rate``) used
            to pace the convergence-poll loop -- deliberately not
            ``align_traj_publish_rate_hz``, which only paces the ramp below.
        align_tolerance_deg/align_timeout/align_settle_time: see
            ``config/gnc_params.yaml``'s ``guidance`` section.
        align_angular_speed_deg/align_angular_accel_deg: SLERP+trapezoid
            align ramp (docs/2026-08-27_align_slerp_trapezoid_next_steps.md).
            Both ``None`` (default) skips the ramp entirely -- a single
            checkpoint step straight to ``hold_quat``, then poll.
    """

    def __init__(self, tf_client, checkpoint_pub, spin_fn, clock_seconds_fn,
                 logger, tf_fresh_fn, dt,
                 align_tolerance_deg=3.0, align_timeout=60.0,
                 align_settle_time=0.5,
                 align_angular_speed_deg=None, align_angular_accel_deg=None,
                 align_traj_publish_rate_hz=20.0):
        self._tf = tf_client
        self._checkpoint_pub = checkpoint_pub
        self._spin = spin_fn
        self._clock_seconds = clock_seconds_fn
        self._log = logger
        self._tf_pose_fresh = tf_fresh_fn
        self._dt = float(dt)
        self._align_tolerance_rad = np.radians(float(align_tolerance_deg))
        self._align_timeout = float(align_timeout)
        # Minimum time [s] the geodesic angle to hold_quat must stay
        # continuously <= align_tolerance_rad before align_to declares
        # convergence -- a single in-tolerance TF sample is not enough
        # evidence of settling: an underdamped attitude correction swings
        # through the target and this window keeps a mid-swing pass from
        # being mistaken for arrival (see docs/archive/achieved/
        # 2026-08-21_pre_align_skipped_low_speed_bug.md).
        self._align_settle_time = float(align_settle_time)
        self._align_angular_speed_rad = (
            None if align_angular_speed_deg is None
            else np.radians(float(align_angular_speed_deg))
        )
        self._align_angular_accel_rad = (
            None if align_angular_accel_deg is None
            else np.radians(float(align_angular_accel_deg))
        )
        self._align_traj_dt = 1.0 / float(align_traj_publish_rate_hz)

    def set_gains(self, align_tolerance_deg=None, align_timeout=None,
                  align_settle_time=None) -> None:
        """Update the align convergence threshold/timeout in place (dynamic
        reconfiguration, docs/archive/achieved/
        2026-08-21_dynamic_parameter_classification.md category A)."""
        if align_tolerance_deg is not None:
            self._align_tolerance_rad = np.radians(float(align_tolerance_deg))
        if align_timeout is not None:
            self._align_timeout = float(align_timeout)
        if align_settle_time is not None:
            self._align_settle_time = float(align_settle_time)

    def needs_align(self, from_quat, to_quat) -> bool:
        """Whether ``from_quat`` is far enough from ``to_quat`` to warrant
        running :meth:`align_to` -- the single source of truth for
        ``align_tolerance_rad`` so callers never read it directly."""
        return _geodesic_angle(from_quat, to_quat) > self._align_tolerance_rad

    def align_to(self, hold_pos, hold_quat, is_cancel_requested):
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
                "[AttitudeAligner] no /gnc/checkpoints subscriber matched "
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
            # GuidanceExecutor.execute() already checked TF freshness
            # earlier in the goal, so this is just a defensive fallback,
            # not the expected path.

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
            "[AttitudeAligner] alignment did not converge within %.1fs -- "
            "proceeding anyway" % self._align_timeout
        )
        return STATUS_SUCCESS
