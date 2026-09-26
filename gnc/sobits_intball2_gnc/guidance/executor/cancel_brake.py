#!/usr/bin/env python3
"""Brake after a move_to cancel (ROS-agnostic), split out of ``GuidanceExecutor``.

Stops along a JAXA ``stoppingProfile`` (``common/utils/stopping_profile.py``) from
the measured state, then holds its end pose (JAXA ``Ctl::cancelTarget``, docs/
archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md).
The same profile, started from a reference state instead, is replan_minco's
emergency stop (:meth:`CancelBrake.profile_from`).
"""
import numpy as np

from sobits_intball2_gnc.common.utils.stopping_profile import StoppingProfile
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle


class CancelBrake:
    """Args mirror ``GuidanceExecutor``'s of the same names (``stopping_*`` ->
    ``tolerance_pos``/``tolerance_att``/``duration_goal``/``wait_cancel``);
    ``tf_fresh_fn`` is its TF-liveness check and ``dt`` its publish period."""

    def __init__(self, tf_client, tf_fresh_fn, setpoint_publisher, checkpoint_publisher,
                 clock_seconds_fn, spin_fn, logger, dt, allocator, mass, inertia,
                 stopping_profile_kwargs, tolerance_pos, tolerance_att, duration_goal,
                 wait_cancel, velocity_fn=None, angular_velocity_fn=None):
        self._tf = tf_client
        self._tf_fresh_fn = tf_fresh_fn
        self._setpoint_pub = setpoint_publisher
        self._checkpoint_pub = checkpoint_publisher
        self._clock_seconds = clock_seconds_fn
        self._spin = spin_fn
        self._log = logger
        self._dt = dt
        self._allocator = allocator
        self._mass = mass
        self._inertia = inertia
        self._stopping_profile_kwargs = dict(stopping_profile_kwargs or {})
        self._tolerance_pos = float(tolerance_pos)
        self._tolerance_att = float(tolerance_att)
        self._duration_goal = float(duration_goal)
        self._wait_cancel = float(wait_cancel)
        self._velocity_fn = velocity_fn
        self._angular_velocity_fn = angular_velocity_fn

    @property
    def available(self):
        return self._allocator is not None and self._mass is not None and self._inertia is not None

    def profile_from(self, p, v, q, omega):
        return StoppingProfile(p, v, q, omega, self._allocator, self._mass, self._inertia,
                               **self._stopping_profile_kwargs)

    def run(self):
        """Brake from the measured state and hold the stop point. Returns whether it
        settled at the stop point (``False`` also when it could not start).

        The hold/arrival point is the profile's own end pose, not JAXA's
        ``r1``: ``r1`` omits the ``v0 * t_rot`` coast while rotation stops.
        """
        if not self.available:
            self._log.warn("[CancelBrake] brake: no allocator/mass/inertia, "
                           "leaving the stop to control's hold fallback")
            return False
        pose = self._tf.get_pose()
        if pose is None or not self._tf_fresh_fn(pose[2]):
            self._log.warn("[CancelBrake] brake: no fresh TF pose, "
                           "leaving the stop to control's hold fallback")
            return False
        pos, quat, _stamp = pose
        estimate = self._velocity_fn() if self._velocity_fn is not None else None
        v0 = np.zeros(3) if estimate is None else np.asarray(estimate.vel, dtype=float)
        gyro = (self._angular_velocity_fn() if self._angular_velocity_fn is not None
                else None)
        w0 = np.zeros(3) if gyro is None else np.asarray(gyro, dtype=float)
        if estimate is None or gyro is None:
            self._log.warn("[CancelBrake] brake: velocity (%s) / gyro (%s) "
                           "unavailable, treated as zero"
                           % (estimate is not None, gyro is not None))

        profile = self.profile_from(pos, v0, quat, w0)
        p_end, _, _, q_end, _, _ = profile.sample(profile.duration)
        self._log.info(
            "[CancelBrake] brake: |v0|=%.3fm/s |w0|=%.3frad/s a_max=%.4fm/s^2 "
            "stop_dist=%.2fm duration=%.1fs"
            % (np.linalg.norm(v0), np.linalg.norm(w0), profile.a_max,
               np.linalg.norm(np.asarray(p_end) - np.asarray(pos)), profile.duration)
        )

        t_start = self._clock_seconds()
        stay_since = None
        while True:
            now = self._clock_seconds()
            elapsed = now - t_start
            self._setpoint_pub.publish(*profile.sample(elapsed))
            current = self._tf.get_pose()
            if current is not None and self._tf_fresh_fn(current[2]):
                near = (
                    np.linalg.norm(np.asarray(current[0]) - p_end) < self._tolerance_pos
                    and geodesic_angle(current[1], q_end) < self._tolerance_att
                )
                stay_since = (stay_since if stay_since is not None else now) if near else None
            # JAXA's controller keeps tracking the profile after cancelTarget's
            # wait loop exits, so never hand over to the hold mid-profile.
            if (elapsed >= profile.duration and stay_since is not None
                    and now - stay_since >= self._duration_goal):
                settled = True
                break
            if elapsed >= profile.duration + self._wait_cancel:
                self._log.warn("[CancelBrake] brake: not settled within "
                               "duration+%.1fs, holding the stop point anyway"
                               % self._wait_cancel)
                settled = False
                break
            self._spin(self._dt)
        self._checkpoint_pub.publish(p_end, q_end)
        self._log.info("[CancelBrake] brake: holding stop point after %.1fs"
                       % (self._clock_seconds() - t_start))
        return settled
