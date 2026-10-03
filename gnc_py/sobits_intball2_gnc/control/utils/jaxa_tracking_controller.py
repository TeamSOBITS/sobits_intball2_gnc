#!/usr/bin/env python3
"""JAXA-controller tracking/hold logic for ``jaxa_control_node`` (ROS-agnostic).

Drives the ported JAXA position/attitude controllers (``sobits_intball2_gnc_cpp``)
from TF pose, IMU gyro, Guidance's trajectory setpoint and the checkpoint hold
target -- the same inputs our ``HoverController`` uses, so Guidance works
unchanged. Design: ``docs/jaxa_controller_port_plan.md`` 7b.

One control computation per new TF pose (JAXA computes once per navigation
message); between poses the previous command is returned.
"""
import numpy as np

ZERO3 = [0.0, 0.0, 0.0]
# Our velocity EMA's vel_filter_alpha is defined at this period.
EMA_REFERENCE_PERIOD_S = 0.02


def ema_alpha(alpha_ref, dt, reference_period=EMA_REFERENCE_PERIOD_S):
    """Alpha giving the same decay per ``reference_period`` as ``alpha_ref`` at step ``dt``."""
    return 1.0 - (1.0 - alpha_ref) ** (dt / reference_period)


class JaxaTrackingController:
    """See module docstring.

    Args:
        position_controller, attitude_controller: ``JaxaPositionController`` /
            ``JaxaAttitudeController`` (from ``jaxa_control_params``).
        vel_filter_alpha: EMA weight defined at 50 Hz (``trajectory_controller.vel_filter_alpha``).
        trajectory_timeout: setpoint older than this [s] counts as ended.
        tf_timeout: TF pose older than this [s] (TF stamp vs. ``t``) gives a zero wrench.
    """

    def __init__(self, position_controller, attitude_controller, vel_filter_alpha,
                 trajectory_timeout, tf_timeout):
        self._pos = position_controller
        self._att = attitude_controller
        self._alpha_ref = float(vel_filter_alpha)
        self._trajectory_timeout = float(trajectory_timeout)
        self._tf_timeout = float(tf_timeout)
        self._hold = None
        self._hold_version = 0
        self._version_at_trajectory_start = None
        self._was_trajectory = False
        self._last_pos = None
        self._last_stamp = None
        self._vel = np.zeros(3)
        self._force = list(ZERO3)
        self._torque = list(ZERO3)
        self.status = "waiting"
        # True when the last step() computed a new command (new TF pose or a stop).
        self.updated = False

    @property
    def trajectory_active(self):
        return self._was_trajectory

    @property
    def velocity(self):
        return self._vel.tolist()

    def set_checkpoints(self, poses):
        """``[(pos, quat), ...]``: hold the first; empty -> hold the next pose seen."""
        self._hold = (list(poses[0][0]), list(poses[0][1])) if poses else None
        self._hold_version += 1
        self._pos.reset()

    def release(self):
        """Forget all tracking state when control resumes after being cut off (free drift).

        The next ``step`` holds the pose it first sees, with a fresh integrator and velocity
        estimate, instead of pulling back toward a target left from before the cut-off.
        """
        self._pos.reset()
        self._hold = None
        self._hold_version += 1
        self._was_trajectory = False
        self._last_pos = self._last_stamp = None
        self._vel = np.zeros(3)
        self._force, self._torque = list(ZERO3), list(ZERO3)
        self.updated = False

    def step(self, t, pose, gyro, setpoint, setpoint_t):
        """Returns ``(force_body, torque_body)``.

        ``pose`` is ``(pos, quat, stamp)`` or None; ``setpoint`` has ``p_des``/
        ``v_des``/``a_des``/``q_des``/``omega_des`` (``q_des`` body frame for omega)
        or is None; ``setpoint_t`` is when it arrived on the ``t`` clock.
        """
        if pose is None or gyro is None or t - pose[2] > self._tf_timeout:
            self.status = "stopped (no TF)" if pose is None else (
                "stopped (no IMU)" if gyro is None else "stopped (TF stale)")
            self._last_pos = None
            self._vel = np.zeros(3)
            self.updated = self._force != ZERO3 or self._torque != ZERO3
            self._force, self._torque = list(ZERO3), list(ZERO3)
            return self._force, self._torque
        pos, quat, stamp = np.asarray(pose[0], dtype=float), list(pose[1]), float(pose[2])
        if self._last_stamp is not None and stamp - self._last_stamp <= 1e-9 and self._last_pos is not None:
            self.updated = False
            return self._force, self._torque
        if self._last_pos is not None:
            dt = stamp - self._last_stamp
            a = ema_alpha(self._alpha_ref, dt)
            self._vel = a * (pos - self._last_pos) / dt + (1.0 - a) * self._vel
        self._last_pos, self._last_stamp = pos, stamp

        live = setpoint is not None and setpoint_t is not None and t - setpoint_t <= self._trajectory_timeout
        if live and not self._was_trajectory:
            self._pos.reset()
            self._version_at_trajectory_start = self._hold_version
        if not live and self._was_trajectory:
            # Same fallback as HoverController: hold where we are, unless a
            # checkpoint arrived during the trajectory (e.g. align at arrival).
            if self._hold_version == self._version_at_trajectory_start:
                self._hold = (pos.tolist(), quat)
            self._pos.reset()
        self._was_trajectory = live
        if self._hold is None:
            self._hold = (pos.tolist(), quat)

        if live:
            r_ref, v_ref, a_ref = setpoint["p_des"], setpoint["v_des"], setpoint["a_des"]
            q_ref, w_ref = setpoint["q_des"] or quat, setpoint["omega_des"] or ZERO3
            self.status = "tracking"
        else:
            (r_ref, q_ref), v_ref, a_ref, w_ref = self._hold, ZERO3, ZERO3, ZERO3
            self.status = "holding"
        self._force = self._pos.force_command(stamp, pos.tolist(), self._vel.tolist(), quat,
                                              list(r_ref), list(v_ref), list(a_ref))
        self._torque = self._att.torque_command(quat, list(gyro), list(q_ref), list(w_ref))
        self.updated = True
        return self._force, self._torque
