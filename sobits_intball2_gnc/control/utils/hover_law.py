#!/usr/bin/env python3
"""IMU-only hover control law (ROS-agnostic, testable).

Damps the angular rate (gyro) toward zero and opposes the linear-acceleration
disturbance (accelerometer, with an EMA bias estimate removed). See
:mod:`sobits_intball2_gnc.control.utils.hover_controller` for how this is
combined with the TF-based :class:`PoseCorrector` in ``tf_imu`` mode.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import deadband

DEFAULT_HOVER = {
    "control_rate": 50.0,
    "kd_w": [0.02, 0.02, 0.02],
    "kp_a": [0.5, 0.5, 0.5],
    "deadband_w": 0.01,
    "deadband_a": 0.02,
    "acc_bias_alpha": 0.01,
    "max_force": 0.1,
    "max_torque": 0.02,
    "mode": "tf_imu",
}


class HoverLaw:
    """IMU-only hover control law (ROS-agnostic, testable)."""

    def __init__(
        self,
        kd_w=DEFAULT_HOVER["kd_w"],
        kp_a=DEFAULT_HOVER["kp_a"],
        deadband_w=DEFAULT_HOVER["deadband_w"],
        deadband_a=DEFAULT_HOVER["deadband_a"],
        acc_bias_alpha=DEFAULT_HOVER["acc_bias_alpha"],
        max_force=DEFAULT_HOVER["max_force"],
        max_torque=DEFAULT_HOVER["max_torque"],
    ) -> None:
        self.kd_w = np.asarray(kd_w, dtype=float)
        self.kp_a = np.asarray(kp_a, dtype=float)
        self.deadband_w = float(deadband_w)
        self.deadband_a = float(deadband_a)
        self.acc_bias_alpha = float(acc_bias_alpha)
        self.max_force = float(max_force)
        self.max_torque = float(max_torque)
        self._acc_bias = np.zeros(3)

    def compute(self, gyro, acc, feedforward_force=None, attitude_ref=None):
        """Return (force, torque) lists for the given IMU sample.

        ``gyro`` / ``acc`` are 3-element body-frame iterables. ``feedforward_force``
        (optional) is summed with the hover corrective force before clamping, so
        free-path motion can command translation on top of hover. ``attitude_ref``
        is reserved for a future pose reference and is currently unused.
        """
        w = np.asarray(gyro, dtype=float)
        a = np.asarray(acc, dtype=float)

        # Torque: damp angular rate toward zero (deadband to ignore noise).
        w_db = deadband(w, self.deadband_w)
        torque = -self.kd_w * w_db

        # Force: oppose acceleration disturbance after removing EMA bias.
        self._acc_bias = (
            (1.0 - self.acc_bias_alpha) * self._acc_bias
            + self.acc_bias_alpha * a
        )
        a_res = deadband(a - self._acc_bias, self.deadband_a)
        force = -self.kp_a * a_res

        # Feed-forward translation hook (for future free-path motion).
        if feedforward_force is not None:
            force = force + np.asarray(feedforward_force, dtype=float)

        force = np.clip(force, -self.max_force, self.max_force)
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return force.tolist(), torque.tolist()
