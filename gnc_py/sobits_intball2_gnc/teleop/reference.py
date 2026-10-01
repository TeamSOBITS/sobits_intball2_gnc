#!/usr/bin/env python3
"""Keyboard-driven moving reference for teleoperation (ROS-free).

Held keys give a body-frame velocity command, slew-limited and shaped to the wrench envelope,
integrated into a reference pose ``(p, v, a, q, w)`` for the trajectory controller. The reference
is independent of the vehicle, so it stalls (decelerates to rest and waits) while the tracking
error exceeds a limit rather than running away. Limits verified in a closed-loop
simulation with the JAXA controller (docs/teleop_keyboard_design.md 8.2/8.3).

Conventions: quaternions ``[x, y, z, w]``; ``p``/``v``/``a`` in the reference frame; ``w`` in the
body frame of ``q`` (same as ``MultiDOFJointTrajectory.velocities.angular``).
"""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_exp, quat_mul, quat_rotate

Setpoint = namedtuple("Setpoint", "p v a q w")


@dataclass(frozen=True)
class TeleopLimits:
    """Speed/acceleration caps (body frame) and the tracking-error limits that stall the reference."""

    vmax: float            # translational speed cap [m/s]
    wmax: float            # angular speed cap [rad/s]
    acc: tuple             # per-axis translational acceleration cap [m/s^2]
    alpha: tuple           # per-axis angular acceleration cap [rad/s^2]
    err_pos: float         # stall above this position error [m]
    err_att: float         # stall above this attitude error [rad]
    resume: float = 0.8    # resume once both errors fall below this fraction of the limits

    def __post_init__(self):
        if not (self.vmax > 0.0 and self.wmax > 0.0 and self.err_pos > 0.0 and self.err_att > 0.0):
            raise ValueError("TeleopLimits: vmax, wmax, err_pos and err_att must be positive")
        if len(self.acc) != 3 or len(self.alpha) != 3 or min(self.acc) <= 0.0 or min(self.alpha) <= 0.0:
            raise ValueError("TeleopLimits: acc and alpha need three positive entries")
        if not 0.0 < self.resume < 1.0:
            raise ValueError("TeleopLimits: resume must be in (0, 1)")


def axis_maxima(envelope):
    """Per-axis maximum of the achievable wrench set ``F @ w <= g``: ``[Fx, Fy, Fz, Tx, Ty, Tz]``."""
    from scipy.optimize import linprog
    f, g = np.asarray(envelope[0], float), np.asarray(envelope[1], float)
    out = []
    for i in range(6):
        c = np.zeros(6)
        c[i] = -1.0
        res = linprog(c, A_ub=f, b_ub=g, bounds=[(None, None)] * 6)
        if not res.success:
            raise ValueError("axis_maxima: envelope is unbounded or empty along axis %d" % i)
        out.append(-res.fun)
    return np.array(out)


def make_limits(envelope, mass, inertia, vmax, wmax, acc_frac, alpha_frac, err_pos, err_att, resume=0.8):
    """Acceleration caps as fractions of what the fans can do on each axis (``max force / mass`` and
    ``max torque / inertia``), so the caps follow the vehicle model instead of hand-typed numbers."""
    maxima = axis_maxima(envelope)
    inertia = np.asarray(inertia, dtype=float)
    diag = np.full(3, float(inertia)) if inertia.ndim == 0 else np.diag(inertia)
    return TeleopLimits(vmax=vmax, wmax=wmax,
                        acc=tuple(acc_frac * maxima[:3] / float(mass)),
                        alpha=tuple(alpha_frac * maxima[3:] / diag),
                        err_pos=err_pos, err_att=err_att, resume=resume)


def error_limits(vmax, wmax):
    """Tracking-error limits that stall the reference, from the speed caps (docs/teleop_keyboard_design.md 8.3).

    Position: about twice the error seen at that speed, 20 mm at 0.05 m/s growing 15 mm per 0.05 m/s.
    Attitude: 5 deg up to 0.2 rad/s, then 3 deg more per 0.1 rad/s.
    """
    return 0.005 + 0.3 * vmax, float(np.radians(5.0 + 30.0 * max(wmax - 0.2, 0.0)))


def limits_for_levels(envelope, mass, inertia, vmax, acc_frac, wmax_per_vmax, alpha_per_acc, resume=0.8):
    """:class:`TeleopLimits` for one (speed, acceleration) setting. Angular speed and acceleration follow the
    translational ones by fixed ratios, and the error limits follow the speeds."""
    wmax = vmax * wmax_per_vmax
    err_pos, err_att = error_limits(vmax, wmax)
    return make_limits(envelope, mass, inertia, vmax, wmax, acc_frac, acc_frac * alpha_per_acc,
                       err_pos, err_att, resume)


class TeleopReference:
    """Integrates a slew-limited body velocity command into a reference setpoint.

    Args:
        limits: :class:`TeleopLimits`.
        mass: vehicle mass [kg].
        inertia: 3x3 inertia matrix [kg m^2] (a scalar is taken as isotropic).
        envelope: ``(F, g)`` with ``F @ wrench <= g`` the achievable body wrench set, or None for no
            envelope shaping.
        centripetal: add ``w x v`` to the feed-forward acceleration (needed on arcs).
    """

    def __init__(self, limits, mass, inertia, envelope=None, centripetal=True):
        self.limits = limits
        self._mass = float(mass)
        inertia = np.asarray(inertia, dtype=float)
        self._inertia = np.eye(3) * inertia if inertia.ndim == 0 else inertia
        self._env = None if envelope is None else (np.asarray(envelope[0], float), np.asarray(envelope[1], float))
        self._centripetal = centripetal
        self._acc = np.asarray(limits.acc, dtype=float)
        self._alpha = np.asarray(limits.alpha, dtype=float)
        self.reset([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    def set_limits(self, limits):
        """Swap the caps. Call only while at rest: a change mid-motion would jump the deceleration."""
        self.limits = limits
        self._acc = np.asarray(limits.acc, dtype=float)
        self._alpha = np.asarray(limits.alpha, dtype=float)

    def reset(self, p, q):
        """Start (or restart) at pose ``(p, q)`` at rest."""
        self._r = np.array(p, dtype=float)
        self._q = np.array(q, dtype=float)
        self._q = self._q / np.linalg.norm(self._q)
        self.vb = np.zeros(3)
        self.wb = np.zeros(3)
        self.stalled = False
        self.scaled = False
        self.pos_err = 0.0
        self.att_err = 0.0

    @property
    def p(self):
        return self._r.copy()

    @property
    def q(self):
        return self._q.copy()

    @property
    def at_rest(self):
        return bool(np.linalg.norm(self.vb) < 1e-6 and np.linalg.norm(self.wb) < 1e-6)

    def step(self, dt, key, p_meas, q_meas):
        """Advance by ``dt`` [s]. ``key`` is the six key values in [-1, 1] ``(vx, vy, vz, wx, wy, wz)``
        (body frame); ``p_meas``/``q_meas`` the measured pose. Returns the new :class:`Setpoint`."""
        lim = self.limits
        self.pos_err = float(np.linalg.norm(self._r - np.asarray(p_meas, dtype=float)))
        self.att_err = float(geodesic_angle(self._q, q_meas))
        if self.stalled:
            self.stalled = not (self.pos_err < lim.resume * lim.err_pos
                                and self.att_err < lim.resume * lim.err_att)
        else:
            self.stalled = self.pos_err > lim.err_pos or self.att_err > lim.err_att
        key = np.zeros(6) if self.stalled else np.clip(np.asarray(key, dtype=float), -1.0, 1.0)

        dv = np.clip(key[:3] * lim.vmax - self.vb, -self._acc * dt, self._acc * dt)
        dw = np.clip(key[3:] * lim.wmax - self.wb, -self._alpha * dt, self._alpha * dt)
        self.scaled = False
        if self._env is not None:
            wrench = np.concatenate([self._mass * (dv / dt + np.cross(self.wb, self.vb)),
                                     self._inertia @ (dw / dt)])
            over = self._env[0] @ wrench
            hot = over > self._env[1]
            if hot.any():
                # Same ratio on force and torque so an arc keeps its radius v/w.
                s = max(float(np.min(self._env[1][hot] / over[hot])), 0.0)
                dv, dw, self.scaled = dv * s, dw * s, True
        self.vb = self.vb + dv
        self.wb = self.wb + dw

        a_body = dv / dt + (np.cross(self.wb, self.vb) if self._centripetal else 0.0)
        self._q = quat_mul(self._q, quat_exp(self.wb * dt))
        self._q = self._q / np.linalg.norm(self._q)
        v_world = quat_rotate(self._q, self.vb)
        self._r = self._r + v_world * dt
        return Setpoint(self._r.copy(), v_world, quat_rotate(self._q, a_body), self._q.copy(), self.wb.copy())
