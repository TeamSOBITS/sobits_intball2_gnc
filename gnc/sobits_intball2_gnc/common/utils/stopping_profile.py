#!/usr/bin/env python3
"""Stopping (brake) profile ported from JAXA ``ctl_only`` (ROS-agnostic).

Faithful port of ``PosAttProfiler::stoppingProfile()`` / ``target()`` /
``posProfile()`` / ``attProfile()`` (``src/pos_att_profiler.cpp``) and
``ThrustAllocator::effectiveFmax()`` / ``effectiveTmax()`` / ``effective()``
(``src/thrust_allocator.cpp``). See
``docs/archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md`` for what was replaced and
why (SOBITS's own allocator instead of JAXA's Wp/Wm one, etc.).

Quaternions are ``[x, y, z, w]``, orientation maps body -> reference frame;
``w0`` / ``omega`` / ``alpha`` are body-frame.
"""
import numpy as np
from scipy.optimize import brentq

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_mul,
    quat_rotate,
)

# JAXA effective(): Solver1(P_TOL=1e-6, D_TOL=1e-6).brent(fdif, 0., 2.)
_BRENT_TOL = 1.0e-6
_BRENT_UPPER = 2.0


def _max_fan_thrust(allocator, force, torque):
    duties = allocator.allocate(force, torque)
    return max((d / allocator.kj) ** 2 for d in duties)


def _effective(allocator, force, torque, eta):
    if not 0.0 < eta < 1.0:
        # eta >= 1 has no root: the allocator clamps each fan at fj_max.
        raise ValueError("eta must be in (0, 1), got %r" % eta)
    force = np.asarray(force, dtype=float)
    torque = np.asarray(torque, dtype=float)
    fan_limit = eta * allocator.fj_max
    return brentq(
        lambda p: _max_fan_thrust(allocator, force * p, torque * p) - fan_limit,
        0.0, _BRENT_UPPER, xtol=_BRENT_TOL,
    )


def effective_fmax(allocator, dir_body, fmax, eta):
    """Largest force [N] along ``dir_body`` with every fan <= ``eta * fj_max``."""
    return fmax * _effective(allocator, np.asarray(dir_body) * fmax, np.zeros(3), eta)


def effective_tmax(allocator, axis_body, tmax, eta):
    """Largest torque [Nm] about ``axis_body`` with every fan <= ``eta * fj_max``."""
    return tmax * _effective(allocator, np.zeros(3), np.asarray(axis_body) * tmax, eta)


def _target(t, te, thr, r, a):
    """JAXA ``target()`` with ``tcb = tce = 0`` (deceleration branch only)."""
    if r < thr:
        return r, 0.0, 0.0
    if t < 0.0:
        return 0.0, 0.0, 0.0
    if t < te:
        tg = te - t
        vp = a * tg
        return r - vp * tg * 0.5, vp, -a
    return r, 0.0, 0.0


class StoppingProfile:
    """JAXA ``stoppingProfile()``: stop rotation first, then translation (ATT_POS).

    Args:
        r0, v0: measured position [m] / velocity [m/s], reference frame.
        q0: measured orientation.
        w0: measured angular velocity [rad/s], body frame.
        allocator: ``ThrustAllocator`` (the same configuration control uses).
        mass: [kg]. inertia: scalar (isotropic) or 3x3 [kg m^2].
        eta: fraction of per-fan max thrust usable (JAXA ``pos_profile.eta_max``).
        f_max, t_max: JAXA ``pos_profile.f_max`` / ``att_profile.t_max``.
        x_threshold, theta_threshold: JAXA ``x_threshold`` / ``theta_threshold``.
        eps_rm, eps_qm: JAXA ``eps_rm`` / ``eps_qm``.
        max_axis_force: optional per-axis body-frame force cap [N]. Not in JAXA:
            control clamps its output per axis (``hover_control.max_force``),
            and a profile demanding more than that is tracked late.
    """

    def __init__(self, r0, v0, q0, w0, allocator, mass, inertia, eta,
                 f_max=181.0032e-3, t_max=8.1904e-3,
                 x_threshold=0.05, theta_threshold=0.017453292519943,
                 eps_rm=0.001, eps_qm=1.0e-9, max_axis_force=None):
        self._r0 = np.array(r0, dtype=float)
        self._v0 = np.array(v0, dtype=float)
        self._q0 = np.array(q0, dtype=float) / np.linalg.norm(q0)
        w0 = np.array(w0, dtype=float)
        self._x_threshold = float(x_threshold)
        self._theta_threshold = float(theta_threshold)
        inertia_matrix = (
            np.eye(3) * float(inertia) if np.ndim(inertia) == 0
            else np.asarray(inertia, dtype=float)
        )

        w = np.linalg.norm(w0)
        self._axis = w0 / w if w > eps_qm else np.array([1.0, 0.0, 0.0])
        im = float(self._axis @ inertia_matrix @ self._axis)
        self._wdmax = effective_tmax(allocator, self._axis, t_max, eta) / im
        self._qtt = w / self._wdmax
        self._qm = 0.5 * self._wdmax * self._qtt ** 2
        self._q1 = quat_mul(self._q0, quat_exp(self._axis * self._qm))

        v = np.linalg.norm(self._v0)
        self._dh = self._v0 / v if v > eps_rm else np.array([1.0, 0.0, 0.0])
        db = quat_rotate(quat_conj(self._q1), self._dh)
        force_max = effective_fmax(allocator, db, f_max, eta)
        if max_axis_force is not None:
            force_max = min(force_max, float(max_axis_force) / np.max(np.abs(db)))
        self._amax = force_max / float(mass)
        self._xtt = v / self._amax
        self._xm = 0.5 * self._amax * self._xtt ** 2
        self._r1 = self._r0 + self._dh * self._xm

    @property
    def duration(self) -> float:
        """JAXA ``te() - t0`` for ATT_POS: ``xtt + qtt`` [s]."""
        return self._xtt + self._qtt

    @property
    def r1(self):
        return self._r1.copy()

    @property
    def q1(self):
        return self._q1.copy()

    @property
    def a_max(self) -> float:
        return self._amax

    @property
    def wd_max(self) -> float:
        return self._wdmax

    def sample(self, t):
        """Return ``(p, v, a, q, omega, alpha)`` at ``t`` [s] since profile start."""
        p, v, a = self._pos_profile(t)
        q, omega, alpha = self._att_profile(t)
        return p, v, a, q, omega, alpha

    def _pos_profile(self, t):
        r0 = self._r0
        if t <= self._qtt:
            return r0 + self._v0 * t, self._v0.copy(), np.zeros(3)
        r0 = r0 + self._v0 * self._qtt
        t -= self._qtt
        xp, vp, ap = _target(t, self._xtt, self._x_threshold, self._xm, self._amax)
        return r0 + self._dh * xp, self._dh * vp, self._dh * ap

    def _att_profile(self, t):
        qp, wp, ap = _target(t, self._qtt, self._theta_threshold, self._qm, self._wdmax)
        q = quat_mul(self._q0, quat_exp(self._axis * qp))
        return q, self._axis * wp, self._axis * ap
