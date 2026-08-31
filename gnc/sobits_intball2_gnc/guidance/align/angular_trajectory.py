#!/usr/bin/env python3
"""Rest-to-rest trapezoidal (or triangular) angular-velocity profile.

Used by ``AttitudeAligner.align_to()`` to feed the checkpoint-driven align
loop a smoothly moving intermediate target instead of a single step input,
removing the composite-axis overshoot documented in
``docs/2026-08-27_composite_axis_overshoot_summary_and_plan.md``. Ported
unchanged from ``test/manual/experiment_align_slerp_trapezoid.py`` (n=1/n=3
sim-verified across 30-180 deg offsets, see
``docs/2026-08-27_align_slerp_trapezoid_next_steps.md``).

No ROS or numpy state is held here -- these are plain-value functions only.
"""
import numpy as np


def trapezoid_duration(theta_total, v_cap, a_max):
    """Rest-to-rest trapezoidal (or triangular, if too short) duration [s]
    for covering angle ``theta_total`` [rad] with cruise rate ``v_cap``
    [rad/s] and acceleration ``a_max`` [rad/s^2]."""
    if theta_total <= 0.0:
        return 0.0
    d_accel = v_cap * v_cap / (2.0 * a_max)
    if theta_total >= 2.0 * d_accel:
        t_accel = v_cap / a_max
        return 2.0 * t_accel + (theta_total - 2.0 * d_accel) / v_cap
    v_peak = np.sqrt(a_max * theta_total)
    return 2.0 * v_peak / a_max


def trapezoid_fraction(t, theta_total, v_cap, a_max, duration):
    """Progress fraction u(t) in [0, 1] along the trapezoid profile."""
    if theta_total <= 0.0 or duration <= 0.0:
        return 1.0
    t = float(np.clip(t, 0.0, duration))
    d_accel = v_cap * v_cap / (2.0 * a_max)
    if theta_total >= 2.0 * d_accel:
        t_accel = v_cap / a_max
        t_decel_start = duration - t_accel
        if t <= t_accel:
            s = 0.5 * a_max * t * t
        elif t <= t_decel_start:
            s = d_accel + v_cap * (t - t_accel)
        else:
            s = theta_total - 0.5 * a_max * (duration - t) ** 2
    else:
        t_accel = duration / 2.0
        if t <= t_accel:
            s = 0.5 * a_max * t * t
        else:
            s = theta_total - 0.5 * a_max * (duration - t) ** 2
    return float(np.clip(s / theta_total, 0.0, 1.0))
