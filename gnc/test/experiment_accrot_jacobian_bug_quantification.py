#!/usr/bin/env python3
"""Offline, sim-free quantification of the accRot bug in minco_solver.cpp.

Does NOT touch minco_solver.cpp. This is a standalone numerical check of
the claim in docs/issue.md ([G] "minco_solver.cppのaccRotが角加速度omega_dot
として扱われている"): the solver's rotation channel represents attitude as
a rotation vector r(t) (a MINCO_S3NU quintic-per-segment polynomial, same
machinery as the position channel), and computes the commanded torque as

    tau = INERTIA * r_ddot(t)                      (minco_solver.cpp:222/297)

treating the raw 2nd derivative of the rotation-vector parametrization as
the body angular acceleration omega_dot. That is only correct in the limit
r -> 0. The exact relation (R(t) = Expm(skew(r(t))), body angular velocity
omega defined by R^T @ Rdot = skew(omega)) is

    omega     = Jr(r) @ r_dot
    omega_dot = Jr(r) @ r_ddot + Jr_dot(r, r_dot) @ r_dot

with the SO(3) right Jacobian
    Jr(r) = I - (1-cos(theta))/theta^2 * [r]_x + (theta-sin(theta))/theta^3 * [r]_x^2
(theta = |r|; Zefran, Kumar & Croke 1995; Watterson, Smith & Kumar IROS 2016).

This script:
  1. Builds a synthetic multi-axis, large-angle rotation-vector trajectory
     r(t) (quintic, matching the polynomial degree MINCO_S3NU uses).
  2. Computes r_ddot(t) analytically -- exactly what minco_solver.cpp uses
     as its omega_dot proxy today.
  3. Computes the TRUE omega(t) two independent ways and cross-checks them:
       (a) analytic Jr(r) @ r_dot
       (b) finite-difference Log(R(t)^T @ R(t+dt)) / dt  (Rodrigues-exact,
           no small-angle assumption)
     Agreement to float precision validates the Jr formula against a
     completely independent (matrix-log) computation.
  4. Computes the TRUE omega_dot(t) by central finite difference of the
     validated omega(t), and compares INERTIA*r_ddot (buggy) against
     INERTIA*omega_dot (correct) across small vs. large, single- vs.
     multi-axis rotation to show the error is negligible near r=0 and
     grows with angle/axis-mixing, consistent with issue.md's claim.

Usage: python3 gnc/test/experiment_accrot_jacobian_bug_quantification.py
"""
import numpy as np
from scipy.spatial.transform import Rotation

INERTIA = 0.0136  # isotropic, matches minco_solver.cpp:29


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def right_jacobian(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    K = skew(r)
    a = (1 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) - a * K + b * (K @ K)


def quintic_rotvec_trajectory(coeffs, t):
    """r(t), r_dot(t), r_ddot(t) for a per-axis quintic r_i(t) = sum_k c_ik t^k."""
    powers = np.array([t**k for k in range(6)])
    dpowers = np.array([0.0] + [k * t**(k - 1) for k in range(1, 6)])
    ddpowers = np.array([0.0, 0.0] + [k * (k - 1) * t**(k - 2) for k in range(2, 6)])
    r = coeffs @ powers
    r_dot = coeffs @ dpowers
    r_ddot = coeffs @ ddpowers
    return r, r_dot, r_ddot


def make_scenario(theta_max_deg, multi_axis):
    """x-axis: rest-to-rest quintic sweep 0 -> theta_max (r,r_dot,r_ddot all
    zero at both endpoints: the standard 10t^3-15t^4+6t^5 "smoothstep5").

    If multi_axis, adds a y-axis "bump" (t^3-2t^4+t^5, also zero
    value/velocity/accel at t=0 and zero value/velocity at t=1) that rises
    and falls independently of x. Using two DIFFERENT time-shapes (not the
    same shape rescaled per axis, which would keep r_dot parallel to r at
    all times -- a degenerate case where the Jacobian correction vanishes
    identically) makes the rotation axis itself sweep over time, which is
    what actually triggers the Jr correction."""
    theta_max = np.deg2rad(theta_max_deg)
    x_shape = np.array([0, 0, 0, 10, -15, 6])  # peak value at t=1: 1.0
    y_shape = np.array([0, 0, 0, 1, -2, 1])  # peak value ~0.0148 at t=0.6
    coeffs = np.zeros((3, 6))
    coeffs[0] = theta_max * x_shape
    if multi_axis:
        y_peak = 0.6**3 - 2 * 0.6**4 + 0.6**5
        coeffs[1] = (0.8 * theta_max / y_peak) * y_shape
    return coeffs


def true_omega_finite_diff(coeffs, t, dt=1e-6):
    r0, _, _ = quintic_rotvec_trajectory(coeffs, t)
    r1, _, _ = quintic_rotvec_trajectory(coeffs, t + dt)
    R0 = Rotation.from_rotvec(r0).as_matrix()
    R1 = Rotation.from_rotvec(r1).as_matrix()
    rel = Rotation.from_matrix(R0.T @ R1)
    return rel.as_rotvec() / dt


def omega_analytic(coeffs, t):
    r, r_dot, _ = quintic_rotvec_trajectory(coeffs, t)
    return right_jacobian(r) @ r_dot


def run_scenario(label, theta_max_deg, multi_axis):
    coeffs = make_scenario(theta_max_deg, multi_axis)
    ts = np.linspace(0.05, 0.95, 25)

    # Cross-check the analytic Jr formula against an independent
    # (matrix-log, no small-angle assumption) computation of omega.
    jr_errors = [
        np.linalg.norm(omega_analytic(coeffs, t) - true_omega_finite_diff(coeffs, t))
        for t in ts
    ]
    max_jr_cross_check_error = max(jr_errors)

    # omega_dot via a fine-step central difference of the (now validated)
    # analytic omega(t) -- NOT a coarse np.gradient over the 25-point ts
    # array, which would inject its own discretization error unrelated to
    # the Jacobian-correction effect being measured here.
    # Normalize by the trajectory's peak true torque (fixed reference), not
    # the instantaneous true value -- an instantaneous-relative metric blows
    # up to spurious "100%" near r_ddot's zero-crossing even when the
    # absolute buggy-vs-true gap there is genuinely tiny (verified: at
    # single-axis rotation the gap is ~1e-9, pure finite-diff noise).
    h = 1e-6
    tau_buggy_series, tau_true_series = [], []
    for t in ts:
        omega_dot_true = (omega_analytic(coeffs, t + h) - omega_analytic(coeffs, t - h)) / (2 * h)
        _, _, r_ddot = quintic_rotvec_trajectory(coeffs, t)
        tau_buggy_series.append(INERTIA * r_ddot)
        tau_true_series.append(INERTIA * omega_dot_true)
    tau_buggy_series = np.array(tau_buggy_series)
    tau_true_series = np.array(tau_true_series)
    peak_true = np.max(np.linalg.norm(tau_true_series, axis=1))
    torque_errors = np.linalg.norm(tau_buggy_series - tau_true_series, axis=1) / peak_true
    n_over = int(np.sum(np.linalg.norm(tau_buggy_series, axis=1) > np.linalg.norm(tau_true_series, axis=1)))

    print(f"{label}: Jr cross-check max|omega_analytic - omega_fd| = {max_jr_cross_check_error:.3e}"
          f" (validates the Jr formula independently of the buggy-code comparison)")
    print(f"{label}: buggy-vs-true torque relative error: mean={np.mean(torque_errors):.1%}"
          f", max={np.max(torque_errors):.1%}, buggy>true at {n_over}/{len(ts)} samples")


if __name__ == "__main__":
    run_scenario("small-angle (5deg, single-axis)", 5.0, multi_axis=False)
    run_scenario("small-angle (5deg, multi-axis)", 5.0, multi_axis=True)
    run_scenario("large-angle (150deg, single-axis)", 150.0, multi_axis=False)
    run_scenario("large-angle (150deg, multi-axis)", 150.0, multi_axis=True)
