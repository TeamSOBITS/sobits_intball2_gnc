#!/usr/bin/env python3
"""Prototype: does a MINCO/GCOPTER-style penalty formulation (translation-only,
axis-independent force box constraint) find a feasible segment time T fast?

Follow-up to docs/2026-08-29_minco_gcopter_survey.md and
docs/2026-08-29_scp_static_prototype_findings.md. GCOPTER itself is a C++
library (github.com/ZJU-FAST-Lab/GCOPTER) with no Python bindings used here --
this script re-implements just the piece under test (its penalty-based
constraint mechanism) in plain Python/scipy, on the simplest possible case:
a single rest-to-rest quintic (minimum-jerk) segment, so the closed-form
peak-acceleration formula gives a ground truth to check the numeric optimizer
against.

Same rest-to-rest scenario and per-axis force box as the earlier prototypes:
  - docs/2026-08-29_scp_static_prototype_findings.md (試作1, axis-independent
    max_accel box, closed-form-adjacent NLP: 0.13s)
  - test/experiment_fan_thrust_box_nlp.py (actuator-space reformulation)

What's different here vs. those: instead of direct collocation (many node
variables), the trajectory shape is *analytic* in T (quintic polynomial,
MINCO's "solve for c given q,T" step), and only T is a free variable, with an
integral penalty on acceleration-box violation added to a w_T*T time cost --
this is the actual mechanism GCOPTER uses to fold user constraints into the
cost, just applied to the minimal 1-segment/1-free-variable case instead of
GCOPTER's general multi-segment/multi-waypoint machinery.

Not a pytest test (no test_ prefix, not collected by colcon test) -- run
directly:
    python3 test/experiment_minco_box_constraint_prototype.py
"""
import time

import numpy as np
from scipy.optimize import minimize

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass

# Same rest-to-rest scenario as docs/2026-08-29_scp_static_prototype_findings.md
P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

# config/gnc_params.yaml trajectory_controller.max_force (x,y,z), theoretical
# per-axis max from the fan model (docs/2026-08-27_max_force_anisotropy_from_fan_model.md)
MAX_FORCE = np.array([0.181, 0.0996, 0.122])
MAX_ACCEL = MAX_FORCE / MASS

N_SAMPLES = 50
PEAK_ACCEL_COEFF = 10.0 * np.sqrt(3.0) / 3.0  # quintic min-jerk peak-accel constant


def quintic_accel(tau, delta_p, T):
    """a(t) for a rest-to-rest quintic (minimum-jerk) segment, tau = t/T in [0,1]."""
    return delta_p * (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / T**2


def analytic_min_time(delta_p, max_accel):
    """Closed-form T making the quintic's peak |accel| exactly hit max_accel,
    per axis (independent boxes), taking the binding (slowest) axis. Ground
    truth to check the penalty-based numeric optimizer against -- production
    multi-segment trajectories won't have this closed form, which is exactly
    why the penalty/quadrature approach is being tested."""
    T_i = np.sqrt(PEAK_ACCEL_COEFF * np.abs(delta_p) / max_accel)
    return np.max(T_i)


def penalty(T, delta_p, max_accel, weight):
    tau = np.linspace(0.0, 1.0, N_SAMPLES)
    a = quintic_accel(tau[:, None], delta_p[None, :], T)  # (N_SAMPLES, 3)
    viol = np.maximum(np.abs(a) - max_accel[None, :], 0.0)
    return weight * np.sum(viol**3) / N_SAMPLES  # mean over samples, summed over axes


def solve_minco_style(
    delta_p, max_accel, w_time=1.0, weight_schedule=(1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)
):
    """Continuation over penalty weight (GCOPTER's standard practical scheme):
    solve, then increase weight and re-solve from the previous T, until the
    max constraint violation is within tolerance."""
    T = analytic_min_time(delta_p, max_accel) * 1.5  # deliberately-loose warm start
    outer_iters = 0
    for weight in weight_schedule:
        outer_iters += 1

        def objective(x, weight=weight):
            T_local = x[0]
            return w_time * T_local + penalty(T_local, delta_p, max_accel, weight)

        result = minimize(
            objective,
            x0=[T],
            method="L-BFGS-B",
            bounds=[(1e-3, None)],
            options={"maxiter": 200, "ftol": 1e-12},
        )
        T = result.x[0]

        tau = np.linspace(0.0, 1.0, N_SAMPLES)
        a = quintic_accel(tau[:, None], delta_p[None, :], T)
        max_violation = np.max(np.maximum(np.abs(a) - max_accel[None, :], 0.0))
        if max_violation < 1e-6:
            break
    return T, outer_iters, max_violation


def main():
    delta_p = P1 - P0

    t_start = time.perf_counter()
    T_found, outer_iters, max_violation = solve_minco_style(delta_p, MAX_ACCEL)
    elapsed = time.perf_counter() - t_start

    T_analytic = analytic_min_time(delta_p, MAX_ACCEL)

    print(f"delta_p:             {delta_p}")
    print(f"max_accel (x,y,z):   {MAX_ACCEL}")
    print(f"analytic T*:         {T_analytic:.4f} s (closed-form ground truth)")
    print(f"penalty-optimized T: {T_found:.4f} s")
    print(f"relative error:      {abs(T_found - T_analytic) / T_analytic:.3e}")
    print(f"outer (continuation) iterations: {outer_iters}")
    print(f"max residual violation: {max_violation:.3e}")
    print(f"total solve time:    {elapsed:.4f} s")


if __name__ == "__main__":
    main()
