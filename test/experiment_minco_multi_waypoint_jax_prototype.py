#!/usr/bin/env python3
"""Same problem as experiment_minco_multi_waypoint_prototype.py, but with
JAX-autodiff (exact) gradients instead of scipy's finite-difference default.

That earlier script found the joint (q,T) optimization gives a real ~16%
time reduction on the real 144.7deg hairpin, but took 1.9-2.75s per solve --
and flagged that this says nothing about MINCO's actual speed claim, because
finite-difference gradients (11 variables re-evaluated per step) are not what
makes MINCO/GCOPTER fast. MINCO's real speedup comes from a closed-form
adjoint that gives dJ/dq,dJ/dT in one pass, without ever forming a Jacobian
by finite differences. Deriving that adjoint by hand is out of scope for a
"try it lightly" prototype, but swapping in *exact* (not finite-difference)
gradients via autodiff isolates how much of the earlier slowness was just
"finite differences are wasteful" vs. something more fundamental.

Not a pytest test (no test_ prefix, not collected by colcon test) -- run
directly:
    python3 test/experiment_minco_multi_waypoint_jax_prototype.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

MASS = 3.216
MAX_FORCE = jnp.array([0.181, 0.0996, 0.122])
MAX_ACCEL = MAX_FORCE / MASS

NEAR_DOCK = jnp.array([10.936, -3.636, 4.121])
ABOVE_DOCK = jnp.array([10.936, -3.636, 5.0])
NAV_ENTRY = jnp.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5

Q0 = NEAR_DOCK
Q2 = ABOVE_DOCK
_midpoint = 0.5 * (Q0 + Q2)
_bulge = NAV_ENTRY - _midpoint
Q1_GIVEN = _midpoint + BULGE_SCALE * _bulge

Q1_BOX_HALF_WIDTH = 0.3
N_SAMPLES = 30
W_ENERGY = 1e-3
ZERO3 = jnp.zeros(3)


def _hermite_bases(tau, order):
    t2, t3, t4 = tau**2, tau**3, tau**4
    if order == 0:
        h00 = 1 - 10 * t3 + 15 * t4 - 6 * t4 * tau
        h10 = tau - 6 * t3 + 8 * t4 - 3 * t4 * tau
        h20 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t4 * tau
        h01 = 10 * t3 - 15 * t4 + 6 * t4 * tau
        h11 = -4 * t3 + 7 * t4 - 3 * t4 * tau
        h21 = 0.5 * t3 - t4 + 0.5 * t4 * tau
    elif order == 1:
        h00 = -30 * t2 + 60 * t3 - 30 * t4
        h10 = 1 - 18 * t2 + 32 * t3 - 15 * t4
        h20 = tau - 4.5 * t2 + 6 * t3 - 2.5 * t4
        h01 = 30 * t2 - 60 * t3 + 30 * t4
        h11 = -12 * t2 + 28 * t3 - 15 * t4
        h21 = 1.5 * t2 - 4 * t3 + 2.5 * t4
    elif order == 2:
        h00 = -60 * tau + 180 * t2 - 120 * t3
        h10 = -36 * tau + 96 * t2 - 60 * t3
        h20 = 1 - 9 * tau + 18 * t2 - 10 * t3
        h01 = 60 * tau - 180 * t2 + 120 * t3
        h11 = -24 * tau + 84 * t2 - 60 * t3
        h21 = 3 * tau - 12 * t2 + 10 * t3
    else:  # order == 3
        h00 = -60 + 360 * tau - 360 * t2
        h10 = -36 + 192 * tau - 180 * t2
        h20 = -9 + 36 * tau - 30 * t2
        h01 = 60 - 360 * tau + 360 * t2
        h11 = -24 + 168 * tau - 180 * t2
        h21 = 3 - 24 * tau + 30 * t2
    return h00, h10, h20, h01, h11, h21


def hermite_deriv(tau, T, p0, v0, a0, p1, v1, a1, deriv_order):
    h00, h10, h20, h01, h11, h21 = _hermite_bases(tau, deriv_order)
    d_tau = (
        p0 * h00 + v0 * T * h10 + a0 * T**2 * h20
        + p1 * h01 + v1 * T * h11 + a1 * T**2 * h21
    )
    return d_tau / T**deriv_order


_TAU = jnp.linspace(0.0, 1.0, N_SAMPLES)[:, None]


def segment_penalty(T, p0, v0, a0, p1, v1, a1, weight):
    accel = hermite_deriv(_TAU, T, p0, v0, a0, p1, v1, a1, 2)
    viol = jnp.maximum(jnp.abs(accel) - MAX_ACCEL[None, :], 0.0)
    return weight * jnp.sum(viol**3) / N_SAMPLES


def segment_energy(T, p0, v0, a0, p1, v1, a1):
    jerk = hermite_deriv(_TAU, T, p0, v0, a0, p1, v1, a1, 3)
    return jnp.sum(jerk**2) / N_SAMPLES * T


def unpack(x):
    return x[0:3], x[3:6], x[6:9], x[9], x[10]


def objective(x, weight):
    q1, v1, a1, T1, T2 = unpack(x)
    e1 = segment_energy(T1, Q0, ZERO3, ZERO3, q1, v1, a1)
    e2 = segment_energy(T2, q1, v1, a1, Q2, ZERO3, ZERO3)
    p1 = segment_penalty(T1, Q0, ZERO3, ZERO3, q1, v1, a1, weight)
    p2 = segment_penalty(T2, q1, v1, a1, Q2, ZERO3, ZERO3, weight)
    return T1 + T2 + W_ENERGY * (e1 + e2) + p1 + p2


_objective_jit = jax.jit(objective)
_grad_jit = jax.jit(jax.grad(objective))


def objective_and_grad(x_np, weight):
    x = jnp.asarray(x_np)
    val = _objective_jit(x, weight)
    grad = _grad_jit(x, weight)
    return float(val), np.asarray(grad, dtype=np.float64)


def max_violation(q1, v1, a1, T1, T2):
    a_seg1 = hermite_deriv(_TAU, T1, Q0, ZERO3, ZERO3, q1, v1, a1, 2)
    a_seg2 = hermite_deriv(_TAU, T2, q1, v1, a1, Q2, ZERO3, ZERO3, 2)
    v1_ = jnp.max(jnp.maximum(jnp.abs(a_seg1) - MAX_ACCEL[None, :], 0.0))
    v2_ = jnp.max(jnp.maximum(jnp.abs(a_seg2) - MAX_ACCEL[None, :], 0.0))
    return float(jnp.maximum(v1_, v2_))


def solve(q1_free, weight_schedule=(1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)):
    if q1_free:
        lower = np.asarray(Q1_GIVEN) - Q1_BOX_HALF_WIDTH
        upper = np.asarray(Q1_GIVEN) + Q1_BOX_HALF_WIDTH
    else:
        lower = upper = np.asarray(Q1_GIVEN)

    T_guess = 15.0
    x = np.concatenate([np.asarray(Q1_GIVEN), np.zeros(3), np.zeros(3), [T_guess, T_guess]])

    bounds = (
        [(lower[i], upper[i]) for i in range(3)]
        + [(None, None)] * 6
        + [(1e-3, None), (1e-3, None)]
    )

    outer_iters = 0
    for weight in weight_schedule:
        outer_iters += 1
        result = minimize(
            objective_and_grad, x, args=(weight,), jac=True, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 500, "ftol": 1e-14},
        )
        x = result.x

    q1, v1, a1, T1, T2 = unpack(jnp.asarray(x))
    viol = max_violation(q1, v1, a1, T1, T2)
    return np.asarray(q1), np.asarray(v1), np.asarray(a1), float(T1), float(T2), viol, outer_iters


def main():
    q1_g, q0_g, q2_g = np.asarray(Q1_GIVEN), np.asarray(Q0), np.asarray(Q2)
    turn_angle = np.degrees(np.arccos(np.clip(
        np.dot(q1_g - q0_g, q2_g - q1_g)
        / (np.linalg.norm(q1_g - q0_g) * np.linalg.norm(q2_g - q1_g)),
        -1.0, 1.0,
    )))
    print(f"hairpin: near_dock -> W -> above_dock, turn_angle={turn_angle:.1f}deg")
    print()

    # Warm-up call so JIT compilation time doesn't pollute the timed solve.
    objective_and_grad(np.concatenate([q1_g, np.zeros(3), np.zeros(3), [15.0, 15.0]]), 1e2)

    for label, q1_free in [("baseline (q1 fixed)", False), ("minco-style (q1 free +-0.3m)", True)]:
        t_start = time.perf_counter()
        q1, v1, a1, T1, T2, viol, outer_iters = solve(q1_free)
        elapsed = time.perf_counter() - t_start
        print(f"--- {label} ---")
        print(f"q1:            {q1}  (moved {np.linalg.norm(q1 - q1_g):.4f} m from given)")
        print(f"T1, T2:        {T1:.4f}, {T2:.4f}  (total {T1 + T2:.4f} s)")
        print(f"max accel-box violation: {viol:.3e}")
        print(f"outer iterations: {outer_iters}, solve time: {elapsed:.4f} s")
        print()


if __name__ == "__main__":
    main()
