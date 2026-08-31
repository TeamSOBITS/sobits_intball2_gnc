#!/usr/bin/env python3
"""6-DOF (position + attitude) extension of experiment_minco_closed_form_prototype.py.

Follow-up to docs/2026-08-29_minco_attitude_torque_integration_plan.md ("方針転換"
section): rather than chasing the paywalled geometric-MINCO paper's SO(3)-native
parameterization, this reuses the same trick ``ToppraTrajectory`` already runs in
production for the ``static`` mode (guidance/trajectory/toppra_trajectory.py) --
treat a rotation vector relative to a fixed reference orientation ``q0`` as a
plain extra 3 joint coordinates alongside position, giving one 6-DOF polynomial
path. MINCO's closed-form energy-minimizing interior derivative (see
experiment_minco_closed_form_prototype.py) and outer (q1,T1,T2) optimization
extend to this 6-DOF path with no structural change -- the Hermite basis math
below is dimension-agnostic.

Coupled force+torque constraint: instead of independent per-axis force/torque
box penalties (the earlier, known-too-generous simplification -- see
guidance/utils/actuation_envelope.py's module docstring, ~68% achievable /
~92% of a real path's feedforward wrench exceeded that box in production),
this uses the REAL wrench_envelope_halfspaces polytope (same one
ToppraTrajectory uses) as a smooth penalty function evaluated at trajectory
sample points -- not as a hard per-node NLP constraint the way the rejected
MPCC wrench_envelope experiment did (docs/
2026-08-29_mpcc_attitude_torque_integration_plan.md "追記（推力予算取り合いへの
対策検討）", ~9951 halfspaces x N_HORIZON stages made acados' SQP unusable).
Here it's just a fixed (F_env, g_env) matrix applied to a dense array of
sampled wrenches via ``F_env @ wrench.T`` -- no per-sample decision variables,
same computational shape TOPP-RA's own (hard-constraint, not penalty) use of
this exact envelope already handles fine in production.

Scenario: same real hairpin (near_dock -> bulge waypoint -> above_dock, 144.7deg
turn) as experiment_minco_closed_form_prototype.py. Attitude waypoints: q0
(reference, rotvec=0) unchanged for segment 1, then face the second leg's
direction (Q2-Q1) via guidance/utils/attitude_reference.compute_q_des -- a
single "corner" attitude switch, not per-sample face_travel (MINCO's whole
point is a coarse waypoint set + closed-form polynomial in between, unlike
TOPP-RA's dense resampling), rest-to-rest (zero rotational velocity/
acceleration) at both ends, matching the existing rest-to-rest translation
boundary conditions.

Question this answers: does adding the real angular-acceleration demand
(coupled into the same wrench-envelope check as translation, sharing the same
8-fan thrust budget) change the achievable min time / feasibility relative to
the translation-only result, and is the real-polytope penalty tractable
inside MINCO's L-BFGS loop (as opposed to the box-constraint approximation)?

Not a pytest test (no test_ prefix, not collected by colcon test) -- run
directly:
    python3 test/experiment_minco_attitude_prototype.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import wrench_envelope_halfspaces
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

jax.config.update("jax_enable_x64", True)

MASS = 3.216      # kg, config/gnc_params.yaml trajectory_controller.mass
INERTIA = 0.0136  # kg*m^2, isotropic, trajectory_controller.inertia
SAFETY_MARGIN = 0.7  # matches config/gnc_params.yaml guidance.wrench_envelope_safety_margin

NEAR_DOCK = jnp.array([10.936, -3.636, 4.121])
ABOVE_DOCK = jnp.array([10.936, -3.636, 5.0])
NAV_ENTRY = jnp.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5

P0 = NEAR_DOCK
P2 = ABOVE_DOCK
_midpoint = 0.5 * (P0 + P2)
_bulge = NAV_ENTRY - _midpoint
P1_GIVEN = _midpoint + BULGE_SCALE * _bulge

P1_BOX_HALF_WIDTH = 0.3
N_SAMPLES = 30
W_ENERGY = 1e-3
ZERO6 = jnp.zeros(6)
IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def _attitude_waypoints():
    """rv0=0 (reference itself); rv1=rv2= the second leg's face_travel target
    (a single corner switch -- see module docstring)."""
    leg2_dir = np.asarray(P2 - P1_GIVEN)
    q_target = compute_q_des(leg2_dir, None, 1e-9, forward_axis=(1.0, 0.0, 0.0))
    rv_target = quat_log(quat_mul(quat_conj(IDENTITY_Q), q_target))
    return jnp.zeros(3), jnp.asarray(rv_target), jnp.asarray(rv_target)


_RV0, _RV1, _RV2 = _attitude_waypoints()
Q0 = jnp.concatenate([P0, _RV0])
Q1_GIVEN_6 = jnp.concatenate([P1_GIVEN, _RV1])
Q2 = jnp.concatenate([P2, _RV2])


def _build_wrench_envelope():
    allocator = ThrustAllocator()
    F_env, g_env = wrench_envelope_halfspaces(
        allocator.A, allocator.fj_max, safety_margin=SAFETY_MARGIN
    )
    return jnp.asarray(F_env), jnp.asarray(g_env)


F_ENV, G_ENV = _build_wrench_envelope()
INERTIA_DIAG = jnp.array([MASS, MASS, MASS, INERTIA, INERTIA, INERTIA])


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


def segment_wrench_penalty(T, p0, v0, a0, p1, v1, a1, weight):
    accel = hermite_deriv(_TAU, T, p0, v0, a0, p1, v1, a1, 2)  # (N, 6)
    wrench = accel * INERTIA_DIAG[None, :]                      # (N, 6)
    viol = jnp.maximum(F_ENV @ wrench.T - G_ENV[:, None], 0.0)  # (n_faces, N)
    return weight * jnp.sum(viol**3) / N_SAMPLES


def segment_energy(T, p0, v0, a0, p1, v1, a1):
    jerk = hermite_deriv(_TAU, T, p0, v0, a0, p1, v1, a1, 3)
    return jnp.sum(jerk**2) / N_SAMPLES * T


def _total_energy(x, q0, q1, q2, T1, T2):
    v1, a1 = x[0:6], x[6:12]
    e1 = segment_energy(T1, q0, ZERO6, ZERO6, q1, v1, a1)
    e2 = segment_energy(T2, q1, v1, a1, q2, ZERO6, ZERO6)
    return e1 + e2


def solve_v1_a1(q0, q1, q2, T1, T2):
    x0 = jnp.zeros(12)
    g = jax.grad(_total_energy)(x0, q0, q1, q2, T1, T2)
    H = jax.hessian(_total_energy)(x0, q0, q1, q2, T1, T2)
    x_star = jnp.linalg.solve(H, -g)
    return x_star[0:6], x_star[6:12]


def outer_objective(p1, T1, T2, weight):
    q1 = jnp.concatenate([p1, _RV1])
    v1, a1 = solve_v1_a1(Q0, q1, Q2, T1, T2)
    e1 = segment_energy(T1, Q0, ZERO6, ZERO6, q1, v1, a1)
    e2 = segment_energy(T2, q1, v1, a1, Q2, ZERO6, ZERO6)
    p1_pen = segment_wrench_penalty(T1, Q0, ZERO6, ZERO6, q1, v1, a1, weight)
    p2_pen = segment_wrench_penalty(T2, q1, v1, a1, Q2, ZERO6, ZERO6, weight)
    return T1 + T2 + W_ENERGY * (e1 + e2) + p1_pen + p2_pen


def unpack(x):
    return x[0:3], x[3], x[4]


def objective(x, weight):
    p1, T1, T2 = unpack(x)
    return outer_objective(p1, T1, T2, weight)


_obj_jit = jax.jit(objective)
_grad_jit = jax.jit(jax.grad(objective))


def objective_and_grad(x_np, weight):
    x = jnp.asarray(x_np)
    return float(_obj_jit(x, weight)), np.asarray(_grad_jit(x, weight), dtype=np.float64)


@jax.jit
def _max_violation_jit(p1, T1, T2):
    q1 = jnp.concatenate([p1, _RV1])
    v1, a1 = solve_v1_a1(Q0, q1, Q2, T1, T2)
    a_seg1 = hermite_deriv(_TAU, T1, Q0, ZERO6, ZERO6, q1, v1, a1, 2)
    a_seg2 = hermite_deriv(_TAU, T2, q1, v1, a1, Q2, ZERO6, ZERO6, 2)
    w1 = a_seg1 * INERTIA_DIAG[None, :]
    w2 = a_seg2 * INERTIA_DIAG[None, :]
    v1_ = jnp.max(jnp.maximum(F_ENV @ w1.T - G_ENV[:, None], 0.0))
    v2_ = jnp.max(jnp.maximum(F_ENV @ w2.T - G_ENV[:, None], 0.0))
    return jnp.maximum(v1_, v2_), v1, a1


def max_violation(p1, T1, T2):
    viol, v1, a1 = _max_violation_jit(p1, T1, T2)
    return float(viol), v1, a1


def solve(p1_free, weight_schedule=(1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)):
    if p1_free:
        lower = np.asarray(P1_GIVEN) - P1_BOX_HALF_WIDTH
        upper = np.asarray(P1_GIVEN) + P1_BOX_HALF_WIDTH
    else:
        lower = upper = np.asarray(P1_GIVEN)

    T_guess = 15.0
    x = np.concatenate([np.asarray(P1_GIVEN), [T_guess, T_guess]])
    bounds = [(lower[i], upper[i]) for i in range(3)] + [(1e-3, None), (1e-3, None)]

    outer_iters = 0
    for weight in weight_schedule:
        outer_iters += 1
        result = minimize(
            objective_and_grad, x, args=(weight,), jac=True, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 500, "ftol": 1e-14},
        )
        x = result.x

    p1, T1, T2 = unpack(jnp.asarray(x))
    viol, v1, a1 = max_violation(p1, T1, T2)
    return np.asarray(p1), np.asarray(v1), np.asarray(a1), float(T1), float(T2), viol, outer_iters


def main():
    p1_g, p0_g, p2_g = np.asarray(P1_GIVEN), np.asarray(P0), np.asarray(P2)
    turn_angle = np.degrees(np.arccos(np.clip(
        np.dot(p1_g - p0_g, p2_g - p1_g)
        / (np.linalg.norm(p1_g - p0_g) * np.linalg.norm(p2_g - p1_g)),
        -1.0, 1.0,
    )))
    print(f"hairpin: near_dock -> W -> above_dock, turn_angle={turn_angle:.1f}deg")
    print(f"attitude waypoints: rv0={np.asarray(_RV0)}, rv1=rv2={np.asarray(_RV1)}")
    print(f"wrench envelope: {F_ENV.shape[0]} halfspaces (real 8-fan zonotope, "
          f"safety_margin={SAFETY_MARGIN})")
    print("v1,a1 (12-dim: 6 translation + 6 rotation) are closed-form energy minimizers")
    print()

    x0 = np.concatenate([p1_g, [15.0, 15.0]])
    objective_and_grad(x0, 1e2)  # warm-up (JIT compile), excluded from timing
    max_violation(jnp.asarray(p1_g), 15.0, 15.0)

    for label, p1_free in [("baseline (p1 fixed)", False), ("minco-style (p1 free +-0.3m)", True)]:
        t_start = time.perf_counter()
        p1, v1, a1, T1, T2, viol, outer_iters = solve(p1_free)
        elapsed = time.perf_counter() - t_start
        print(f"--- {label} ---")
        print(f"p1:            {p1}  (moved {np.linalg.norm(p1 - p1_g):.4f} m from given)")
        print(f"v1 (interior, closed-form, [trans;rot]): {v1}")
        print(f"a1 (interior, closed-form, [trans;rot]): {a1}")
        print(f"T1, T2:        {T1:.4f}, {T2:.4f}  (total {T1 + T2:.4f} s)")
        print(f"max wrench-envelope violation: {viol:.3e}")
        print(f"outer iterations: {outer_iters}, solve time: {elapsed:.4f} s")
        print()


if __name__ == "__main__":
    main()
