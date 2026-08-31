#!/usr/bin/env python3
"""Follow-up to test/experiment_fan_thrust_box_nlp_warmstart.py: warm-starting
scipy SLSQP didn't help (0.155s cold vs 0.166s warm for a re-plan sub-leg,
still ~1.5x over the 10Hz/0.1s replan budget in config/gnc_params.yaml
guidance.replan_rate_hz). This script tests a different lever: reformulate as
a QP instead of an NLP.

For a *fixed* T, trapezoidal collocation is linear in (p, v, f) -- dt=T/(n-1)
is then a constant, so the position/velocity defects are linear equalities,
and the only remaining constraint (0 <= f_j <= fj_max, the fan-thrust box) is
already linear. So "is T feasible?" becomes a convex QP feasibility problem,
and the min-time trajectory can be found by bisecting on T. This is the same
idea docs/2026-08-29_scp_static_prototype_findings.md prototype 3 tried
(cvxpy+OSQP+bisection) -- but that one used the real ~10000-face wrench
envelope polytope and took ~30s per QP solve (worse than SLSQP). Swapping in
the fan-thrust box constraint (same trick that fixed the NLP version, see
docs/2026-08-29_mpcc_romero_investigation.md) should make each QP solve
trivially small (a few hundred box bounds + linear equalities, not 10000
polytope faces) -- this script checks whether that holds up and whether
bisection-to-convergence beats the 0.1s replan budget.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_fan_thrust_box_qp_bisection.py
"""
import time

import cvxpy as cp
import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
N_NODES = 15
REPLAN_BUDGET_S = 1.0 / 10.0  # guidance.replan_rate_hz = 10.0

P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

# Same disturbance scenario as the warm-start experiment: re-plan from 1/3
# along the leg with a lateral (v_perp) velocity error injected.
V_PERP_ERROR = 0.05
REPLAN_NODE_FRACTION = 1.0 / 3.0


def build_qp_feasibility(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, n=N_NODES):
    """Return (problem, solve_fn) -- a cvxpy Problem that is feasible iff T
    admits a dynamically-consistent, fan-thrust-box-respecting trajectory
    from (p_start, v_start) to p_goal in time T."""
    dt = T / (n - 1)
    p = cp.Variable((n, 3))
    v = cp.Variable((n, 3))
    f = cp.Variable((n, n_fans))

    a = (f @ A_force.T) / MASS  # (n, 3)

    constraints = [
        p[0] == p_start,
        p[n - 1] == p_goal,
        v[0] == v_start,
        v[n - 1] == 0,
        f >= 0,
        f <= fj_max,
    ]
    for k in range(n - 1):
        constraints.append(p[k + 1] - p[k] == dt / 2.0 * (v[k + 1] + v[k]))
        constraints.append(v[k + 1] - v[k] == dt / 2.0 * (a[k + 1] + a[k]))

    # Feasibility problem -- minimize a trivial regularizer just to keep OSQP
    # well-posed (pure feasibility with a zero objective can be degenerate).
    objective = cp.Minimize(cp.sum_squares(f) * 1e-6)
    problem = cp.Problem(objective, constraints)
    return problem


def is_feasible(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, n=N_NODES):
    problem = build_qp_feasibility(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, n)
    t0 = time.perf_counter()
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except cp.error.SolverError:
        return False, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    feasible = problem.status in ("optimal", "optimal_inaccurate")
    return feasible, elapsed


def bisect_min_time(A_force, fj_max, n_fans, p_start, v_start, p_goal, t_lo, t_hi, n=N_NODES, tol=0.05):
    total_time = 0.0
    n_solves = 0
    solve_times = []
    feasible_hi, elapsed = is_feasible(A_force, fj_max, n_fans, p_start, v_start, p_goal, t_hi, n)
    total_time += elapsed
    solve_times.append(elapsed)
    n_solves += 1
    if not feasible_hi:
        raise RuntimeError(f"t_hi={t_hi} infeasible, widen search bracket")

    while t_hi - t_lo > tol:
        t_mid = 0.5 * (t_lo + t_hi)
        feasible, elapsed = is_feasible(A_force, fj_max, n_fans, p_start, v_start, p_goal, t_mid, n)
        total_time += elapsed
        solve_times.append(elapsed)
        n_solves += 1
        if feasible:
            t_hi = t_mid
        else:
            t_lo = t_mid

    return t_hi, total_time, n_solves, solve_times


def main():
    allocator = ThrustAllocator()
    A_force = allocator.A[:3, :]
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count
    n = N_NODES

    print("--- Single QP feasibility solve timing (fixed T, sanity check) ---")
    feasible, elapsed = is_feasible(A_force, fj_max, n_fans, P0, np.zeros(3), P1, 15.0, n)
    print(f"T=15.0s feasible={feasible}, solve time={elapsed:.4f}s")

    print("\n--- Re-plan scenario: disturbed mid-flight state, bisect min T ---")
    # Reuse the same disturbed replan state as the warm-start experiment
    # would produce, approximated directly (no need to re-solve the NLP
    # baseline here -- just need a plausible off-nominal (p, v)).
    direction = (P1 - P0)
    direction /= np.linalg.norm(direction)
    lateral = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    lateral /= np.linalg.norm(lateral)

    p_replan = P0 + REPLAN_NODE_FRACTION * (P1 - P0)
    v_replan = 0.05 * direction + V_PERP_ERROR * lateral

    t_min, total_time, n_solves, solve_times = bisect_min_time(
        A_force, fj_max, n_fans, p_replan, v_replan, P1, t_lo=0.5, t_hi=15.0, n=n
    )
    print(f"bisected min T: {t_min:.3f}s")
    print(f"total bisection time: {total_time:.3f}s over {n_solves} QP solves")
    print(f"per-solve times: {[f'{t:.4f}' for t in solve_times]}")
    print(f"mean per-solve: {np.mean(solve_times):.4f}s, max: {np.max(solve_times):.4f}s")

    print("\n--- Comparison against 10Hz replan budget "
          f"({REPLAN_BUDGET_S:.3f}s) ---")
    print(f"single QP solve (mean):  {np.mean(solve_times):.4f}s "
          f"({'OK' if np.mean(solve_times) < REPLAN_BUDGET_S else 'OVER BUDGET'})")
    print(f"full bisection ({n_solves} solves): {total_time:.3f}s "
          f"({'OK' if total_time < REPLAN_BUDGET_S else 'OVER BUDGET'})")


if __name__ == "__main__":
    main()
