#!/usr/bin/env python3
"""Follow-up to test/experiment_fan_thrust_box_qp_bisection.py: that script
showed a single fixed-T QP feasibility solve (fan-thrust box constraint) is
fast enough (~0.05s) for the 10Hz replan budget, but flagged several
unresolved issues (see docs/2026-08-29_mpcc_romero_investigation.md "この結果
の限界"): no infeasible-T fallback, a dummy objective with no anti-chattering
regularization, and no actual T-update rule tested across repeated ticks.

This script builds a small offline closed-loop harness (no ROS/sim) to probe
those three issues together:
  - T-update rule: decrement T by dt_tick each tick (less time needed as we
    approach the goal); if that T is infeasible, grow it (x1.5, up to 5
    retries) and re-check feasibility until one succeeds.
  - Anti-chattering regularization: penalize the new solve's first-node fan
    thrust f[0] deviating from the previous tick's *commanded* f[0], instead
    of the previous scripts' plain sum_squares(f)*1e-6 dummy objective.
  - Disturbance injection: at one tick, add a lateral (v_perp) velocity error
    directly to the true state (same failure mode as before), then observe
    whether the T-update rule recovers, how many retries it needs, and
    whether that tick's total solve time blows the 0.1s budget.

The "true state" rollout is a simplification -- each tick applies the fresh
solve's node-0 acceleration (constant over the tick) to advance (p, v) by
dt_tick, then re-solves from there next tick. This is not a physically exact
integrator, but it's enough to observe whether repeated re-solves keep the
commanded fan-thrust profile from jumping around from tick to tick.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_replanning_closed_loop.py
"""
import time

import cvxpy as cp
import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
N_NODES = 15
DT_TICK = 0.1  # guidance.replan_rate_hz = 10.0
REPLAN_BUDGET_S = DT_TICK

P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

N_TICKS = 40
DISTURB_TICK = 8
V_PERP_ERROR = 0.05

T_MIN_FLOOR = 1.0
T_GROWTH_FACTOR = 1.5
MAX_RETRIES = 5

CHATTER_WEIGHT = 1e-2  # penalize f[0] deviation from previous commanded f[0]


def build_qp(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, f0_prev_cmd, n=N_NODES):
    dt = T / (n - 1)
    p = cp.Variable((n, 3))
    v = cp.Variable((n, 3))
    f = cp.Variable((n, n_fans))
    a = (f @ A_force.T) / MASS

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

    objective = cp.Minimize(
        cp.sum_squares(f) * 1e-6
        + CHATTER_WEIGHT * cp.sum_squares(f[0, :] - f0_prev_cmd)
    )
    problem = cp.Problem(objective, constraints)
    return problem, p, v, f


def solve_at_T(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, f0_prev_cmd, n=N_NODES):
    problem, p, v, f = build_qp(A_force, fj_max, n_fans, p_start, v_start, p_goal, T, f0_prev_cmd, n)
    t0 = time.perf_counter()
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except cp.error.SolverError:
        return False, time.perf_counter() - t0, None, None
    elapsed = time.perf_counter() - t0
    feasible = problem.status in ("optimal", "optimal_inaccurate")
    if not feasible:
        return False, elapsed, None, None
    return True, elapsed, f.value[0], (f.value @ A_force.T)[0] / MASS


def replan_tick(A_force, fj_max, n_fans, p_true, v_true, p_goal, T_guess, f0_prev_cmd):
    """T-update rule: try T_guess; if infeasible, grow it up to MAX_RETRIES
    times. Returns (T_used, f0_cmd, a0, total_elapsed, n_attempts, feasible)."""
    T = max(T_guess, T_MIN_FLOOR)
    total_elapsed = 0.0
    for attempt in range(1, MAX_RETRIES + 1):
        feasible, elapsed, f0_cmd, a0 = solve_at_T(
            A_force, fj_max, n_fans, p_true, v_true, p_goal, T, f0_prev_cmd
        )
        total_elapsed += elapsed
        if feasible:
            return T, f0_cmd, a0, total_elapsed, attempt, True
        T *= T_GROWTH_FACTOR
    return T, f0_prev_cmd, np.zeros(3), total_elapsed, MAX_RETRIES, False


def main():
    allocator = ThrustAllocator()
    A_force = allocator.A[:3, :]
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count

    direction = (P1 - P0)
    direction /= np.linalg.norm(direction)
    lateral = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    lateral /= np.linalg.norm(lateral)

    p_true = P0.copy()
    v_true = np.zeros(3)
    f0_prev_cmd = np.zeros(n_fans)
    T_estimate = 12.0  # rough initial guess, close to earlier NLP's T~13.4s

    print(f"{'tick':>4} {'T_used':>8} {'attempts':>9} {'solve_s':>9} "
          f"{'over_budget':>12} {'|f0-f0_prev|':>13} {'dist_to_goal':>13}")

    over_budget_ticks = []
    infeasible_ticks = []
    chatter_norms = []

    for tick in range(N_TICKS):
        if tick == DISTURB_TICK:
            v_true = v_true + V_PERP_ERROR * lateral
            print(f"--- disturbance injected at tick {tick}: "
                  f"+{V_PERP_ERROR} m/s lateral ---")

        dist_to_goal = np.linalg.norm(P1 - p_true)
        if dist_to_goal < 0.01 and np.linalg.norm(v_true) < 0.01:
            print(f"converged at tick {tick} (dist={dist_to_goal*1000:.2f}mm)")
            break

        T_guess = max(T_estimate - DT_TICK, T_MIN_FLOOR)
        T_used, f0_cmd, a0, elapsed, attempts, feasible = replan_tick(
            A_force, fj_max, n_fans, p_true, v_true, P1, T_guess, f0_prev_cmd
        )
        over_budget = elapsed > REPLAN_BUDGET_S
        chatter = np.linalg.norm(f0_cmd - f0_prev_cmd)

        if over_budget:
            over_budget_ticks.append(tick)
        if not feasible:
            infeasible_ticks.append(tick)
        chatter_norms.append(chatter)

        print(f"{tick:>4} {T_used:>8.3f} {attempts:>9} {elapsed:>9.4f} "
              f"{str(over_budget):>12} {chatter:>13.5f} {dist_to_goal:>13.4f}")

        # Advance true state by one tick using this solve's node-0 acceleration.
        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK
        f0_prev_cmd = f0_cmd
        T_estimate = T_used

    print("\n--- Summary ---")
    print(f"ticks over 0.1s budget: {over_budget_ticks}")
    print(f"ticks that needed retries (infeasible at first T_guess or worse): "
          f"{infeasible_ticks}")
    print(f"chatter (|f0-f0_prev|) mean={np.mean(chatter_norms):.5f}, "
          f"max={np.max(chatter_norms):.5f}")
    print(f"final distance to goal: {np.linalg.norm(P1 - p_true)*1000:.2f} mm")


if __name__ == "__main__":
    main()
