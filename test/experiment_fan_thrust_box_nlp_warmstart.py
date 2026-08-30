#!/usr/bin/env python3
"""Follow-up to test/experiment_fan_thrust_box_nlp.py: does warm-starting
from a previous solution make the fan-thrust box-constraint NLP fast enough
for repeated ("replanning") re-solves, or is 1.5s/solve a hard floor?

Scenario: solve once (cold start) for the full P0->P1 rest-to-rest leg, same
as experiment_fan_thrust_box_nlp.py. Then simulate a mid-flight disturbance
-- at some node along that solution, perturb the velocity sideways (a
lateral/v_perp error, the same failure mode that stalls the current
HeuristicSegmentTimeAllocator, see main_plan.md "[G] replanningモードでの
segment_time_infeasible時の大きなオーバーシュート") and re-solve a fresh
min-time problem from that perturbed state to P1. Compare cold-start vs.
warm-start (previous solution's tail, reused as the initial guess) solve
time for this re-plan.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_fan_thrust_box_nlp_warmstart.py
"""
import time

import numpy as np
from scipy.optimize import minimize

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
N_NODES = 15

P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

# Simulated disturbance: at the replan node, add a lateral velocity error
# (perpendicular to the direction-of-travel) of this magnitude [m/s] -- the
# same kind of v_perp mismatch that stalls HeuristicSegmentTimeAllocator.
V_PERP_ERROR = 0.05
REPLAN_NODE_FRACTION = 1.0 / 3.0


def unpack(x, n_fans, n):
    T = x[0]
    rest = x[1:]
    p = rest[: n * 3].reshape(n, 3)
    v = rest[n * 3 : n * 6].reshape(n, 3)
    f = rest[n * 6 :].reshape(n, n_fans)
    return T, p, v, f


def pack(T, p, v, f):
    return np.concatenate([[T], p.ravel(), v.ravel(), f.ravel()])


def solve_nlp(A_force, fj_max, n_fans, p_start, v_start, p_goal, x0, n=N_NODES):
    def accel(f_nodes):
        return (f_nodes @ A_force.T) / MASS

    def defects(x):
        T, p, v, f = unpack(x, n_fans, n)
        dt = T / (n - 1)
        a = accel(f)
        pos_defect = p[1:] - p[:-1] - dt / 2.0 * (v[1:] + v[:-1])
        vel_defect = v[1:] - v[:-1] - dt / 2.0 * (a[1:] + a[:-1])
        boundary = np.concatenate(
            [p[0] - p_start, p[-1] - p_goal, v[0] - v_start, v[-1]]
        )
        return np.concatenate([pos_defect.ravel(), vel_defect.ravel(), boundary])

    def objective(x):
        return x[0]

    bounds = [(1e-3, None)]
    bounds += [(None, None)] * (n * 3)
    bounds += [(None, None)] * (n * 3)
    bounds += [(0.0, fj_max)] * (n * n_fans)

    constraints = [{"type": "eq", "fun": defects}]

    t_start = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    elapsed = time.perf_counter() - t_start
    residual = np.max(np.abs(defects(result.x))) if result.success else float("nan")
    return result, elapsed, residual


def cold_guess(p_start, v_start, p_goal, n, n_fans, fj_max, T_guess=10.0):
    p_guess = np.linspace(p_start, p_goal, n)
    v_guess = np.zeros((n, 3))
    v_guess[0] = v_start
    f_guess = np.full((n, n_fans), fj_max / 2.0)
    return pack(T_guess, p_guess, v_guess, f_guess)


def main():
    allocator = ThrustAllocator()
    A_force = allocator.A[:3, :]
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count
    n = N_NODES

    print("--- Step 1: cold solve full P0 -> P1 leg (baseline, no disturbance) ---")
    x0_cold = cold_guess(P0, np.zeros(3), P1, n, n_fans, fj_max)
    result1, elapsed1, residual1 = solve_nlp(
        A_force, fj_max, n_fans, P0, np.zeros(3), P1, x0_cold, n
    )
    T1, p1, v1, f1 = unpack(result1.x, n_fans, n)
    print(f"success: {result1.success}, solve time: {elapsed1:.3f}s, "
          f"T={T1:.3f}s, residual={residual1:.3e}")

    replan_idx = int(round((n - 1) * REPLAN_NODE_FRACTION))
    p_replan = p1[replan_idx].copy()
    v_replan = v1[replan_idx].copy()
    direction = (P1 - P0)
    direction /= np.linalg.norm(direction)
    lateral = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    lateral /= np.linalg.norm(lateral)
    v_replan_disturbed = v_replan + V_PERP_ERROR * lateral

    print(f"\n--- Step 2: disturbance injected at node {replan_idx}/{n - 1} ---")
    print(f"planned v there: {v_replan}, disturbed v: {v_replan_disturbed} "
          f"(+{V_PERP_ERROR} m/s lateral)")

    print("\n--- Step 3a: re-solve from disturbed state, COLD start ---")
    x0_replan_cold = cold_guess(
        p_replan, v_replan_disturbed, P1, n, n_fans, fj_max, T_guess=T1 * (1 - REPLAN_NODE_FRACTION)
    )
    result_cold, elapsed_cold, residual_cold = solve_nlp(
        A_force, fj_max, n_fans, p_replan, v_replan_disturbed, P1, x0_replan_cold, n
    )
    print(f"success: {result_cold.success}, solve time: {elapsed_cold:.3f}s, "
          f"residual={residual_cold:.3e}")

    print("\n--- Step 3b: re-solve from disturbed state, WARM start "
          "(reuse original solution's tail) ---")
    tail_p = p1[replan_idx:]
    tail_v = v1[replan_idx:]
    tail_f = f1[replan_idx:]
    n_tail = tail_p.shape[0]
    if n_tail < n:
        pad_p = np.tile(tail_p[-1], (n - n_tail, 1))
        pad_v = np.zeros((n - n_tail, 3))
        pad_f = np.tile(tail_f[-1], (n - n_tail, 1))
        tail_p = np.vstack([tail_p, pad_p])
        tail_v = np.vstack([tail_v, pad_v])
        tail_f = np.vstack([tail_f, pad_f])
    tail_v[0] = v_replan_disturbed
    T_warm_guess = T1 * (1 - REPLAN_NODE_FRACTION)
    x0_replan_warm = pack(T_warm_guess, tail_p, tail_v, tail_f)
    result_warm, elapsed_warm, residual_warm = solve_nlp(
        A_force, fj_max, n_fans, p_replan, v_replan_disturbed, P1, x0_replan_warm, n
    )
    print(f"success: {result_warm.success}, solve time: {elapsed_warm:.3f}s, "
          f"residual={residual_warm:.3e}")

    print("\n--- Summary ---")
    print(f"baseline cold solve (full leg):      {elapsed1:.3f}s")
    print(f"replan, cold start:                  {elapsed_cold:.3f}s")
    print(f"replan, warm start (previous tail):  {elapsed_warm:.3f}s")
    if elapsed_cold > 0:
        print(f"warm-start speedup: {elapsed_cold / elapsed_warm:.2f}x")


if __name__ == "__main__":
    main()
