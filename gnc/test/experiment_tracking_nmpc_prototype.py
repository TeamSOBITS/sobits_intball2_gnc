#!/usr/bin/env python3
"""Progress-free tracking NMPC prototype (docs/2026-08-29_progress_free_nmpc_plan.md).

Same P0->P1 scenario, disturbance injection, and fan-thrust box constraint
(ThrustAllocator.A) as test/experiment_mpcc_translation_prototype.py, but with
NO theta/vtheta progress state and NO contour/lag decomposition. Instead the
state is just [p, v] and the cost tracks a fixed, absolute-time-parameterized
reference r(t), v(t) (a quintic min-jerk blend from P0 to P1, held at P1
after it settles) supplied to the solver as an acados stage parameter.

This tests the hypothesis from docs/2026-08-29_progress_free_nmpc_plan.md:
the MPCC "progress stall" (docs/2026-08-29_mpcc_implementation_next_steps.md)
comes from theta/vtheta having no restoring force back toward progress once
the lag-cost gradient flips sign. A tracking NMPC has no such state -- the
reference position at time t is what it is regardless of how far behind the
vehicle has fallen -- so progress stall should not be able to occur here by
construction. What's NOT guaranteed to survive the change: MPCC's other
strength (no T-blowup/infeasibility under disturbance) relied on theta/vtheta
being continuous states with no discrete reallocation; this design also has
no discrete reallocation (t_k is just wall/sim clock), so that property is
expected to carry over, but is checked below rather than assumed.

Not a pytest test (no test_ prefix) -- standalone experiment:
    ACADOS_SOURCE_DIR=<path> LD_LIBRARY_PATH=<path>/lib \
        python3 test/experiment_tracking_nmpc_prototype.py [--disturbance weak|strong]
"""
import argparse
import os
import time

import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
DT_TICK = 0.1  # guidance.replan_rate_hz = 10.0
REPLAN_BUDGET_S = DT_TICK

P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

N_TICKS = 40
DISTURB_TICK = 8
V_PERP_WEAK = 0.05
V_PERP_STRONG = 0.4

N_HORIZON = 20
TF_HORIZON = 2.0  # same fixed receding horizon as the MPCC prototype

V_PEAK_TARGET = 0.3  # same order as the MPCC prototype's VTHETA_MAX, so the
                      # two designs are chasing a comparable nominal speed

W_POS = 5e2
W_VEL = 5e2
W_F = 1e-3

W_POS_TERM = 1e2
W_VEL_TERM = 5.0


def build_path_frame(p0, p1):
    """Only used to pick a lateral direction for disturbance injection --
    the tracking cost itself has no path-local frame."""
    direction = p1 - p0
    length = np.linalg.norm(direction)
    direction = direction / length
    n1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(n1) < 1e-6:
        n1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    n1 = n1 / np.linalg.norm(n1)
    return direction, length, n1


def quintic_reference(t, total_time, p0, p1):
    """Min-jerk position/velocity blend from p0 to p1, held at p1 (v=0)
    once t >= total_time -- so a fixed t_k always resolves to a well-defined
    reference, including "we're late, target is just sitting at the goal"."""
    if t <= 0.0:
        return p0.copy(), np.zeros(3)
    if t >= total_time:
        return p1.copy(), np.zeros(3)
    tau = t / total_time
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sdot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / total_time
    delta = p1 - p0
    return p0 + s * delta, sdot * delta


def build_solver(A_force, fj_max, n_fans, tag):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    p = ca.SX.sym("p", 3)
    v = ca.SX.sym("v", 3)
    x = ca.vertcat(p, v)

    f = ca.SX.sym("f", n_fans)
    u = f

    ref = ca.SX.sym("ref", 6)  # [r_ref(3), v_ref(3)], set per-stage before each solve
    r_ref = ref[0:3]
    v_ref = ref[3:6]

    A_force_ca = ca.DM(A_force)
    accel = (A_force_ca @ f) / MASS

    model = AcadosModel()
    model.name = "tracking_nmpc_prototype"
    model.x = x
    model.u = u
    model.p = ref
    model.f_expl_expr = ca.vertcat(v, accel)

    nx = 6
    nu = n_fans

    stage_cost = (
        W_POS * ca.sumsqr(p - r_ref)
        + W_VEL * ca.sumsqr(v - v_ref)
        + W_F * ca.sumsqr(f)
    )
    terminal_cost = W_POS_TERM * ca.sumsqr(p - r_ref) + W_VEL_TERM * ca.sumsqr(v - v_ref)

    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbu = np.zeros(n_fans)
    ocp.constraints.ubu = np.full(n_fans, fj_max)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(nx)

    ocp.parameter_values = np.zeros(6)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = TF_HORIZON
    # tag keeps concurrent runs (e.g. weak/strong in parallel) from racing on
    # the same codegen dir -- a shared dir caused "file too short" dlopen
    # errors when two builds wrote the .so simultaneously.
    ocp.code_export_directory = f"/tmp/acados_tracking_nmpc_prototype_codegen_{tag}"

    return AcadosOcpSolver(ocp, json_file=f"/tmp/acados_tracking_nmpc_prototype_ocp_{tag}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturbance", choices=["weak", "strong"], default="strong")
    parser.add_argument("--ticks", type=int, default=N_TICKS,
                         help="override N_TICKS, e.g. to check long-horizon recovery "
                              "under an actuator-limited disturbance")
    args = parser.parse_args()
    n_ticks = args.ticks
    v_perp_error = V_PERP_WEAK if args.disturbance == "weak" else V_PERP_STRONG

    if "ACADOS_SOURCE_DIR" not in os.environ:
        raise SystemExit(
            "ACADOS_SOURCE_DIR not set -- point it at the acados build "
            "before running this script (see docs/2026-08-29_mpcc_implementation_next_steps.md)"
        )

    allocator = ThrustAllocator()
    A_force = allocator.A[:3, :]
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count

    direction, path_length, lateral_dir = build_path_frame(P0, P1)
    total_time = 1.875 * path_length / V_PEAK_TARGET
    solver = build_solver(A_force, fj_max, n_fans, tag=args.disturbance)

    dt_stage = TF_HORIZON / N_HORIZON

    p_true = P0.copy()
    v_true = np.zeros(3)

    print(f"path length={path_length*1000:.1f}mm, total_time={total_time:.2f}s, "
          f"disturbance={args.disturbance} (+{v_perp_error} m/s lateral at tick {DISTURB_TICK})")
    print(f"{'tick':>4} {'solve_s':>9} {'over_budget':>12} {'status':>7} "
          f"{'pos_err_mm':>11} {'dist_to_goal_mm':>16}")

    over_budget_ticks = []
    infeasible_ticks = []
    solve_times = []

    for tick in range(n_ticks):
        if tick == DISTURB_TICK:
            v_true = v_true + v_perp_error * lateral_dir
            print(f"--- disturbance injected at tick {tick}: "
                  f"+{v_perp_error} m/s lateral ---")

        t_now = tick * DT_TICK

        x0 = np.concatenate([p_true, v_true])
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        for k in range(N_HORIZON + 1):
            t_k = t_now + k * dt_stage
            r_ref_k, v_ref_k = quintic_reference(t_k, total_time, P0, P1)
            solver.set(k, "p", np.concatenate([r_ref_k, v_ref_k]))

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        solve_times.append(elapsed)

        u0 = solver.get(0, "u")
        f0 = u0[:n_fans]
        a0 = (A_force @ f0) / MASS

        r_ref_now, v_ref_now = quintic_reference(t_now, total_time, P0, P1)
        pos_err_mm = np.linalg.norm(p_true - r_ref_now) * 1000
        dist_to_goal_mm = np.linalg.norm(P1 - p_true) * 1000

        over_budget = elapsed > REPLAN_BUDGET_S
        if over_budget:
            over_budget_ticks.append(tick)
        if status != 0:
            infeasible_ticks.append(tick)

        print(f"{tick:>4} {elapsed:>9.4f} {str(over_budget):>12} {status:>7} "
              f"{pos_err_mm:>11.2f} {dist_to_goal_mm:>16.2f} "
              f"|f0|={np.linalg.norm(f0):.4f} v=({v_true[0]:.4f},{v_true[1]:.4f},{v_true[2]:.4f})")

        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK

    solve_times = np.array(solve_times)
    print("\n--- Summary ---")
    print(f"ticks over {REPLAN_BUDGET_S*1000:.0f}ms budget: {over_budget_ticks}")
    print(f"ticks with solver status != 0 (infeasible/failed): {infeasible_ticks}")
    print(f"solve time: mean={solve_times.mean()*1000:.3f}ms "
          f"max={solve_times.max()*1000:.3f}ms")
    print(f"final distance to goal: {np.linalg.norm(P1 - p_true)*1000:.2f} mm")


if __name__ == "__main__":
    main()
