#!/usr/bin/env python3
"""Feasibility check for the MPCC implementation next steps
(docs/2026-08-29_mpcc_implementation_next_steps.md, item "acados/CasADi
導入可否（コンテナ内ビルド確認が最初の一歩）").

Not a unit test (pytest non-collected, no test_ prefix) -- a one-shot probe
that acados + CasADi can actually build and solve an NLP in this container,
and that Real-Time Iteration (RTI, the solve mode Romero et al.'s MPCC uses)
runs fast enough for a 10Hz replanning budget.

Requires acados built from source with the acados_template Python package
installed (not checked into this repo -- built once in scratchpad per
docs/2026-08-29_mpcc_implementation_next_steps.md). Needs ACADOS_SOURCE_DIR
and LD_LIBRARY_PATH pointing at that build before running.

Model here is a translation-only double integrator (state: position,
velocity; input: acceleration), the same simplification the fan-thrust-box
prototype used (docs/archive/achieved/
2026-08-29_fan_thrust_box_constraint_replanning_prototype.md) -- not the full
IntBall2 8-fan/attitude model, just enough to exercise acados's actual solve
path end-to-end.
"""
import os
import time

import numpy as np


def build_solver():
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    px, py, vx, vy = ca.SX.sym("px"), ca.SX.sym("py"), ca.SX.sym("vx"), ca.SX.sym("vy")
    ax, ay = ca.SX.sym("ax"), ca.SX.sym("ay")
    x = ca.vertcat(px, py, vx, vy)
    u = ca.vertcat(ax, ay)

    model = AcadosModel()
    model.name = "double_integrator_2d"
    model.x = x
    model.u = u
    model.f_expl_expr = ca.vertcat(vx, vy, ax, ay)

    ocp = AcadosOcp()
    ocp.model = model

    nx, nu = 4, 2
    ny, ny_e = nx + nu, nx
    N, Tf = 20, 2.0

    ocp.solver_options.N_horizon = N
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 0.1])
    ocp.cost.W_e = np.diag([10.0, 10.0, 1.0, 1.0])
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.yref_e = np.array([1.0, 0.5, 0.0, 0.0])

    a_max = 0.06  # ~IntBall2-scale accel bound, box constraint only
    ocp.constraints.lbu = np.array([-a_max, -a_max])
    ocp.constraints.ubu = np.array([a_max, a_max])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf
    ocp.code_export_directory = "/tmp/acados_feasibility_check_codegen"

    return AcadosOcpSolver(ocp, json_file="/tmp/acados_feasibility_check_ocp.json")


def main():
    if "ACADOS_SOURCE_DIR" not in os.environ:
        raise SystemExit(
            "ACADOS_SOURCE_DIR not set -- point it at the acados build "
            "before running this script (see docs/2026-08-29_mpcc_implementation_next_steps.md)"
        )

    solver = build_solver()

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    n_ticks = 200
    solve_times = []
    for _ in range(n_ticks):
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        t0 = time.perf_counter()
        status = solver.solve()
        solve_times.append(time.perf_counter() - t0)
        x0 = solver.get(1, "x")

    solve_times = np.array(solve_times[5:])  # drop JIT/cache warmup ticks
    replan_budget_s = 0.1  # guidance.replan_rate_hz=10.0 in config/gnc_params.yaml
    print(f"RTI solve time over {len(solve_times)} ticks (N=20, nx=4, nu=2):")
    print(
        f"  mean={solve_times.mean()*1000:.3f}ms  "
        f"max={solve_times.max()*1000:.3f}ms  "
        f"min={solve_times.min()*1000:.3f}ms"
    )
    print(f"  10Hz replan budget ({replan_budget_s*1000:.0f}ms): "
          f"{'OK' if solve_times.max() < replan_budget_s else 'EXCEEDED'}")
    print(f"  final solver status={status} (0 == converged)")


if __name__ == "__main__":
    main()
