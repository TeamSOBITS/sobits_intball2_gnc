#!/usr/bin/env python3
"""MPCC basic-form prototype (docs/2026-08-29_mpcc_implementation_next_steps.md,
"次の一歩": 並進のみでcontouring/lag+RTIを再現し、T急拡大問題が本当に解消するか検証).

Same P0->P1 scenario, disturbance injection, and fan-thrust box constraint
(ThrustAllocator.A) as test/experiment_replanning_closed_loop.py, but replaces
the "fixed-T QP feasibility + T-update rule" design with an actual MPCC-style
formulation, solved with acados SQP_RTI:

  - State is augmented with progress theta (arc length along the P0->P1
    line) and progress speed vtheta -- NOT a discretely-reallocated total
    time T. There is nothing to make "infeasible" the way t_min_perp > t_max
    made HeuristicSegmentTimeAllocator infeasible.
  - Cost decomposes the position error into contour error (perpendicular to
    the path) and lag error (along the path), in a path-fixed local frame --
    the actual Romero et al. contouring/lag idea (see docs/2026-08-29_mpcc_
    implementation_next_steps.md item 1), rather than the fixed-T collocation
    used previously.
  - Progress is rewarded with a genuine linear "-mu*vtheta" term (Romero et
    al.'s actual formulation), via acados EXTERNAL cost -- an earlier attempt
    approximated this as a LINEAR_LS quadratic-tracking-to-a-fixed-speed term,
    which let theta race to that speed regardless of whether real progress
    could keep up; see "並進のみMPCC基本形プロトタイプ 結果" in the next-steps
    doc for what that broke and why this version replaces it.
  - Solved with SQP_RTI (Real-Time Iteration, item 3) at a *fixed* receding
    horizon (Tf, N constant) -- this is the structural difference from the
    fixed-T one-shot solves: there is no total-duration variable to blow up.

What this script does NOT include (still open per the next-steps doc):
  - Attitude/torque (item 2, translation only).
  - True anti-chattering cost on delta-f between ticks (item 5) -- this
    prototype regularizes f magnitude only, not f[k]-f_prev_commanded.
  - Any tuning beyond "does it run/stay feasible/stay in budget."

Not a pytest test (no test_ prefix) -- standalone experiment:
    ACADOS_SOURCE_DIR=<path> LD_LIBRARY_PATH=<path>/lib \
        python3 test/experiment_mpcc_translation_prototype.py [--disturbance weak|strong]
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
TF_HORIZON = 2.0  # fixed receding horizon, s -- NOT a total-trajectory-duration variable.
                  # Tried widening to Tf=6.0/N=60 on the theory that a short horizon
                  # can't see the ~12.5s settling time of a 0.4 m/s v_perp correction
                  # (LP-verified ~0.032 m/s^2 max lateral accel) -- overshoot did not
                  # improve (if anything, slightly worse) and solve time went from
                  # ~1ms to ~4-7ms for no benefit, so reverted. The overshoot is not
                  # a horizon-length problem.

VTHETA_MAX = 0.3  # generous upper bound -- the *lag* cost, not this bound, is
                  # what should keep theta from outrunning real progress
ATHETA_MAX = 0.05

W_CONTOUR = 5e2
W_LAG = 5e2  # now the *saturated* (pseudo-Huber) lag weight, not a quadratic
             # one -- see LAG_HUBER_DELTA and its use in build_solver (option 3,
             # docs/2026-08-29_mpcc_progress_stall_fix_plan.md). A plain
             # quadratic e_local[2]**2 here is what caused progress stall: its
             # marginal cost 2*W_LAG*lag grows without bound, so it always
             # eventually outbids any fixed progress reward once lag crosses
             # mu/(2*W_LAG), pinning vtheta (or, after option 1, real velocity)
             # at whatever boundary is nearest with no restoring force back.
LAG_HUBER_DELTA = 0.05  # m -- lag magnitude beyond which the saturated cost's
             # marginal growth flattens toward W_LAG*LAG_HUBER_DELTA (constant),
             # instead of continuing to grow linearly in lag forever. Chosen
             # as a "few cm is acceptable, meters is not" threshold; not yet
             # tuned against data.
W_V_LATERAL = 3e3  # heavy: kill v_perp-type disturbances before they become
             # position error. Path-frame lateral velocity (perpendicular to
             # the fixed P0->P1 direction) -- option 2 (goal-direction-relative
             # version) was tried and rejected: closing_rate's 1/|P1-p| term
             # made |v|^2-closing_rate^2 non-convex in p, which combined with
             # SQP_RTI (1 Newton step/tick) + hessian_approx=EXACT to blow up
             # (48.7m final distance, 722/1000 ticks infeasible from tick 278
             # onward, see plan doc "案2...却下"). Reverted to this form.
W_V_ALONG = 50.0  # raised from 1.0 (oscillation follow-up after option 3):
             # the residual overshoot is an along-path speed that swings sign
             # repeatedly (a0_along_path/v_along oscillate) even after theta
             # stalls, and a direction-agnostic |v|^2 brake (W_V_DAMP, tried
             # and reverted above) didn't touch it. v_local[2] is already
             # exactly this along-path speed and is linear in v (no 1/dist
             # term like closing_rate), so raising its weight targets the
             # oscillating DOF directly without reintroducing the
             # non-convexity that broke option 2.
MU_PROGRESS = 1.0  # progress reward coefficient (docs/2026-08-29_mpcc_progress_
                    # stall_fix_plan.md option 1: reward is now the real closing
                    # rate toward P1, not vtheta -- see build_solver). Raising
                    # this to 100.0 (stall fix) and lowering it to 0.3
                    # (overshoot follow-up, on the theory that a lower peak
                    # closing speed driven by this reward would shrink the
                    # v^2/(2*a_max) stopping distance) were both tried and
                    # both had *zero* measurable effect on the trajectory --
                    # because |f0| sits at fj_max (thrust-saturated) through
                    # the disturbance-response phase, driven by W_CONTOUR/
                    # W_LAG/W_V_LATERAL. MU_PROGRESS's marginal cost is
                    # irrelevant once actuators are already maxed out
                    # responding to bigger terms; the peak along-path speed
                    # that drives the overshoot is a geometric byproduct of
                    # killing the 0.4 m/s lateral disturbance, not something
                    # this reward controls. Left at 1.0 (no evidence to move
                    # it either way).
W_F = 1e-3
W_ATHETA = 1e-2
# A stage-wise |v|^2 "W_V_DAMP" damping term (docs/2026-08-29_mpcc_progress_
# stall_fix_plan.md oscillation follow-up) was tried at weight 5.0 against
# the residual overshoot left by option 3 (dist-to-goal swinging 265mm<->
# 3900mm+ over the 1000-tick strong run) and reverted: it barely changed the
# oscillation (169mm vs 165mm final distance, same swing pattern) while
# introducing 2 new transient-infeasible ticks. Whatever is driving the
# oscillation, a direction-agnostic brake on total speed isn't the lever.

W_E_TERM = 1e2
W_V_TERM = 5.0


def build_path_frame(p0, p1):
    direction = p1 - p0
    length = np.linalg.norm(direction)
    direction = direction / length
    n1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(n1) < 1e-6:
        n1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(direction, n1)
    # columns: [n1, n2, direction] -- last row of R^T @ e is the lag error
    R = np.column_stack([n1, n2, direction])
    return direction, length, R


def build_solver(A_force, fj_max, n_fans, R, path_length):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    p = ca.SX.sym("p", 3)
    v = ca.SX.sym("v", 3)
    theta = ca.SX.sym("theta")
    vtheta = ca.SX.sym("vtheta")
    x = ca.vertcat(p, v, theta, vtheta)

    f = ca.SX.sym("f", n_fans)
    atheta = ca.SX.sym("atheta")
    u = ca.vertcat(f, atheta)

    A_force_ca = ca.DM(A_force)
    accel = (A_force_ca @ f) / MASS

    model = AcadosModel()
    model.name = "mpcc_translation_prototype"
    model.x = x
    model.u = u
    model.f_expl_expr = ca.vertcat(v, accel, vtheta, atheta)

    nx = 8
    nu = n_fans + 1
    direction = ca.DM(R[:, 2])
    Rt = ca.DM(R.T)

    # e_local = R^T @ (p - P0 - theta*direction): contour (perpendicular,
    # rows 0:2) + lag (along-path, row 2) error, computed directly as a
    # CasADi expression -- avoids the LINEAR_LS "residual = Vx@x + Vu@u -
    # yref" bookkeeping (a first attempt mismatched the constant -P0 term
    # between Vx and yref and silently broke the cost's coupling to u; a
    # second attempt approximated Romero et al.'s linear "-mu*vtheta" progress
    # reward as a LINEAR_LS quadratic-tracking-to-a-fixed-speed term, which
    # let theta race to that speed regardless of whether real progress could
    # keep up). EXTERNAL cost lets both be expressed as what they actually
    # are: a quadratic contour/lag/velocity/effort cost plus a genuine linear
    # progress reward.
    e = p - ca.DM(P0) - theta * direction
    e_local = Rt @ e
    v_local = Rt @ v

    # Progress reward tied to the real closing rate toward P1 (-d/dt|p-P1|),
    # not to vtheta directly (docs/2026-08-29_mpcc_progress_stall_fix_plan.md
    # option 1). The old "-mu*vtheta" term paid out even when theta outran
    # real p (spinning up vtheta cost nothing once lag's marginal cost
    # exceeded mu), which is what let vtheta pin at its lower bound forever
    # once that crossover happened (progress stall, see next-steps doc). This
    # version only pays when p is actually approaching P1; theta now has no
    # direct incentive of its own and is pulled along solely by the lag term
    # wanting theta to track p's real path-projection.
    to_goal = ca.DM(P1) - p
    dist_to_goal = ca.sqrt(ca.sumsqr(to_goal) + 1e-6)
    closing_rate = ca.dot(to_goal, v) / dist_to_goal

    # Tried decaying the progress reward as the goal is approached (mu *
    # tanh(dist_to_goal/D) instead of flat mu) to see if it would make the
    # MPC start braking earlier and avoid the ~54mm-closest-approach-then-
    # 1.7m-rebound overshoot (docs/2026-08-29_mpcc_progress_stall_fix_plan.md
    # "overshoot follow-up"). No measurable effect (identical trajectory to
    # 4 decimal places) -- reverted. The overshoot turned out not to be
    # driven by the progress reward at all: it is the terminal cost
    # (W_E_TERM/W_V_TERM) trying to zero p-P1 and v within the Tf=2.0s
    # horizon, and LP-verified along-path accel (~0.03-0.046 m/s^2) simply
    # cannot arrest a ~0.33 m/s closing speed in 2s -- an actuator/horizon
    # limit, not a reward-shape one.

    # Saturated (pseudo-Huber) lag cost (option 3, docs/2026-08-29_mpcc_
    # progress_stall_fix_plan.md): quadratic for small lag, but its marginal
    # cost flattens toward a constant (~W_LAG*LAG_HUBER_DELTA) instead of
    # growing without bound. The plain-quadratic version's unbounded marginal
    # cost is what caused progress stall -- once lag grew past mu/(2*W_LAG),
    # advancing further always looked worse, with nothing to push back. A
    # capped marginal cost means a large-enough lag can no longer out-bid a
    # fixed progress reward, in principle. Pseudo-Huber form (fully smooth,
    # C-infinity, still convex in its argument) rather than the classic
    # piecewise Huber, so composed with the affine e_local[2] it stays convex
    # in (p, theta) -- avoiding the non-convexity that broke option 2's
    # EXACT-Hessian/SQP_RTI combination.
    lag_cost = W_LAG * LAG_HUBER_DELTA ** 2 * (
        ca.sqrt(1.0 + (e_local[2] / LAG_HUBER_DELTA) ** 2) - 1.0
    )

    stage_cost = (
        W_CONTOUR * ca.sumsqr(e_local[0:2])
        + lag_cost
        + W_V_LATERAL * ca.sumsqr(v_local[0:2])
        + W_V_ALONG * v_local[2] ** 2
        - MU_PROGRESS * closing_rate
        + W_F * ca.sumsqr(f)
        + W_ATHETA * atheta ** 2
    )
    terminal_cost = W_E_TERM * ca.sumsqr(p - ca.DM(P1)) + W_V_TERM * ca.sumsqr(v)

    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbu = np.concatenate([np.zeros(n_fans), [-ATHETA_MAX]])
    ocp.constraints.ubu = np.concatenate([np.full(n_fans, fj_max), [ATHETA_MAX]])
    ocp.constraints.idxbu = np.arange(nu)
    # vtheta >= 0 (progress must not run backward -- also the physically
    # sensible choice). Tried relaxing this to -0.02 on the theory that the
    # hard floor at 0 was trapping the solution at that corner; it wasn't the
    # bound *value* -- vtheta just relocated to and pinned at the new -0.02
    # corner instead, reproducing the identical stall. So the "progress
    # stall" (see next-steps doc) is not a boundary-tuning problem: whichever
    # bound is nearest becomes a self-reinforcing trap once the marginal cost
    # of advancing theta (2*W_LAG*lag) exceeds MU_PROGRESS, because nothing in
    # this cost structure creates a restoring force back toward progress once
    # that crossover happens. A real fix needs a different mechanism, not
    # another bound/weight guess.
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([VTHETA_MAX])
    ocp.constraints.idxbx = np.array([7])  # vtheta >= 0, <= VTHETA_MAX
    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"  # EXTERNAL cost; the closing-rate
    # progress term is nonlinear in p (division by dist_to_goal), so this is
    # the true (non-constant) Hessian via CasADi AD, not a GN approximation
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = TF_HORIZON
    ocp.code_export_directory = "/tmp/acados_mpcc_translation_prototype_codegen"

    return AcadosOcpSolver(ocp, json_file="/tmp/acados_mpcc_translation_prototype_ocp.json")


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

    direction, path_length, R = build_path_frame(P0, P1)
    solver = build_solver(A_force, fj_max, n_fans, R, path_length)

    p_true = P0.copy()
    v_true = np.zeros(3)
    theta_true = 0.0
    vtheta_true = 0.0

    print(f"path length={path_length*1000:.1f}mm, disturbance={args.disturbance} "
          f"(+{v_perp_error} m/s lateral at tick {DISTURB_TICK})")
    print(f"{'tick':>4} {'solve_s':>9} {'over_budget':>12} {'status':>7} "
          f"{'contour_mm':>11} {'lag_mm':>8} {'dist_to_goal_mm':>16}")

    over_budget_ticks = []
    infeasible_ticks = []
    solve_times = []

    for tick in range(n_ticks):
        if tick == DISTURB_TICK:
            v_true = v_true + v_perp_error * R[:, 0]
            print(f"--- disturbance injected at tick {tick}: "
                  f"+{v_perp_error} m/s lateral ---")

        x0 = np.concatenate([p_true, v_true, [theta_true], [vtheta_true]])
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        solve_times.append(elapsed)

        u0 = solver.get(0, "u")
        f0, atheta0 = u0[:n_fans], u0[n_fans]
        a0 = (A_force @ f0) / MASS

        e = p_true - P0 - theta_true * direction
        e_local = R.T @ e
        contour_mm = np.linalg.norm(e_local[:2]) * 1000
        lag_mm = e_local[2] * 1000
        dist_to_goal_mm = np.linalg.norm(P1 - p_true) * 1000

        over_budget = elapsed > REPLAN_BUDGET_S
        if over_budget:
            over_budget_ticks.append(tick)
        if status != 0:
            infeasible_ticks.append(tick)

        a0_along = np.dot(a0, direction)
        v_local = R.T @ v_true
        print(f"{tick:>4} {elapsed:>9.4f} {str(over_budget):>12} {status:>7} "
              f"{contour_mm:>11.2f} {lag_mm:>8.2f} {dist_to_goal_mm:>16.2f} "
              f"|f0|={np.linalg.norm(f0):.4f} vtheta={vtheta_true:.4f} a0_along_path={a0_along:.5f} "
              f"v_lat=({v_local[0]:.4f},{v_local[1]:.4f}) v_along={v_local[2]:.4f}")

        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK
        vtheta_true = vtheta_true + atheta0 * DT_TICK
        theta_true = theta_true + vtheta_true * DT_TICK

    solve_times = np.array(solve_times)
    print("\n--- Summary ---")
    print(f"ticks over {REPLAN_BUDGET_S*1000:.0f}ms budget: {over_budget_ticks}")
    print(f"ticks with solver status != 0 (infeasible/failed): {infeasible_ticks}")
    print(f"solve time: mean={solve_times.mean()*1000:.3f}ms "
          f"max={solve_times.max()*1000:.3f}ms")
    print(f"final distance to goal: {np.linalg.norm(P1 - p_true)*1000:.2f} mm")


if __name__ == "__main__":
    main()
