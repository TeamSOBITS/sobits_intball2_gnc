#!/usr/bin/env python3
"""MPCC + attitude integration, step 1 (docs/2026-08-29_mpcc_attitude_torque_
integration_plan.md, "検証計画" item 1): duplicates
test/experiment_mpcc_translation_prototype.py (left unmodified) and adds
attitude as a *progress-free* tracking cost -- no attitude theta/vtheta, just
a fixed target quaternion the vehicle chases throughout. Goal is only to see
whether attitude cost sharing the same 8-fan thrust budget as translation
disturbs the already-validated translation MPCC (thrust saturation,
progress stall, oscillation) -- see plan doc item 2.

Dynamics extension (plan doc item 1): ``ThrustAllocator.A`` is already 6x8
(rows 0:3 force, rows 3:6 torque, see thrust_allocator.py's ``self.A``
construction), so unlike the translation-only prototype (which sliced
``A[:3, :]``), this uses the full matrix: ``wrench = A_full @ f`` feeds both
the translation accel and the angular accel from the *same* per-fan thrusts.

Angular dynamics (plan doc item 4): inertia is isotropic (0.0136 kg*m^2,
config/gnc_params.yaml trajectory_controller.inertia), so the Euler equation's
gyroscopic cross term ``w x (I*w)`` is identically zero (w x w = 0 for a
scalar-multiple-of-identity I) -- angular acceleration is just
``torque / INERTIA``, no nonlinear coupling to model.

Quaternion convention: [x, y, z, w] Hamilton, matching
control/utils/quat_math.py (``quat_mul``/``quat_conj``/``geodesic_angle``).
Attitude error is computed as the vector part of ``q_target^-1 (x) q``
(body-frame small-angle proxy) -- a known, documented simplification (plan
doc item 5's "非凸性/クォータニオン誤差表現" caution): this is only valid near
the target (no wraparound handling), so the target's antipodal sign is
resolved once against the initial attitude before the solver is built, and
this prototype does not attempt a genuine geodesic (e.g. angle-axis /
tangent-space) attitude cost -- fine for "does progress-free tracking even
coexist with the translation MPCC" but would need revisiting for a target
attitude far from the initial one, or path-following (plan doc item 3,
explicitly deferred to a later step).

No unit-quaternion-norm constraint is enforced in the OCP (plan doc item 5)
-- q is left to drift under ERK integration within the horizon; the *true*
simulated state is renormalized every tick (see ``main()``) so open-loop
drift doesn't compound across ticks, but within-horizon drift is unchecked.

Not a pytest test (no test_ prefix) -- standalone experiment:
    ACADOS_SOURCE_DIR=<path> LD_LIBRARY_PATH=<path>/lib \
        python3 test/experiment_mpcc_attitude_tracking_prototype.py [--disturbance weak|strong]
"""
import argparse
import os
import time

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
INERTIA = 0.0136  # kg*m^2, isotropic, trajectory_controller.inertia
DT_TICK = 0.1  # guidance.replan_rate_hz = 10.0
REPLAN_BUDGET_S = DT_TICK

P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

# Target attitude: 30deg about world +z from identity -- a modest, arbitrary
# rotation (not tied to any real mission waypoint), just enough to give the
# attitude cost something nonzero to track while translation is under way.
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
_half = np.radians(30.0) / 2.0
Q_TARGET = np.array([0.0, 0.0, np.sin(_half), np.cos(_half)])

N_TICKS = 40
DISTURB_TICK = 8
V_PERP_WEAK = 0.05
V_PERP_STRONG = 0.4
# Attitude disturbance (angular velocity kick, mirrors the translation
# lateral-velocity kick) injected at the same tick -- exercises the shared
# 8-fan budget under simultaneous translation + attitude upset, per plan doc
# item 2's saturation concern.
W_DISTURB_WEAK = np.array([0.05, 0.0, 0.0])
W_DISTURB_STRONG = np.array([0.3, 0.0, 0.0])

N_HORIZON = 20
TF_HORIZON = 2.0  # unchanged from translation-only prototype

VTHETA_MAX = 0.3
ATHETA_MAX = 0.05

W_CONTOUR = 5e2
W_LAG = 5e2
LAG_HUBER_DELTA = 0.05
W_V_LATERAL = 3e3
W_V_ALONG = 50.0
MU_PROGRESS = 1.0
W_F = 1e-3
W_ATHETA = 1e-2

W_E_TERM = 1e2
W_V_TERM = 5.0

# Attitude tracking weights -- untuned (plan doc: "検証計画" step 1 is about
# whether the two costs coexist at all, not about tuning either one).
W_ATT = 1e2
W_W = 1.0
W_ATT_TERM = 2e2
W_W_TERM = 2.0


def build_path_frame(p0, p1):
    direction = p1 - p0
    length = np.linalg.norm(direction)
    direction = direction / length
    n1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(n1) < 1e-6:
        n1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(direction, n1)
    R = np.column_stack([n1, n2, direction])
    return direction, length, R


def resolve_target_hemisphere(q0, q_target):
    """Flip ``q_target``'s sign if needed so it's on the same hemisphere as
    ``q0`` (shortest-arc side) -- see module docstring's quaternion-error
    caveat: the error cost below has no other wraparound handling."""
    if np.dot(q0, q_target) < 0.0:
        return -np.asarray(q_target, dtype=float)
    return np.asarray(q_target, dtype=float)


def build_solver(A_full, fj_max, n_fans, R, q_target):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    p = ca.SX.sym("p", 3)
    v = ca.SX.sym("v", 3)
    theta = ca.SX.sym("theta")
    vtheta = ca.SX.sym("vtheta")
    q = ca.SX.sym("q", 4)
    w = ca.SX.sym("w", 3)
    x = ca.vertcat(p, v, theta, vtheta, q, w)

    f = ca.SX.sym("f", n_fans)
    atheta = ca.SX.sym("atheta")
    u = ca.vertcat(f, atheta)

    A_full_ca = ca.DM(A_full)
    wrench = A_full_ca @ f
    accel = wrench[0:3] / MASS
    torque = wrench[3:6]
    wdot = torque / INERTIA  # isotropic inertia -> no w x (I*w) term, see module docstring

    # qdot = 0.5 * q (x) [w; 0], Hamilton [x,y,z,w] product, w in body frame.
    qw = ca.vertcat(w, 0.0)
    qdot = 0.5 * ca.vertcat(
        q[3] * qw[0] + q[0] * qw[3] + q[1] * qw[2] - q[2] * qw[1],
        q[3] * qw[1] - q[0] * qw[2] + q[1] * qw[3] + q[2] * qw[0],
        q[3] * qw[2] + q[0] * qw[1] - q[1] * qw[0] + q[2] * qw[3],
        q[3] * qw[3] - q[0] * qw[0] - q[1] * qw[1] - q[2] * qw[2],
    )

    model = AcadosModel()
    model.name = "mpcc_attitude_tracking_prototype"
    model.x = x
    model.u = u
    model.f_expl_expr = ca.vertcat(v, accel, vtheta, atheta, qdot, wdot)

    nx = 15
    nu = n_fans + 1
    direction = ca.DM(R[:, 2])
    Rt = ca.DM(R.T)

    e = p - ca.DM(P0) - theta * direction
    e_local = Rt @ e
    v_local = Rt @ v

    to_goal = ca.DM(P1) - p
    dist_to_goal = ca.sqrt(ca.sumsqr(to_goal) + 1e-6)
    closing_rate = ca.dot(to_goal, v) / dist_to_goal

    lag_cost = W_LAG * LAG_HUBER_DELTA ** 2 * (
        ca.sqrt(1.0 + (e_local[2] / LAG_HUBER_DELTA) ** 2) - 1.0
    )

    # Attitude error: vector part of q_target^-1 (x) q (see module docstring
    # for the small-angle / no-wraparound caveat).
    q_target_dm = ca.DM(q_target)
    q_target_conj = ca.vertcat(-q_target_dm[0], -q_target_dm[1], -q_target_dm[2], q_target_dm[3])
    q_err = ca.vertcat(
        q_target_conj[3] * q[0] + q_target_conj[0] * q[3] + q_target_conj[1] * q[2] - q_target_conj[2] * q[1],
        q_target_conj[3] * q[1] - q_target_conj[0] * q[2] + q_target_conj[1] * q[3] + q_target_conj[2] * q[0],
        q_target_conj[3] * q[2] + q_target_conj[0] * q[1] - q_target_conj[1] * q[0] + q_target_conj[2] * q[3],
        q_target_conj[3] * q[3] - q_target_conj[0] * q[0] - q_target_conj[1] * q[1] - q_target_conj[2] * q[2],
    )
    att_cost = W_ATT * ca.sumsqr(q_err[0:3]) + W_W * ca.sumsqr(w)

    stage_cost = (
        W_CONTOUR * ca.sumsqr(e_local[0:2])
        + lag_cost
        + W_V_LATERAL * ca.sumsqr(v_local[0:2])
        + W_V_ALONG * v_local[2] ** 2
        - MU_PROGRESS * closing_rate
        + W_F * ca.sumsqr(f)
        + W_ATHETA * atheta ** 2
        + att_cost
    )
    terminal_cost = (
        W_E_TERM * ca.sumsqr(p - ca.DM(P1)) + W_V_TERM * ca.sumsqr(v)
        + W_ATT_TERM * ca.sumsqr(q_err[0:3]) + W_W_TERM * ca.sumsqr(w)
    )

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

    # Explicit achievable-wrench envelope (guidance/utils/actuation_envelope.py)
    # was tried here as a general linear stage constraint (F_env @ (A_full @ f)
    # <= g_env) on the theory that it might change solver conditioning versus
    # the per-fan box alone. Reverted: wrench_envelope_halfspaces returns
    # ~9951 halfspace facets for this 8-fan geometry (not a small number --
    # confirmed by direct measurement), and adding that many general
    # constraints per stage (x N_HORIZON) made the SQP take minutes per
    # solver.solve() call, unusable at any replan rate. This reproduces the
    # exact same finding already on record for the SCP prototype (docs/
    # 2026-08-29_guidance_dir_and_dead_code_survey.md, SCP row: "本物のwrench
    # envelope（1万面ポリトープ）は正確だが88秒〜で遅すぎた") -- should have
    # checked that note before trying this here. It is also mathematically
    # redundant with the per-fan box above: {A_full @ f : 0 <= f <= fj_max}
    # IS the set F_env @ w <= g_env describes, and wrench is computed
    # directly as A_full @ f in the dynamics, so there is no independent
    # per-axis force/torque box for the envelope to correct (unlike
    # ToppraTrajectory's original bug that motivated actuation_envelope.py).
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([VTHETA_MAX])
    ocp.constraints.idxbx = np.array([7])  # vtheta >= 0, <= VTHETA_MAX
    ocp.constraints.x0 = np.concatenate([np.zeros(8), Q0, np.zeros(3)])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = TF_HORIZON
    ocp.code_export_directory = "/tmp/acados_mpcc_attitude_tracking_prototype_codegen"

    return AcadosOcpSolver(ocp, json_file="/tmp/acados_mpcc_attitude_tracking_prototype_ocp.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturbance", choices=["weak", "strong"], default="strong")
    parser.add_argument("--ticks", type=int, default=N_TICKS)
    args = parser.parse_args()
    n_ticks = args.ticks
    v_perp_error = V_PERP_WEAK if args.disturbance == "weak" else V_PERP_STRONG
    w_disturb = W_DISTURB_WEAK if args.disturbance == "weak" else W_DISTURB_STRONG

    if "ACADOS_SOURCE_DIR" not in os.environ:
        raise SystemExit(
            "ACADOS_SOURCE_DIR not set -- point it at the acados build "
            "before running this script (see docs/2026-08-29_mpcc_implementation_next_steps.md)"
        )

    allocator = ThrustAllocator()
    A_full = allocator.A
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count

    direction, path_length, R = build_path_frame(P0, P1)
    q_target = resolve_target_hemisphere(Q0, Q_TARGET)
    solver = build_solver(A_full, fj_max, n_fans, R, q_target)

    p_true = P0.copy()
    v_true = np.zeros(3)
    theta_true = 0.0
    vtheta_true = 0.0
    q_true = Q0.copy()
    w_true = np.zeros(3)

    print(f"path length={path_length*1000:.1f}mm, disturbance={args.disturbance} "
          f"(+{v_perp_error} m/s lateral, +{w_disturb} rad/s angular at tick {DISTURB_TICK})")
    print(f"{'tick':>4} {'solve_s':>9} {'over_budget':>12} {'status':>7} "
          f"{'contour_mm':>11} {'lag_mm':>8} {'dist_to_goal_mm':>16} {'att_err_deg':>12}")

    over_budget_ticks = []
    infeasible_ticks = []
    solve_times = []

    for tick in range(n_ticks):
        if tick == DISTURB_TICK:
            v_true = v_true + v_perp_error * R[:, 0]
            w_true = w_true + w_disturb
            print(f"--- disturbance injected at tick {tick}: "
                  f"+{v_perp_error} m/s lateral, +{w_disturb} rad/s angular ---")

        x0 = np.concatenate([p_true, v_true, [theta_true], [vtheta_true], q_true, w_true])
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        solve_times.append(elapsed)

        u0 = solver.get(0, "u")
        f0, atheta0 = u0[:n_fans], u0[n_fans]
        wrench0 = A_full @ f0
        a0 = wrench0[0:3] / MASS
        torque0 = wrench0[3:6]
        wdot0 = torque0 / INERTIA

        e = p_true - P0 - theta_true * direction
        e_local = R.T @ e
        contour_mm = np.linalg.norm(e_local[:2]) * 1000
        lag_mm = e_local[2] * 1000
        dist_to_goal_mm = np.linalg.norm(P1 - p_true) * 1000
        q_err = quat_mul(quat_conj(q_target), q_true)
        att_err_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3])))

        over_budget = elapsed > REPLAN_BUDGET_S
        if over_budget:
            over_budget_ticks.append(tick)
        if status != 0:
            infeasible_ticks.append(tick)

        a0_along = np.dot(a0, direction)
        v_local = R.T @ v_true
        print(f"{tick:>4} {elapsed:>9.4f} {str(over_budget):>12} {status:>7} "
              f"{contour_mm:>11.2f} {lag_mm:>8.2f} {dist_to_goal_mm:>16.2f} {att_err_deg:>12.3f} "
              f"|f0|={np.linalg.norm(f0):.4f} vtheta={vtheta_true:.4f} a0_along_path={a0_along:.5f} "
              f"v_lat=({v_local[0]:.4f},{v_local[1]:.4f}) v_along={v_local[2]:.4f} "
              f"|w|={np.linalg.norm(w_true):.4f}")

        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK
        vtheta_true = vtheta_true + atheta0 * DT_TICK
        theta_true = theta_true + vtheta_true * DT_TICK
        w_true = w_true + wdot0 * DT_TICK
        q_true = q_true + quat_mul(q_true, np.concatenate([w_true, [0.0]])) * 0.5 * DT_TICK
        q_true = q_true / np.linalg.norm(q_true)

    solve_times = np.array(solve_times)
    print("\n--- Summary ---")
    print(f"ticks over {REPLAN_BUDGET_S*1000:.0f}ms budget: {over_budget_ticks}")
    print(f"ticks with solver status != 0 (infeasible/failed): {infeasible_ticks}")
    print(f"solve time: mean={solve_times.mean()*1000:.3f}ms "
          f"max={solve_times.max()*1000:.3f}ms")
    print(f"final distance to goal: {np.linalg.norm(P1 - p_true)*1000:.2f} mm")
    q_err_final = quat_mul(quat_conj(q_target), q_true)
    att_err_final_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err_final[:3]), abs(q_err_final[3])))
    print(f"final attitude error: {att_err_final_deg:.3f} deg")


if __name__ == "__main__":
    main()
