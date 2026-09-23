#!/usr/bin/env python3
"""MPCC curved-path prototype -- checks whether the contour/lag decomposition
used by ``experiment_mpcc_translation_prototype.py`` generalizes to an
actually curved reference path, and (unlike that prototype and the
straight-line ``experiment_mpcc_attitude_tracking_prototype.py``) also
integrates attitude tracking along the curve.

That prototype (and the attitude-integration one,
``experiment_mpcc_attitude_tracking_prototype.py``) only ever used a straight
line ``P0->P1``: the reference point ``p_ref(theta) = P0 + theta*direction``
and a single *fixed* local frame ``R`` built once from that one direction
vector. Progress ``theta`` only ever moves the reference point along that
one fixed direction -- there is no mechanism in that code for a curving
path, and none was ever tested (confirmed by inspection: nothing in the
existing MPCC experiment scripts or docs exercises a curve). Real reference
paths (MINCO/TOPP-RA output) do curve, so this matters before any of the
disturbance-recovery findings in ``docs/2026-09-18_mpcc_attitude_weight_
tuning_step2_findings.md`` can be said to generalize beyond a straight line.

Path: a constant-curvature quarter-circle arc (radius ``ARC_RADIUS``),
parametrized by arc length so ``theta`` keeps its original meaning
("distance travelled along the path", same units/role as the straight-line
version). Reference position/tangent are closed-form (``cos``/``sin`` of
``theta/ARC_RADIUS``), embedded directly as CasADi expressions -- no spline
fitting or MINCO call needed for this first check. Contour/lag are
redefined via tangent projection instead of a fixed frame:

    e = p - p_ref(theta)
    lag = e . t_ref(theta)                  (along the local tangent)
    e_perp = e - lag * t_ref(theta)          (everything else)
    contour_cost = W_CONTOUR * |e_perp|^2
    lag_cost = <same saturated/pseudo-Huber form as before>(lag)

This reduces to the straight-line version's e_local[0:2]/e_local[2] split
when t_ref is constant (a straight line's tangent doesn't depend on theta),
so it is a strict generalization, not a different formulation.

``--disturbance`` (weak/strong lateral + angular kick, same magnitudes as
the straight-line prototypes) is included for a follow-up check once the
undisturbed case is confirmed to work.

Attitude (following ``experiment_mpcc_attitude_tracking_prototype.py``'s
quaternion state/cost, shared 8-fan wrench, isotropic-inertia dynamics --
see that module's docstring for those details, not repeated here): unlike
the straight-line prototype's fixed target quaternion, the target here is
"face direction of travel" -- it rotates *with* ``theta`` so it always
points along ``t_ref(theta)``. Since the arc is planar, this is just a
rotation about the arc's fixed out-of-plane normal by ``phi = theta/radius``
(the same angle ``t_ref`` itself is parametrized by), starting from identity
at ``theta=0``: ``q_target(theta) = [axis*sin(phi/2), cos(phi/2)]``, ``axis
= cross(direction, e_r)`` (the axis for which Rodrigues' rotation formula
reproduces ``t_ref`` exactly -- verified by hand, see the sign of the
``sin(phi)`` term in ``t_ref`` vs. this axis choice). This is built as a
CasADi expression of the symbolic ``theta`` state, not evaluated once at
setup, so the OCP sees the reference rotate within the horizon. A straight
line is the degenerate case of this same construction (``t_ref`` constant
-> ``phi`` frozen -> fixed target), matching the straight-line prototype's
fixed-target design after the fact. ``phi`` stays within +/-``ARC_SWEEP``/2
of 0 over this quarter-circle, so no hemisphere/wraparound handling is
needed (unlike the straight-line prototype's arbitrary fixed target).

Not a pytest test (no test_ prefix) -- standalone experiment:
    ACADOS_SOURCE_DIR=<path> LD_LIBRARY_PATH=<path>/lib \
        python3 test/experiment_mpcc_curved_path_prototype.py [--disturbance weak|strong]
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
_P1_STRAIGHT = np.array([11.0, -4.3, 5.0])  # only used to derive an initial tangent direction

ARC_RADIUS = 0.6  # m -- comparable scale to the straight-line prototypes' ~1.04m path
ARC_SWEEP = np.pi / 2  # quarter circle -- a genuine 90 degree turn, not a gentle bend

Q0 = np.array([0.0, 0.0, 0.0, 1.0])

N_TICKS = 60
DISTURB_TICK = 8
V_PERP_WEAK = 0.05
V_PERP_STRONG = 0.4
W_DISTURB_WEAK = 0.05
W_DISTURB_STRONG = 0.3

N_HORIZON = 20
TF_HORIZON = 2.0

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

# Weights from the subprocess-isolated re-sweep, docs/2026-09-18_mpcc_
# attitude_weight_moderate_disturbance_long_horizon.md (the original
# ThreadPoolExecutor-based sweep raced on these as module globals and its
# numbers didn't reproduce); confirmed stable to 800 ticks for the
# straight-line prototype.
W_ATT = 1e2
W_W = 1e1
W_ATT_TERM = 2e2
W_W_TERM = 2e1


def build_arc(p0, p1_straight, radius, sweep):
    """Quarter-circle arc through ``p0``, tangent to the ``p0->p1_straight``
    direction at its start. Returns ``(center, e_r, direction, path_length,
    p1_arc)`` -- ``e_r``/``direction`` are the orthonormal in-plane basis
    (radially-outward-at-start / initial-tangent), ``p1_arc`` is the arc's
    end point (the new goal, replacing the old straight-line ``P1``)."""
    direction = p1_straight - p0
    direction = direction / np.linalg.norm(direction)
    n1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(n1) < 1e-6:
        n1 = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    n1 = n1 / np.linalg.norm(n1)
    e_r = np.cross(direction, n1)  # radially-outward-from-center direction at theta=0
    center = p0 - radius * e_r
    path_length = radius * sweep
    p1_arc = center + radius * (
        np.cos(sweep) * e_r + np.sin(sweep) * direction
    )
    return center, e_r, direction, path_length, p1_arc


def build_solver(A_full, fj_max, n_fans, center, e_r, direction, radius, p1_arc, tag=""):
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
    wdot = torque / INERTIA  # isotropic inertia -> no w x (I*w) term

    qw = ca.vertcat(w, 0.0)
    qdot = 0.5 * ca.vertcat(
        q[3] * qw[0] + q[0] * qw[3] + q[1] * qw[2] - q[2] * qw[1],
        q[3] * qw[1] - q[0] * qw[2] + q[1] * qw[3] + q[2] * qw[0],
        q[3] * qw[2] + q[0] * qw[1] - q[1] * qw[0] + q[2] * qw[3],
        q[3] * qw[3] - q[0] * qw[0] - q[1] * qw[1] - q[2] * qw[2],
    )

    model = AcadosModel()
    model.name = "mpcc_curved_path_prototype"
    model.x = x
    model.u = u
    model.f_expl_expr = ca.vertcat(v, accel, vtheta, atheta, qdot, wdot)

    nx = 15
    nu = n_fans + 1

    phi = theta / radius
    center_ca, e_r_ca, direction_ca = ca.DM(center), ca.DM(e_r), ca.DM(direction)
    p_ref = center_ca + radius * (ca.cos(phi) * e_r_ca + ca.sin(phi) * direction_ca)
    t_ref = -ca.sin(phi) * e_r_ca + ca.cos(phi) * direction_ca  # dp_ref/dtheta, unit (arc-length param)

    e = p - p_ref
    lag = ca.dot(e, t_ref)
    e_perp = e - lag * t_ref
    v_along = ca.dot(v, t_ref)
    v_perp_vec = v - v_along * t_ref

    to_goal = ca.DM(p1_arc) - p
    dist_to_goal = ca.sqrt(ca.sumsqr(to_goal) + 1e-6)
    closing_rate = ca.dot(to_goal, v) / dist_to_goal

    lag_cost = W_LAG * LAG_HUBER_DELTA ** 2 * (
        ca.sqrt(1.0 + (lag / LAG_HUBER_DELTA) ** 2) - 1.0
    )

    # "Face direction of travel": target quaternion is a rotation of Q0 by
    # phi(theta) about the arc's normal, chosen so it reproduces t_ref exactly
    # (see module docstring) -- symbolic in theta, so it rotates within the
    # horizon rather than being fixed at solver-build time.
    axis_ca = ca.DM(np.cross(direction, e_r))
    half = phi / 2.0
    q_target = ca.vertcat(axis_ca * ca.sin(half), ca.cos(half))
    q_target_conj = ca.vertcat(-q_target[0:3], q_target[3])
    q_err = ca.vertcat(
        q_target_conj[3] * q[0] + q_target_conj[0] * q[3] + q_target_conj[1] * q[2] - q_target_conj[2] * q[1],
        q_target_conj[3] * q[1] - q_target_conj[0] * q[2] + q_target_conj[1] * q[3] + q_target_conj[2] * q[0],
        q_target_conj[3] * q[2] + q_target_conj[0] * q[1] - q_target_conj[1] * q[0] + q_target_conj[2] * q[3],
        q_target_conj[3] * q[3] - q_target_conj[0] * q[0] - q_target_conj[1] * q[1] - q_target_conj[2] * q[2],
    )
    att_cost = W_ATT * ca.sumsqr(q_err[0:3]) + W_W * ca.sumsqr(w)

    stage_cost = (
        W_CONTOUR * ca.sumsqr(e_perp)
        + lag_cost
        + W_V_LATERAL * ca.sumsqr(v_perp_vec)
        + W_V_ALONG * v_along ** 2
        - MU_PROGRESS * closing_rate
        + W_F * ca.sumsqr(f)
        + W_ATHETA * atheta ** 2
        + att_cost
    )
    terminal_cost = (
        W_E_TERM * ca.sumsqr(p - ca.DM(p1_arc)) + W_V_TERM * ca.sumsqr(v)
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
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([VTHETA_MAX])
    ocp.constraints.idxbx = np.array([7])
    ocp.constraints.x0 = np.concatenate([np.zeros(8), Q0, np.zeros(3)])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = TF_HORIZON
    suffix = tag or str(os.getpid())
    ocp.code_export_directory = f"/tmp/acados_mpcc_curved_path_prototype_codegen_{suffix}"

    return AcadosOcpSolver(ocp, json_file=f"/tmp/acados_mpcc_curved_path_prototype_ocp_{suffix}.json")


def eval_ref(center, e_r, direction, radius, theta):
    phi = theta / radius
    p_ref = center + radius * (np.cos(phi) * e_r + np.sin(phi) * direction)
    t_ref = -np.sin(phi) * e_r + np.cos(phi) * direction
    return p_ref, t_ref


def eval_q_target(axis, radius, theta):
    phi = theta / radius
    half = phi / 2.0
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def main():
    global W_ATT, W_W, W_ATT_TERM, W_W_TERM
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturbance", choices=["none", "weak", "strong"], default="none")
    parser.add_argument("--ticks", type=int, default=N_TICKS)
    parser.add_argument("--w-att", type=float, default=W_ATT)
    parser.add_argument("--w-w", type=float, default=W_W)
    parser.add_argument("--w-att-term", type=float, default=W_ATT_TERM)
    parser.add_argument("--w-w-term", type=float, default=W_W_TERM)
    parser.add_argument("--v-perp", type=float, default=None,
                         help="override lateral disturbance kick (m/s); "
                              "takes precedence over --disturbance's weak/strong preset")
    parser.add_argument("--w-disturb", type=float, default=None,
                         help="override angular disturbance kick magnitude, about the arc "
                              "normal (rad/s); takes precedence over --disturbance's weak/strong preset")
    parser.add_argument("--tag", default="",
                         help="unique suffix for acados codegen paths, so parallel runs don't clobber each other")
    parser.add_argument("--arc-radius", type=float, default=ARC_RADIUS,
                         help="tighter turn = smaller radius (higher curvature, more centripetal "
                              "accel demand for the same speed)")
    parser.add_argument("--arc-sweep-deg", type=float, default=np.degrees(ARC_SWEEP))
    args = parser.parse_args()
    arc_radius = args.arc_radius
    arc_sweep = np.radians(args.arc_sweep_deg)
    n_ticks = args.ticks
    v_perp_error = {"none": 0.0, "weak": V_PERP_WEAK, "strong": V_PERP_STRONG}[args.disturbance]
    w_disturb_mag = {"none": 0.0, "weak": W_DISTURB_WEAK, "strong": W_DISTURB_STRONG}[args.disturbance]
    if args.v_perp is not None:
        v_perp_error = args.v_perp
    if args.w_disturb is not None:
        w_disturb_mag = args.w_disturb
    W_ATT, W_W, W_ATT_TERM, W_W_TERM = args.w_att, args.w_w, args.w_att_term, args.w_w_term

    if "ACADOS_SOURCE_DIR" not in os.environ:
        raise SystemExit("ACADOS_SOURCE_DIR not set -- point it at the acados build")

    allocator = ThrustAllocator()
    A_full = allocator.A
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count

    center, e_r, direction, path_length, p1_arc = build_arc(P0, _P1_STRAIGHT, arc_radius, arc_sweep)
    axis = np.cross(direction, e_r)
    solver = build_solver(A_full, fj_max, n_fans, center, e_r, direction, arc_radius, p1_arc, tag=args.tag)

    p_true = P0.copy()
    v_true = np.zeros(3)
    theta_true = 0.0
    vtheta_true = 0.0
    q_true = Q0.copy()
    w_true = np.zeros(3)

    print(f"arc: radius={arc_radius}m sweep={np.degrees(arc_sweep):.0f}deg "
          f"path_length={path_length*1000:.1f}mm p1_arc={p1_arc}")
    print(f"disturbance={args.disturbance} (+{v_perp_error} m/s lateral, "
          f"+{w_disturb_mag} rad/s angular at tick {DISTURB_TICK})")
    print(f"{'tick':>4} {'solve_s':>9} {'status':>7} {'contour_mm':>11} {'lag_mm':>8} "
          f"{'dist_to_goal_mm':>16} {'theta_mm':>9} {'att_err_deg':>12}")

    infeasible_ticks = []
    solve_times = []

    for tick in range(n_ticks):
        if tick == DISTURB_TICK and (v_perp_error > 0.0 or w_disturb_mag > 0.0):
            v_true = v_true + v_perp_error * e_r
            w_true = w_true + w_disturb_mag * axis
            print(f"--- disturbance injected at tick {tick}: "
                  f"+{v_perp_error} m/s lateral, +{w_disturb_mag} rad/s angular ---")

        x0 = np.concatenate([p_true, v_true, [theta_true], [vtheta_true], q_true, w_true])
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        t0 = time.perf_counter()
        status = solver.solve()
        elapsed = time.perf_counter() - t0
        solve_times.append(elapsed)
        if status != 0:
            infeasible_ticks.append(tick)

        u0 = solver.get(0, "u")
        f0, atheta0 = u0[:n_fans], u0[n_fans]
        wrench0 = A_full @ f0
        a0 = wrench0[0:3] / MASS
        torque0 = wrench0[3:6]
        wdot0 = torque0 / INERTIA

        p_ref, t_ref = eval_ref(center, e_r, direction, arc_radius, theta_true)
        e = p_true - p_ref
        lag_mm = np.dot(e, t_ref) * 1000
        contour_mm = np.linalg.norm(e - np.dot(e, t_ref) * t_ref) * 1000
        dist_to_goal_mm = np.linalg.norm(p1_arc - p_true) * 1000
        q_target = eval_q_target(axis, arc_radius, theta_true)
        q_err = quat_mul(quat_conj(q_target), q_true)
        att_err_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3])))

        print(f"{tick:>4} {elapsed:>9.4f} {status:>7} {contour_mm:>11.2f} {lag_mm:>8.2f} "
              f"{dist_to_goal_mm:>16.2f} {theta_true*1000:>9.1f} {att_err_deg:>12.3f} "
              f"|f0|={np.linalg.norm(f0):.4f} vtheta={vtheta_true:.4f} |w|={np.linalg.norm(w_true):.4f}")

        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK
        # plant-side integration has no OCP box to bound it (that's only enforced
        # inside the solver's prediction horizon) -- clamp to the same physical
        # range so a stuck solver (atheta0 pinned) can't run vtheta away unbounded
        vtheta_true = np.clip(vtheta_true + atheta0 * DT_TICK, 0.0, VTHETA_MAX)
        theta_true = theta_true + vtheta_true * DT_TICK
        w_true = w_true + wdot0 * DT_TICK
        q_true = q_true + quat_mul(q_true, np.concatenate([w_true, [0.0]])) * 0.5 * DT_TICK
        q_true = q_true / np.linalg.norm(q_true)

    solve_times = np.array(solve_times)
    print("\n--- Summary ---")
    print(f"ticks with solver status != 0 (infeasible/failed): {infeasible_ticks}")
    print(f"solve time: mean={solve_times.mean()*1000:.3f}ms max={solve_times.max()*1000:.3f}ms")
    print(f"final distance to goal: {np.linalg.norm(p1_arc - p_true)*1000:.2f} mm")
    q_target_final = eval_q_target(axis, arc_radius, theta_true)
    q_err_final = quat_mul(quat_conj(q_target_final), q_true)
    att_err_final_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err_final[:3]), abs(q_err_final[3])))
    print(f"final attitude error: {att_err_final_deg:.3f} deg")


if __name__ == "__main__":
    main()
