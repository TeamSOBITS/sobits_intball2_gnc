#!/usr/bin/env python3
"""MPCC + attitude, reference path from the production Hermite spline
generator instead of a hand-derived closed-form circle
(``experiment_mpcc_curved_path_prototype.py``).

Motivation (docs/2026-09-20_mpcc_static_mode_disturbance_sweep_and_hermite_
next_step.md "次の一手"): MINCO has known open bugs (accRot Jacobian
missing, heuristic time-stretch whack-a-mole) that would contaminate any
MPCC-side finding if used as the reference-path source now. ``sobits_
intball2_gnc.guidance.trajectory.generation.hermite_spline_trajectory_
generator.HermiteSplineTrajectoryGenerator`` is simpler (closed-form
piecewise cubic, no known bugs) and is an actual production component
(current stand-in for min-snap in the real Guidance pipeline), so it lets
this MPCC track record advance without waiting on MINCO or conflating the
two.

Key formulation change from the circular-arc prototype: the arc's ``theta``
was arc length (``t_ref = dp_ref/dtheta`` was unit-norm for free, since the
arc was parametrized that way on purpose). ``HermiteSplineTrajectoryGenerator
.generate()`` returns per-segment cubic coefficients parametrized by
*segment-local time* (``segment_times`` argument), not arc length -- a
generic cubic's arc length isn't a closed-form function of its time
parameter, so exact arc-length reparametrization isn't available here
without numerical inversion. Instead, ``theta`` is reinterpreted as "nominal
elapsed time along the spline's own time parametrization" and ``t_ref`` is
explicitly normalized (``dp_ref/dtheta / |dp_ref/dtheta|``) rather than
assumed unit-norm. This is a strict generalization of the same contour/lag
decomposition (e = p - p_ref(theta), lag = e.t_ref, contour = e - lag*t_ref)
used by both prior prototypes -- nothing in that decomposition required
theta to be literal arc length, only that t_ref(theta) be a unit tangent.
Practically: under perfect tracking, theta advances at rate 1 (matching the
Hermite spline's own timing), so ``VTHETA_MAX``/``ATHETA_MAX`` are chosen
around that unit-rate nominal, not the arc-length-per-second scale the prior
prototypes used.

Path: two waypoints away from a start point (a 90 degree turn split into two
~0.58m straight chords, blended into a curve by the Hermite spline's
Catmull-Rom-style tangent at the interior waypoint -- see module-level
constants), all sharing one Z plane so the "face direction of travel"
attitude target can reuse the curved-path prototype's single-fixed-normal
rotation trick (``q_target(theta) = rotation by phi(theta) about the plane
normal``, ``phi`` referenced so ``phi(0)=0`` and thus ``q_target(0)=Q0``,
matching that prototype's convention) instead of a full 3D two-vector
quaternion construction. A true 3D (non-planar) Hermite path is future work.

``generate()``'s tangent estimator clamps start/end tangents to zero (start/
end at rest) unless ``v0`` is given -- a literal zero start tangent would
make ``t_ref(0)`` a 0/0 division. Passing ``v0`` (this prototype's initial
tangent direction, at nominal speed) sidesteps that and matches the
"replanning starts from nonzero measured velocity" convention this
generator was built for in the first place (see its module docstring).

Not a pytest test (no test_ prefix) -- standalone experiment:
    ACADOS_SOURCE_DIR=<path> LD_LIBRARY_PATH=<path>/lib \
        python3 test/experiment_mpcc_hermite_path_prototype.py [--disturbance weak]
"""
import argparse
import os
import time

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
INERTIA = 0.0136  # kg*m^2, isotropic, trajectory_controller.inertia
DT_TICK = 0.1  # guidance.replan_rate_hz = 10.0
REPLAN_BUDGET_S = DT_TICK

# Two ~0.58m chords, 90 degree turn at the interior waypoint, all in the
# z=5.163 plane (see module docstring on why planar).
WAYPOINTS = np.array([
    [10.155, -3.715, 5.163],
    [10.655, -4.015, 5.163],
    [10.955, -3.515, 5.163],
])
NOMINAL_SPEED = 0.15  # m/s, used only to size segment_times
SEGMENT_TIMES = np.array([
    np.linalg.norm(WAYPOINTS[1] - WAYPOINTS[0]) / NOMINAL_SPEED,
    np.linalg.norm(WAYPOINTS[2] - WAYPOINTS[1]) / NOMINAL_SPEED,
])
PLANE_NORMAL = np.array([0.0, 0.0, 1.0])

Q0 = np.array([0.0, 0.0, 0.0, 1.0])

N_TICKS = 80
DISTURB_TICK = 8
V_PERP_WEAK = 0.05
W_DISTURB_WEAK = 0.05

N_HORIZON = 20
TF_HORIZON = 2.0

# theta now measures "nominal seconds along the spline's own time
# parametrization" (see module docstring), not arc length -- under perfect
# tracking vtheta~=1 the whole time, hence the unit-rate-centered bounds.
VTHETA_MAX = 2.0
ATHETA_MAX = 0.5

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

# Same values confirmed for the straight-line prototype at weak disturbance
# (docs/2026-09-20_mpcc_static_mode_disturbance_sweep_and_hermite_next_step.md
# "重要な発見": the curved-path prototype's W_ATT=1e2 default reintroduces
# infeasibility on weak disturbance -- do not copy that default here).
W_ATT = 1e1
W_W = 1e1
W_ATT_TERM = 2e1
W_W_TERM = 2e1


def build_hermite_coeffs(v0_direction, nominal_speed):
    v0 = v0_direction / np.linalg.norm(v0_direction) * nominal_speed
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES, v0=v0)
    return coeffs  # (n_segments, 3, 8), only [..., 0:4] nonzero (cubic)


def _eval_segment(coeffs_seg, tau):
    """``coeffs_seg``: (3, 4) local cubic coefficients (xyz x [c0,c1,c2,c3]),
    plain numpy floats. Returns ``(p, dp_dtau)`` as length-3 lists of scalars
    -- built elementwise (not via numpy-array-times-tau) so ``tau`` can be
    either a python/numpy float or a CasADi SX expression without numpy
    trying (and failing) to broadcast an object array."""
    p = [coeffs_seg[d, 0] + coeffs_seg[d, 1] * tau + coeffs_seg[d, 2] * tau ** 2
         + coeffs_seg[d, 3] * tau ** 3 for d in range(3)]
    dp_dtau = [coeffs_seg[d, 1] + 2.0 * coeffs_seg[d, 2] * tau + 3.0 * coeffs_seg[d, 3] * tau ** 2
               for d in range(3)]
    return p, dp_dtau


def build_solver(A_full, fj_max, n_fans, coeffs, segment_times, tag=""):
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
    model.name = "mpcc_hermite_path_prototype"
    model.x = x
    model.u = u
    model.f_expl_expr = ca.vertcat(v, accel, vtheta, atheta, qdot, wdot)

    nx = 15
    nu = n_fans + 1

    # Piecewise p_ref(theta)/dp_ref_dtheta(theta): clamp theta into
    # [0, T_total] first (OCP box constraint already does this at the state
    # level, this is defensive for horizon-internal predictions that
    # transiently step outside it), then nested if_else on the cumulative
    # segment boundaries to pick the active segment and evaluate its local
    # cubic at tau = theta - segment_start.
    t_bounds = np.concatenate([[0.0], np.cumsum(segment_times)])
    t_total = t_bounds[-1]
    theta_c = ca.fmin(ca.fmax(theta, 0.0), t_total)

    n_segments = coeffs.shape[0]
    p_ref = ca.DM.zeros(3)
    dp_ref_dtheta = ca.DM.zeros(3)
    for i in range(n_segments):
        tau_i = theta_c - t_bounds[i]
        p_i, dp_i = _eval_segment(coeffs[i], tau_i)
        in_segment = ca.logic_and(theta_c >= t_bounds[i], theta_c <= t_bounds[i + 1])
        p_ref = ca.if_else(in_segment, ca.vertcat(*p_i), p_ref)
        dp_ref_dtheta = ca.if_else(in_segment, ca.vertcat(*dp_i), dp_ref_dtheta)

    t_ref = dp_ref_dtheta / ca.sqrt(ca.sumsqr(dp_ref_dtheta) + 1e-9)

    e = p - p_ref
    lag = ca.dot(e, t_ref)
    e_perp = e - lag * t_ref
    v_along = ca.dot(v, t_ref)
    v_perp_vec = v - v_along * t_ref

    p1_goal = ca.DM(WAYPOINTS[-1])
    to_goal = p1_goal - p
    dist_to_goal = ca.sqrt(ca.sumsqr(to_goal) + 1e-6)
    closing_rate = ca.dot(to_goal, v) / dist_to_goal

    lag_cost = W_LAG * LAG_HUBER_DELTA ** 2 * (
        ca.sqrt(1.0 + (lag / LAG_HUBER_DELTA) ** 2) - 1.0
    )

    # "Face direction of travel", planar case (see module docstring): phi(theta)
    # is t_ref(theta)'s absolute angle in the (in-plane-axis, normal x
    # in-plane-axis) basis, referenced against phi(0) so q_target(0)=Q0.
    in_plane_axis_np = (WAYPOINTS[1] - WAYPOINTS[0]) / np.linalg.norm(WAYPOINTS[1] - WAYPOINTS[0])
    binormal_np = np.cross(PLANE_NORMAL, in_plane_axis_np)
    in_plane_axis = ca.DM(in_plane_axis_np)
    binormal = ca.DM(binormal_np)
    normal_ca = ca.DM(PLANE_NORMAL)
    phi = ca.atan2(ca.dot(t_ref, binormal), ca.dot(t_ref, in_plane_axis))
    _, dp0 = _eval_segment(coeffs[0], 0.0)
    dp0 = np.array(dp0, dtype=float)
    t_ref0_np = dp0 / np.linalg.norm(dp0)
    phi0 = float(np.arctan2(np.dot(t_ref0_np, binormal_np), np.dot(t_ref0_np, in_plane_axis_np)))
    phi_rel = phi - phi0
    half = phi_rel / 2.0
    q_target = ca.vertcat(normal_ca * ca.sin(half), ca.cos(half))
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
        W_E_TERM * ca.sumsqr(p - p1_goal) + W_V_TERM * ca.sumsqr(v)
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
    ocp.code_export_directory = f"/tmp/acados_mpcc_hermite_path_prototype_codegen_{suffix}"

    return AcadosOcpSolver(ocp, json_file=f"/tmp/acados_mpcc_hermite_path_prototype_ocp_{suffix}.json")


def eval_ref_numpy(coeffs, segment_times, theta):
    t_bounds = np.concatenate([[0.0], np.cumsum(segment_times)])
    theta_c = min(max(theta, 0.0), t_bounds[-1])
    for i in range(coeffs.shape[0]):
        if t_bounds[i] <= theta_c <= t_bounds[i + 1] or i == coeffs.shape[0] - 1:
            tau = theta_c - t_bounds[i]
            p, dp = _eval_segment(coeffs[i][:, :4], tau)
            p = np.array(p, dtype=float)
            dp = np.array(dp, dtype=float)
            t_ref = dp / np.linalg.norm(dp)
            return p, t_ref
    raise AssertionError("unreachable")


def eval_q_target_numpy(coeffs, segment_times, in_plane_axis, normal, phi0, theta):
    _, t_ref = eval_ref_numpy(coeffs, segment_times, theta)
    binormal = np.cross(normal, in_plane_axis)
    phi = np.arctan2(np.dot(t_ref, binormal), np.dot(t_ref, in_plane_axis))
    phi_rel = phi - phi0
    half = phi_rel / 2.0
    return np.concatenate([normal * np.sin(half), [np.cos(half)]])


def main():
    global W_ATT, W_W, W_ATT_TERM, W_W_TERM
    parser = argparse.ArgumentParser()
    parser.add_argument("--disturbance", choices=["none", "weak"], default="none")
    parser.add_argument("--ticks", type=int, default=N_TICKS)
    parser.add_argument("--w-att", type=float, default=W_ATT)
    parser.add_argument("--w-w", type=float, default=W_W)
    parser.add_argument("--w-att-term", type=float, default=W_ATT_TERM)
    parser.add_argument("--w-w-term", type=float, default=W_W_TERM)
    parser.add_argument("--v-perp", type=float, default=None)
    parser.add_argument("--w-disturb", type=float, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    n_ticks = args.ticks
    v_perp_error = {"none": 0.0, "weak": V_PERP_WEAK}[args.disturbance]
    w_disturb_mag = {"none": 0.0, "weak": W_DISTURB_WEAK}[args.disturbance]
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

    in_plane_axis = (WAYPOINTS[1] - WAYPOINTS[0]) / np.linalg.norm(WAYPOINTS[1] - WAYPOINTS[0])
    coeffs = build_hermite_coeffs(in_plane_axis, NOMINAL_SPEED)
    _, t_ref0_np = eval_ref_numpy(coeffs, SEGMENT_TIMES, 0.0)
    binormal0 = np.cross(PLANE_NORMAL, in_plane_axis)
    phi0 = float(np.arctan2(np.dot(t_ref0_np, binormal0), np.dot(t_ref0_np, in_plane_axis)))

    solver = build_solver(A_full, fj_max, n_fans, coeffs, SEGMENT_TIMES, tag=args.tag)

    p_true = WAYPOINTS[0].copy()
    v_true = in_plane_axis * NOMINAL_SPEED  # matches the v0 baked into the spline's start tangent
    theta_true = 0.0
    vtheta_true = 1.0  # nominal rate under perfect tracking, see module docstring
    q_true = Q0.copy()
    w_true = np.zeros(3)

    t_total = SEGMENT_TIMES.sum()
    print(f"waypoints={WAYPOINTS.tolist()} segment_times={SEGMENT_TIMES.tolist()} "
          f"t_total={t_total:.2f}s disturbance={args.disturbance} "
          f"(+{v_perp_error} m/s lateral, +{w_disturb_mag} rad/s angular at tick {DISTURB_TICK})")
    print(f"{'tick':>4} {'solve_s':>9} {'status':>7} {'contour_mm':>11} {'lag_mm':>8} "
          f"{'dist_to_goal_mm':>16} {'theta':>7} {'att_err_deg':>12}")

    infeasible_ticks = []
    solve_times = []

    for tick in range(n_ticks):
        if tick == DISTURB_TICK and (v_perp_error > 0.0 or w_disturb_mag > 0.0):
            perp_dir = np.cross(PLANE_NORMAL, in_plane_axis)
            v_true = v_true + v_perp_error * perp_dir
            w_true = w_true + w_disturb_mag * PLANE_NORMAL
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

        p_ref, t_ref = eval_ref_numpy(coeffs, SEGMENT_TIMES, theta_true)
        e = p_true - p_ref
        lag_mm = np.dot(e, t_ref) * 1000
        contour_mm = np.linalg.norm(e - np.dot(e, t_ref) * t_ref) * 1000
        dist_to_goal_mm = np.linalg.norm(WAYPOINTS[-1] - p_true) * 1000
        q_target = eval_q_target_numpy(coeffs, SEGMENT_TIMES, in_plane_axis, PLANE_NORMAL, phi0, theta_true)
        q_err = quat_mul(quat_conj(q_target), q_true)
        att_err_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err[:3]), abs(q_err[3])))

        print(f"{tick:>4} {elapsed:>9.4f} {status:>7} {contour_mm:>11.2f} {lag_mm:>8.2f} "
              f"{dist_to_goal_mm:>16.2f} {theta_true:>7.3f} {att_err_deg:>12.3f} "
              f"|f0|={np.linalg.norm(f0):.4f} vtheta={vtheta_true:.4f} |w|={np.linalg.norm(w_true):.4f}")

        v_true = v_true + a0 * DT_TICK
        p_true = p_true + v_true * DT_TICK
        vtheta_true = np.clip(vtheta_true + atheta0 * DT_TICK, 0.0, VTHETA_MAX)
        theta_true = theta_true + vtheta_true * DT_TICK
        w_true = w_true + wdot0 * DT_TICK
        q_true = q_true + quat_mul(q_true, np.concatenate([w_true, [0.0]])) * 0.5 * DT_TICK
        q_true = q_true / np.linalg.norm(q_true)

    solve_times = np.array(solve_times)
    print("\n--- Summary ---")
    print(f"ticks with solver status != 0 (infeasible/failed): {infeasible_ticks}")
    print(f"solve time: mean={solve_times.mean()*1000:.3f}ms max={solve_times.max()*1000:.3f}ms")
    print(f"final distance to goal: {np.linalg.norm(WAYPOINTS[-1] - p_true)*1000:.2f} mm")
    q_target_final = eval_q_target_numpy(coeffs, SEGMENT_TIMES, in_plane_axis, PLANE_NORMAL, phi0, theta_true)
    q_err_final = quat_mul(quat_conj(q_target_final), q_true)
    att_err_final_deg = np.degrees(2.0 * np.arctan2(np.linalg.norm(q_err_final[:3]), abs(q_err_final[3])))
    print(f"final attitude error: {att_err_final_deg:.3f} deg")


if __name__ == "__main__":
    main()
