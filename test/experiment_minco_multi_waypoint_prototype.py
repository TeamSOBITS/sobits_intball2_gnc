#!/usr/bin/env python3
"""Prototype: does letting MINCO also move the intermediate waypoint (not
just segment time T) help, on a real multi-segment scenario?

Follow-up to docs/2026-08-29_minco_gcopter_survey.md and
test/experiment_minco_box_constraint_prototype.py, which only tested a single
segment with both waypoints fixed (q wasn't a free variable at all -- the
part of MINCO's value proposition this script actually exercises). Scenario:
the real ~144.7deg hairpin from test/manual/send_hairpin_naventry.py
(near_dock -> nav_entry-direction bulge waypoint -> above_dock), same
per-axis force box constraint as before.

Two segments, quintic Hermite (boundary p,v,a explicit per segment, matching
at the interior knot by construction -- no separate continuity constraint
needed since both segments are built from the same interior v1,a1). Decision
variables: interior waypoint position q1, interior velocity/acceleration
v1/a1 (free -- this is what MINCO solves for internally via a banded linear
system given fixed q,T; here they're just NLP variables, which is a
deliberate simplification, see docs/2026-08-29_minco_gcopter_survey.md), and
segment durations T1,T2.

An unconstrained-shape hairpin has a trivial "solution" -- push q1 onto the
straight line between endpoints and the turn disappears. That would test
nothing (real waypoints exist because of an obstacle/corridor requirement
elsewhere in guidance, not modeled here), so q1 is bounded to a small box
around the given hairpin waypoint (+-0.3m/axis) rather than left fully free.
Two conditions are compared:
  - baseline: q1 pinned exactly at the given waypoint (matches
    experiment_minco_box_constraint_prototype.py's fixed-q approach, just
    2 segments instead of 1)
  - minco: q1 free within the box

A small jerk-energy term is included (w_energy) -- without it, v1/a1 have no
cost pushing them toward a physically sensible shape (only the accel-box
penalty and time cost constrain them), which could let the optimizer pick a
degenerate interior velocity/acceleration. This is the actual reason MINCO's
formulation always couples an energy term with constraints ("MINimum
COntrol effort" trajectory class) -- it's not an incidental extra.

Not a pytest test (no test_ prefix, not collected by colcon test) -- run
directly:
    python3 test/experiment_minco_multi_waypoint_prototype.py
"""
import time

import numpy as np
from scipy.optimize import minimize

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
MAX_FORCE = np.array([0.181, 0.0996, 0.122])  # config/gnc_params.yaml (x,y,z)
MAX_ACCEL = MAX_FORCE / MASS

# test/manual/send_hairpin_naventry.py, near_dock -> above_dock via a
# nav_entry-direction bulge waypoint (bulge_scale=1.5 -> ~144.7deg turn).
NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5

Q0 = NEAR_DOCK
Q2 = ABOVE_DOCK
_midpoint = 0.5 * (Q0 + Q2)
_bulge = NAV_ENTRY - _midpoint
Q1_GIVEN = _midpoint + BULGE_SCALE * _bulge  # the hairpin waypoint W

Q1_BOX_HALF_WIDTH = 0.3  # m/axis, how far q1 may move from Q1_GIVEN

N_SAMPLES = 30  # per segment, for the accel-box quadrature penalty
W_ENERGY = 1e-3  # jerk-energy regularization weight (see module docstring)


# Quintic Hermite basis (tau in [0,1]) and derivatives w.r.t. tau, up to 3rd
# (jerk). p(tau) = p0*h00 + v0*T*h10 + a0*T^2*h20 + p1*h01 + v1*T*h11 + a1*T^2*h21.
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
    elif order == 3:
        h00 = -60 + 360 * tau - 360 * t2
        h10 = -36 + 192 * tau - 180 * t2
        h20 = -9 + 36 * tau - 30 * t2
        h01 = 60 - 360 * tau + 360 * t2
        h11 = -24 + 168 * tau - 180 * t2
        h21 = 3 - 24 * tau + 30 * t2
    else:
        raise ValueError(order)
    return h00, h10, h20, h01, h11, h21


def hermite_deriv(tau, T, p0, v0, a0, p1, v1, a1, deriv_order):
    """d^deriv_order p / dt^deriv_order at tau=t/T, vectorized over axes."""
    h00, h10, h20, h01, h11, h21 = _hermite_bases(tau, deriv_order)
    d_tau = (
        p0 * h00 + v0 * T * h10 + a0 * T**2 * h20
        + p1 * h01 + v1 * T * h11 + a1 * T**2 * h21
    )
    return d_tau / T**deriv_order


def segment_penalty(T, p0, v0, a0, p1, v1, a1, max_accel, weight):
    tau = np.linspace(0.0, 1.0, N_SAMPLES)[:, None]
    accel = hermite_deriv(tau, T, p0, v0, a0, p1, v1, a1, deriv_order=2)  # (N,3)
    viol = np.maximum(np.abs(accel) - max_accel[None, :], 0.0)
    return weight * np.sum(viol**3) / N_SAMPLES


def segment_energy(T, p0, v0, a0, p1, v1, a1):
    """Quadrature approx of integral jerk^2 dt over the segment."""
    tau = np.linspace(0.0, 1.0, N_SAMPLES)[:, None]
    jerk = hermite_deriv(tau, T, p0, v0, a0, p1, v1, a1, deriv_order=3)  # (N,3)
    return np.sum(jerk**2) / N_SAMPLES * T


def unpack(x):
    q1, v1, a1 = x[0:3], x[3:6], x[6:9]
    T1, T2 = x[9], x[10]
    return q1, v1, a1, T1, T2


def pack(q1, v1, a1, T1, T2):
    return np.concatenate([q1, v1, a1, [T1, T2]])


def max_violation(q1, v1, a1, T1, T2):
    tau = np.linspace(0.0, 1.0, N_SAMPLES)[:, None]
    a_seg1 = hermite_deriv(tau, T1, Q0, np.zeros(3), np.zeros(3), q1, v1, a1, 2)
    a_seg2 = hermite_deriv(tau, T2, q1, v1, a1, Q2, np.zeros(3), np.zeros(3), 2)
    v1_seg1 = np.max(np.maximum(np.abs(a_seg1) - MAX_ACCEL[None, :], 0.0))
    v1_seg2 = np.max(np.maximum(np.abs(a_seg2) - MAX_ACCEL[None, :], 0.0))
    return max(v1_seg1, v1_seg2)


def solve(q1_free, w_time=1.0, weight_schedule=(1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14)):
    if q1_free:
        lower = Q1_GIVEN - Q1_BOX_HALF_WIDTH
        upper = Q1_GIVEN + Q1_BOX_HALF_WIDTH
    else:
        lower = upper = Q1_GIVEN

    T_guess = 15.0  # loose warm start, both segments
    x = pack(Q1_GIVEN.copy(), np.zeros(3), np.zeros(3), T_guess, T_guess)

    bounds = (
        [(lower[i], upper[i]) for i in range(3)]
        + [(None, None)] * 6  # v1, a1 free
        + [(1e-3, None), (1e-3, None)]  # T1, T2 > 0
    )

    outer_iters = 0
    for weight in weight_schedule:
        outer_iters += 1

        def objective(x, weight=weight):
            q1, v1, a1, T1, T2 = unpack(x)
            e1 = segment_energy(T1, Q0, np.zeros(3), np.zeros(3), q1, v1, a1)
            e2 = segment_energy(T2, q1, v1, a1, Q2, np.zeros(3), np.zeros(3))
            p1 = segment_penalty(T1, Q0, np.zeros(3), np.zeros(3), q1, v1, a1, MAX_ACCEL, weight)
            p2 = segment_penalty(T2, q1, v1, a1, Q2, np.zeros(3), np.zeros(3), MAX_ACCEL, weight)
            return w_time * (T1 + T2) + W_ENERGY * (e1 + e2) + p1 + p2

        result = minimize(
            objective, x, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        x = result.x

    q1, v1, a1, T1, T2 = unpack(x)
    viol = max_violation(q1, v1, a1, T1, T2)
    return q1, v1, a1, T1, T2, viol, outer_iters


def main():
    turn_angle = np.degrees(np.arccos(np.clip(
        np.dot(Q1_GIVEN - Q0, Q2 - Q1_GIVEN)
        / (np.linalg.norm(Q1_GIVEN - Q0) * np.linalg.norm(Q2 - Q1_GIVEN)),
        -1.0, 1.0,
    )))
    print(f"hairpin: near_dock -> W -> above_dock, turn_angle={turn_angle:.1f}deg")
    print(f"Q1_GIVEN: {Q1_GIVEN}")
    print()

    for label, q1_free in [("baseline (q1 fixed)", False), ("minco-style (q1 free +-0.3m)", True)]:
        t_start = time.perf_counter()
        q1, v1, a1, T1, T2, viol, outer_iters = solve(q1_free)
        elapsed = time.perf_counter() - t_start
        print(f"--- {label} ---")
        print(f"q1:            {q1}  (moved {np.linalg.norm(q1 - Q1_GIVEN):.4f} m from given)")
        print(f"v1 (interior): {v1}")
        print(f"a1 (interior): {a1}")
        print(f"T1, T2:        {T1:.4f}, {T2:.4f}  (total {T1 + T2:.4f} s)")
        print(f"max accel-box violation: {viol:.3e}")
        print(f"outer iterations: {outer_iters}, solve time: {elapsed:.4f} s")
        print()


if __name__ == "__main__":
    main()
