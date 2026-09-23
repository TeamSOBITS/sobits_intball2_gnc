#!/usr/bin/env python3
"""Closed-form quintic Hermite segment (ROS-agnostic, pure).

Given position/velocity/acceleration boundary conditions at both ends of a
single segment of duration ``T``, returns the unique degree-5 polynomial
matching all six conditions -- no optimization, no wrench-envelope
constraint (a fully-determined linear solve, 6 conditions for 6
coefficients). This is the "local layer" primitive of the replanning_minco
v4 design's global/local split (``docs/
2026-09-01_replanning_minco_v4_production_port_plan.md`` Phase 4): every
tick connects the current estimated state to a look-ahead point on the
(slower-cadence) global MINCO trajectory via one of these, without invoking
LBFGS (``gnc/test/experiment_minco_native/bench_v5_multiscenario.cpp``'s
``runScenario`` local-segment construction is the offline-verified
reference this ports).

Coefficients are returned in **ascending** power order, matching
:mod:`sobits_intball2_gnc.guidance.utils.polynomial`'s convention.
"""
import numpy as np

_N_COEFFS = 6


def solve_quintic_hermite_coeffs(p0, v0, a0, p1, v1, a1, duration):
    """Return ``(n_axes, 6)`` ascending-power coefficients for the segment.

    ``p0``/``v0``/``a0``/``p1``/``v1``/``a1`` are all shape ``(n_axes,)``
    (any common axis count -- 3 for position, 3 for a rotation vector).
    ``duration`` must be > 0.
    """
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    if not (p0.shape == v0.shape == a0.shape == p1.shape == v1.shape == a1.shape):
        raise ValueError("p0/v0/a0/p1/v1/a1 must all share the same shape")
    if duration <= 0.0:
        raise ValueError("duration must be > 0")

    T = float(duration)
    # Row i: [tau**0 .. tau**5] derivative-`order_i` coefficients at tau in
    # {0, T} -- the standard Hermite basis matrix, solved directly (no
    # transcribed closed-form expression) to keep this trivially verifiable
    # against sobits_intball2_gnc.guidance.utils.polynomial.evaluate().
    basis = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],                              # p(0)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],                              # v(0)
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],                              # a(0)
        [1.0, T, T**2, T**3, T**4, T**5],                            # p(T)
        [0.0, 1.0, 2.0 * T, 3.0 * T**2, 4.0 * T**3, 5.0 * T**4],     # v(T)
        [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T**2, 20.0 * T**3],          # a(T)
    ])
    rhs = np.stack([p0, v0, a0, p1, v1, a1], axis=0)  # (6, n_axes)
    coeffs = np.linalg.solve(basis, rhs)  # (6, n_axes)
    return coeffs.T  # (n_axes, 6), ascending power order per row
