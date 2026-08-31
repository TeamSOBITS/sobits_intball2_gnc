"""Unit tests for guidance/utils/actuation_envelope.py (plain-value, no ROS)."""
import numpy as np
import pytest
from scipy.optimize import linprog

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)

_ALLOC = ThrustAllocator()


def _lp_max_scale(direction):
    """Independent reference implementation: maximize t s.t. A@f = t*direction,
    0 <= f <= fj_max. Used to cross-check wrench_envelope_halfspaces without
    sharing any code path with it."""
    A = _ALLOC.A
    n = A.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.hstack([A, -direction.reshape(-1, 1)])
    b_eq = np.zeros(A.shape[0])
    bounds = [(0.0, _ALLOC.fj_max)] * n + [(0.0, None)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.success
    return res.x[-1]


def test_pure_single_axis_scales_up_to_its_own_lp_max():
    # A single fan-aligned axis, scaled to just inside its own
    # independently-computed achievable max (_lp_max_scale), must be
    # feasible; scaled to just outside, infeasible. Derives the boundary
    # from the LP cross-check rather than a hardcoded config value (which
    # turned out to be a rounded approximation slightly outside the true
    # geometric max -- 0.181 vs the true ~0.18096 for +X).
    F, g = wrench_envelope_halfspaces(_ALLOC.A, _ALLOC.fj_max)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_max = _lp_max_scale(direction)

    inside = 0.999 * t_max * direction
    outside = 1.001 * t_max * direction
    assert np.all(F @ inside <= g + 1e-9)
    assert np.any(F @ outside > g + 1e-9)


def test_matches_independent_lp_support_function():
    # Cross-check against a from-scratch LP formulation (module docstring
    # claims these agree to float precision) -- this is what actually
    # caught the original bug: a diagonal 2-axis force request that's
    # individually within each axis's own max_force but not jointly
    # achievable.
    F, g = wrench_envelope_halfspaces(_ALLOC.A, _ALLOC.fj_max)
    y = np.array([0.181, 0.0996, 0.0, 0.0, 0.0, 0.0])
    d = y / np.linalg.norm(y)
    t_lp = _lp_max_scale(d)

    Fd = F @ d
    positive = Fd > 1e-12
    t_hull = np.min(g[positive] / Fd[positive])
    assert t_hull == pytest.approx(t_lp, rel=1e-4)
    # And confirm this combined request is NOT fully achievable (the whole
    # point of replacing the old independent per-axis box) -- y itself has
    # norm > t_lp, i.e. y lies outside the envelope.
    assert np.linalg.norm(y) > t_lp


def test_zero_wrench_is_feasible():
    F, g = wrench_envelope_halfspaces(_ALLOC.A, _ALLOC.fj_max)
    assert np.all(g >= -1e-9)


def test_rejects_too_many_fans():
    huge_matrix = np.zeros((6, 32))
    with pytest.raises(ValueError):
        wrench_envelope_halfspaces(huge_matrix, 0.06)
