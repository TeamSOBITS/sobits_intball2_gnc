"""Unit tests for solve_quintic_hermite_coeffs (docs/
2026-09-01_replanning_minco_v4_production_port_plan.md Phase 4 local layer).
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector
from sobits_intball2_gnc.guidance.utils.quintic_hermite import (
    solve_quintic_hermite_coeffs,
)


def test_matches_boundary_conditions():
    p0, v0, a0 = np.array([1.0, 2.0, 3.0]), np.array([0.1, -0.2, 0.0]), np.array([0.01, 0.0, -0.02])
    p1, v1, a1 = np.array([4.0, 1.0, 3.5]), np.array([0.3, 0.0, 0.1]), np.array([0.0, 0.02, 0.0])
    T = 1.7

    coeffs = solve_quintic_hermite_coeffs(p0, v0, a0, p1, v1, a1, T)

    assert coeffs.shape == (3, 6)
    assert np.allclose(evaluate_vector(coeffs, 0.0, order=0), p0)
    assert np.allclose(evaluate_vector(coeffs, 0.0, order=1), v0)
    assert np.allclose(evaluate_vector(coeffs, 0.0, order=2), a0)
    assert np.allclose(evaluate_vector(coeffs, T, order=0), p1)
    assert np.allclose(evaluate_vector(coeffs, T, order=1), v1)
    assert np.allclose(evaluate_vector(coeffs, T, order=2), a1)


def test_zero_boundary_conditions_gives_zero_polynomial():
    zero3 = np.zeros(3)
    coeffs = solve_quintic_hermite_coeffs(zero3, zero3, zero3, zero3, zero3, zero3, 1.0)
    assert np.allclose(coeffs, 0.0)


def test_single_axis_shape():
    coeffs = solve_quintic_hermite_coeffs(
        np.array([0.0]), np.array([1.0]), np.array([0.0]),
        np.array([1.0]), np.array([1.0]), np.array([0.0]), 1.0,
    )
    assert coeffs.shape == (1, 6)


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        solve_quintic_hermite_coeffs(
            np.zeros(3), np.zeros(3), np.zeros(3),
            np.zeros(2), np.zeros(3), np.zeros(3), 1.0,
        )


def test_nonpositive_duration_rejected():
    zero3 = np.zeros(3)
    with pytest.raises(ValueError):
        solve_quintic_hermite_coeffs(zero3, zero3, zero3, zero3, zero3, zero3, 0.0)
    with pytest.raises(ValueError):
        solve_quintic_hermite_coeffs(zero3, zero3, zero3, zero3, zero3, zero3, -1.0)
