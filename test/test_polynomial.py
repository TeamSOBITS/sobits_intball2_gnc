"""Unit tests for guidance/utils/polynomial.py (plain-value, no ROS)."""
import numpy as np

from sobits_intball2_gnc.guidance.utils.polynomial import evaluate, evaluate_vector

# p(t) = 1 + 2t + 3t^2 + 4t^3
COEFFS = [1.0, 2.0, 3.0, 4.0]


def test_position_at_zero_is_constant_term():
    assert evaluate(COEFFS, 0.0, order=0) == 1.0


def test_position_matches_hand_calculation():
    # p(2) = 1 + 4 + 12 + 32 = 49
    assert np.isclose(evaluate(COEFFS, 2.0, order=0), 49.0)


def test_velocity_matches_hand_calculation():
    # v(t) = 2 + 6t + 12t^2 -> v(2) = 2 + 12 + 48 = 62
    assert np.isclose(evaluate(COEFFS, 2.0, order=1), 62.0)


def test_acceleration_matches_hand_calculation():
    # a(t) = 6 + 24t -> a(2) = 6 + 48 = 54
    assert np.isclose(evaluate(COEFFS, 2.0, order=2), 54.0)


def test_derivative_order_beyond_degree_is_zero():
    assert evaluate(COEFFS, 5.0, order=4) == 0.0


def test_evaluate_vector_applies_per_axis():
    coeffs = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])  # p_axis(t) = k * t
    result = evaluate_vector(coeffs, 2.0, order=0)
    assert np.allclose(result, [2.0, 4.0, 6.0])
