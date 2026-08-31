"""Contract test for guidance/trajectory_generation/ implementations
(docs/architecture_guidelines.md 5 節: shared properties every
BaseTrajectoryGenerator implementation must satisfy).

MinSnapTrajectoryGenerator is intentionally NOT parametrized here: its core
solve was decided not to be implemented (2026-08-24, see
guidance/trajectory_generation/min_snap_trajectory_generator.py's module
docstring), so it cannot satisfy this contract.
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

GENERATOR_CLASSES = [HermiteSplineTrajectoryGenerator]

WAYPOINTS = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.5, 0.0], [0.0, 1.5, 1.0]]
SEGMENT_TIMES = [3.0, 2.0, 4.0]


@pytest.mark.parametrize("generator_cls", GENERATOR_CLASSES)
def test_coeffs_shape_matches_contract(generator_cls):
    coeffs = generator_cls().generate(WAYPOINTS, SEGMENT_TIMES)
    assert coeffs.shape == (len(SEGMENT_TIMES), 3, 8)


@pytest.mark.parametrize("generator_cls", GENERATOR_CLASSES)
def test_passes_through_every_waypoint(generator_cls):
    coeffs = generator_cls().generate(WAYPOINTS, SEGMENT_TIMES)
    for i, T in enumerate(SEGMENT_TIMES):
        p_start = evaluate_vector(coeffs[i], 0.0, order=0)
        p_end = evaluate_vector(coeffs[i], T, order=0)
        assert np.allclose(p_start, WAYPOINTS[i], atol=1e-6)
        assert np.allclose(p_end, WAYPOINTS[i + 1], atol=1e-6)


@pytest.mark.parametrize("generator_cls", GENERATOR_CLASSES)
def test_position_and_velocity_continuous_across_boundaries(generator_cls):
    coeffs = generator_cls().generate(WAYPOINTS, SEGMENT_TIMES)
    for i in range(len(SEGMENT_TIMES) - 1):
        p_end = evaluate_vector(coeffs[i], SEGMENT_TIMES[i], order=0)
        p_next_start = evaluate_vector(coeffs[i + 1], 0.0, order=0)
        v_end = evaluate_vector(coeffs[i], SEGMENT_TIMES[i], order=1)
        v_next_start = evaluate_vector(coeffs[i + 1], 0.0, order=1)
        assert np.allclose(p_end, p_next_start, atol=1e-6)
        assert np.allclose(v_end, v_next_start, atol=1e-6)


@pytest.mark.parametrize("generator_cls", GENERATOR_CLASSES)
def test_rejects_fewer_than_two_waypoints(generator_cls):
    with pytest.raises(ValueError):
        generator_cls().generate([[0.0, 0.0, 0.0]], [])
