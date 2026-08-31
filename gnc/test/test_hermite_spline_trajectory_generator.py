"""Unit tests for
guidance/trajectory_generation/hermite_spline_trajectory_generator.py
(plain-value, no ROS)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

WAYPOINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
SEGMENT_TIMES = [2.0, 3.0]


def test_output_shape():
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES)
    assert coeffs.shape == (2, 3, 8)
    # Degree-3 stand-in: coefficients above tau**3 must stay zero.
    assert np.allclose(coeffs[:, :, 4:], 0.0)


def test_passes_through_every_waypoint():
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES)
    p_seg0_start = evaluate_vector(coeffs[0], 0.0, order=0)
    p_seg0_end = evaluate_vector(coeffs[0], SEGMENT_TIMES[0], order=0)
    p_seg1_start = evaluate_vector(coeffs[1], 0.0, order=0)
    p_seg1_end = evaluate_vector(coeffs[1], SEGMENT_TIMES[1], order=0)

    assert np.allclose(p_seg0_start, WAYPOINTS[0])
    assert np.allclose(p_seg0_end, WAYPOINTS[1])
    assert np.allclose(p_seg1_start, WAYPOINTS[1])
    assert np.allclose(p_seg1_end, WAYPOINTS[2])


def test_velocity_continuous_across_boundary():
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES)
    v_end_seg0 = evaluate_vector(coeffs[0], SEGMENT_TIMES[0], order=1)
    v_start_seg1 = evaluate_vector(coeffs[1], 0.0, order=1)
    assert np.allclose(v_end_seg0, v_start_seg1)


def test_starts_and_ends_at_rest():
    # Only holds for the default v0=None (rest-to-rest) start -- see
    # test_v0_overrides_start_tangent below.
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES)
    v_start = evaluate_vector(coeffs[0], 0.0, order=1)
    v_end = evaluate_vector(coeffs[-1], SEGMENT_TIMES[-1], order=1)
    assert np.allclose(v_start, [0.0, 0.0, 0.0])
    assert np.allclose(v_end, [0.0, 0.0, 0.0])


def test_v0_overrides_start_tangent():
    v0 = [0.3, -0.1, 0.0]
    coeffs = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES, v0=v0)
    v_start = evaluate_vector(coeffs[0], 0.0, order=1)
    assert np.allclose(v_start, v0)
    # Passing through every waypoint and cross-segment continuity must still
    # hold with a nonzero start tangent (docs/archive/achieved/
    # session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md
    # 4-2 節).
    p_start = evaluate_vector(coeffs[0], 0.0, order=0)
    p_end = evaluate_vector(coeffs[-1], SEGMENT_TIMES[-1], order=0)
    assert np.allclose(p_start, WAYPOINTS[0])
    assert np.allclose(p_end, WAYPOINTS[-1])


def test_v0_does_not_affect_interior_tangents():
    coeffs_default = HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, SEGMENT_TIMES)
    coeffs_v0 = HermiteSplineTrajectoryGenerator().generate(
        WAYPOINTS, SEGMENT_TIMES, v0=[0.3, -0.1, 0.0]
    )
    v_interior_default = evaluate_vector(coeffs_default[0], SEGMENT_TIMES[0], order=1)
    v_interior_v0 = evaluate_vector(coeffs_v0[0], SEGMENT_TIMES[0], order=1)
    assert np.allclose(v_interior_default, v_interior_v0)


def test_v0_rejects_wrong_shape():
    with pytest.raises(ValueError):
        HermiteSplineTrajectoryGenerator().generate(
            WAYPOINTS, SEGMENT_TIMES, v0=[0.3, -0.1]
        )


def test_rejects_single_waypoint():
    with pytest.raises(ValueError):
        HermiteSplineTrajectoryGenerator().generate([[0.0, 0.0, 0.0]], [])


def test_rejects_mismatched_segment_times_length():
    with pytest.raises(ValueError):
        HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, [1.0])


def test_rejects_non_positive_segment_time():
    with pytest.raises(ValueError):
        HermiteSplineTrajectoryGenerator().generate(WAYPOINTS, [2.0, 0.0])
