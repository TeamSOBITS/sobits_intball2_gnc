"""Unit tests for guidance/segment_time/heuristic_segment_time_allocator.py
(plain-value, no ROS)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)


def test_straight_line_uses_distance_over_speed():
    allocator = HeuristicSegmentTimeAllocator(target_speed=2.0)
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    segment_times = allocator.allocate(waypoints)
    assert np.allclose(segment_times, [1.0, 2.0])


def test_zero_angle_gain_gives_pure_distance_allocation():
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, angle_time_gain=5.0)
    # Collinear waypoints: deviation angle is 0 at the interior point, so the
    # turn-angle boost must vanish even with a nonzero gain.
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    segment_times = allocator.allocate(waypoints)
    assert np.allclose(segment_times, [1.0, 1.0])


def test_sharp_turn_gets_more_time_than_gentle_turn():
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, angle_time_gain=1.0)

    gentle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]
    sharp = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]  # 90 degree turn

    gentle_times = allocator.allocate(gentle)
    sharp_times = allocator.allocate(sharp)

    # Both share the same first segment (length 1.0), so any difference in
    # its allocated time isolates the turn-angle boost.
    assert sharp_times[0] > gentle_times[0]


def test_output_never_below_min_segment_time():
    allocator = HeuristicSegmentTimeAllocator(
        target_speed=1000.0, min_segment_time=0.5,
    )
    waypoints = [[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]]
    segment_times = allocator.allocate(waypoints)
    assert np.all(segment_times >= 0.5)


def test_rejects_single_waypoint():
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0)
    with pytest.raises(ValueError):
        allocator.allocate([[0.0, 0.0, 0.0]])


def test_rejects_non_positive_target_speed():
    with pytest.raises(ValueError):
        HeuristicSegmentTimeAllocator(target_speed=0.0)


def test_rejects_negative_angle_time_gain():
    with pytest.raises(ValueError):
        HeuristicSegmentTimeAllocator(target_speed=1.0, angle_time_gain=-1.0)


def test_max_accel_none_reproduces_pure_distance_allocation():
    allocator = HeuristicSegmentTimeAllocator(target_speed=2.0, max_accel=None)
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert np.allclose(allocator.allocate(waypoints), [1.0])


def test_max_accel_raises_time_above_naive_estimate_cruise_case():
    # v_cap=1.0, a_max=1.0, distance=4.0: reaches cruise speed
    # (2*d_accel = 1.0 <= 4.0), trapezoid time = 2*(v/a) + (d-2*d_accel)/v
    # = 2 + 3 = 5.0, vs. the naive distance/target_speed = 4.0.
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    waypoints = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert np.allclose(allocator.allocate(waypoints), [5.0])


def test_max_accel_triangular_profile_for_short_segment():
    # v_cap=1.0, a_max=1.0, distance=0.5: never reaches cruise speed
    # (2*d_accel = 1.0 > 0.5), triangular time = 2*sqrt(a*d)/a
    # = 2*sqrt(0.5) ~= 1.4142, vs. the naive distance/target_speed = 0.5.
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    waypoints = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    assert np.allclose(allocator.allocate(waypoints), [2.0 * np.sqrt(0.5)])


def test_rejects_non_positive_max_accel():
    with pytest.raises(ValueError):
        HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=0.0)
