"""Unit tests for ThrustAllocator (plain-value constructor, no ROS)."""
import math

from sobits_intball2_gnc.control.utils.thrust_allocator import (
    DEFAULT_FAN_POSITIONS,
    DEFAULT_FAN_VECTORS,
    ThrustAllocator,
)


def test_zero_wrench_gives_zero_duties():
    alloc = ThrustAllocator()
    assert alloc.allocate([0, 0, 0], [0, 0, 0]) == [0.0] * 8


def test_all_duties_in_unit_range():
    alloc = ThrustAllocator()
    for force, torque in (
        ([0.05, 0, 0], [0, 0, 0]),
        ([-0.05, 0, 0], [0, 0, 0]),
        ([0, 0.03, -0.02], [0.005, 0, 0]),
        ([0, 0, 0], [0, 0, 0.01]),
    ):
        duties = alloc.allocate(force, torque)
        assert len(duties) == 8
        assert all(0.0 <= d <= 1.0 for d in duties)


def test_flat_geometry_reconstructs_eight_fans():
    alloc = ThrustAllocator(
        fan_positions=DEFAULT_FAN_POSITIONS, fan_vectors=DEFAULT_FAN_VECTORS
    )
    assert alloc.fan_count == 8
    assert alloc.A.shape == (6, 8)


def test_golden_duties_regression():
    """Pin the numeric behavior (equivalent to the pre-refactor allocator)."""
    alloc = ThrustAllocator()
    x_force = alloc.allocate([0.05, 0, 0], [0, 0, 0])
    expected_x = [0.0, 0.0, 0.0, 0.0, 0.375427, 0.379057, 0.367911, 0.36417]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(x_force, expected_x))

    z_torque = alloc.allocate([0, 0, 0], [0, 0, 0.01])
    expected_z = [0.781574, 0.0, 0.0, 0.781574, 0.0, 0.0, 0.781574, 0.781574]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(z_torque, expected_z))


def test_uneven_geometry_length_raises():
    import pytest

    with pytest.raises(ValueError):
        ThrustAllocator(fan_positions=[0.0, 1.0], fan_vectors=[0.0, 1.0])
