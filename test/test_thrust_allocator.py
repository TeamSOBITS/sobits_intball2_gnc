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
    """Pin the numeric behavior of the NNLS-based allocator.

    These values are NOT equivalent to the old unconstrained-pinv-then-clamp
    allocator: that version silently discarded ~50% of the requested wrench
    for single-axis commands by clamping away one side of an opposing fan
    pair (see docs/phase0_findings.md). NNLS finds a non-negative solution
    that achieves the full requested wrench directly, so these duties are
    higher.
    """
    alloc = ThrustAllocator()
    x_force = alloc.allocate([0.05, 0, 0], [0, 0, 0])
    expected_x = [0.0, 0.074749, 0.105189, 0.0, 0.533507, 0.533507, 0.528245, 0.523035]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(x_force, expected_x))

    # This one saturates fj_max: achieving the full torque non-negatively
    # needs more raw thrust than the clamp did, so it now scales down at the
    # duty=1.0 ceiling instead of stopping short at ~0.78 (i.e. it uses more
    # of the available authority, not less).
    z_torque = alloc.allocate([0, 0, 0], [0, 0, 0.01])
    expected_z = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(z_torque, expected_z))


def test_single_axis_force_fully_achieved_not_halved():
    # Regression guard for the ~50% loss the old pinv+clamp allocator had on
    # any single-axis force/torque (see docs/phase0_findings.md observation 5):
    # verify the *achieved* wrench (not just clamped duties) matches request.
    alloc = ThrustAllocator(fj_max=10.0)  # fj_max high enough to not saturate
    for force in ([0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]):
        duties = alloc.allocate(force, [0, 0, 0])
        thrust = [(d / alloc.kj) ** 2 for d in duties]  # invert duty = kj*sqrt(f)
        achieved = alloc.A @ thrust
        assert all(math.isclose(a, b, abs_tol=1e-4)
                   for a, b in zip(achieved[:3], force))


def test_uneven_geometry_length_raises():
    import pytest

    with pytest.raises(ValueError):
        ThrustAllocator(fan_positions=[0.0, 1.0], fan_vectors=[0.0, 1.0])
