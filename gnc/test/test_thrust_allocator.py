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
    """Pin the numeric behavior of the weighted-bounded-least-squares allocator.

    These values are NOT equivalent to the older NNLS-then-uniform-rescale
    allocator: that version's post-hoc rescale, triggered whenever the raw
    NNLS solution exceeded fj_max on any one fan, punished every fan equally
    -- for this fan geometry's short moment arms, even a small torque request
    drove the raw solution far above fj_max, so the rescale crushed
    co-requested force by 40-90%+ even though that force alone was nowhere
    near saturating (see thrust_allocator.py module docstring and
    docs/archive/achieved/trajectory_force_duration_investigation.md).
    lsq_linear solves the ``0 <= f <= fj_max`` bounded problem directly (no
    rescale step), with a per-channel weighting so force/torque residuals are
    compared as a fraction of their own reference budget rather than raw
    units. For unsaturated single-axis requests like these, the numbers are
    close to (not identical to) the old NNLS ones -- small differences come
    from the weighting and a minimal-total-thrust tie-break term.
    """
    alloc = ThrustAllocator()
    x_force = alloc.allocate([0.05, 0, 0], [0, 0, 0])
    expected_x = [0.0, 0.072697, 0.103554, 0.0, 0.533186, 0.533226, 0.528206, 0.523074]
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(x_force, expected_x))

    z_torque = alloc.allocate([0, 0, 0], [0, 0, 0.01])
    expected_z = [1.0, 0.000561, 0.000561, 1.0, 0.000561, 0.000561, 1.0, 1.0]
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


def test_torque_request_does_not_crush_coincident_force():
    # Regression guard for the 2026-08-20 bug: a modest torque request used
    # to trigger a uniform rescale of all 8 fans, crushing a co-requested
    # force that was nowhere near saturating on its own (see module docstring
    # of thrust_allocator.py). Force should stay close to fully achieved
    # regardless of how much torque is asked for alongside it.
    alloc = ThrustAllocator()
    force = [0.08, 0, 0]
    baseline = alloc.allocate(force, [0, 0, 0])
    baseline_fx = alloc.A @ [(d / alloc.kj) ** 2 for d in baseline]
    assert math.isclose(baseline_fx[0], 0.08, abs_tol=1e-4)

    for tz in (0.01, 0.05, 0.1, 0.2):
        duties = alloc.allocate(force, [0, 0, tz])
        thrust = [(d / alloc.kj) ** 2 for d in duties]
        achieved_fx = (alloc.A @ thrust)[0]
        # Old behavior crushed this to <0.05 (a >=40% loss) even for tz as
        # small as 0.01; the fix should keep it within a few percent of 0.08.
        assert math.isclose(achieved_fx, 0.08, abs_tol=0.01), (
            f"tz={tz}: achieved_fx={achieved_fx} far from requested 0.08"
        )


def test_uneven_geometry_length_raises():
    import pytest

    with pytest.raises(ValueError):
        ThrustAllocator(fan_positions=[0.0, 1.0], fan_vectors=[0.0, 1.0])
