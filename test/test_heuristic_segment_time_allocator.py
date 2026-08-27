"""Unit tests for guidance/segment_time/heuristic_segment_time_allocator.py
(plain-value, no ROS)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.segment_time.base_segment_time_allocator import (
    SegmentTimeInfeasibleError,
)
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


# --- v0 (start velocity) extension --------------------------------------
# docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md 2 節/3 節:
# T_min/T_max below come from the exact cubic-Hermite polynomial
# (m0=v_parallel, m1=0), not the rest-to-rest _trapezoidal_time model above.


def test_v0_requires_max_accel():
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0)  # max_accel=None
    with pytest.raises(ValueError):
        allocator.allocate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[0.1, 0.0, 0.0])


def test_v0_rejects_wrong_shape():
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    with pytest.raises(ValueError):
        allocator.allocate([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[0.1, 0.0])


def test_v0_only_affects_first_segment():
    # v_parallel=2.0 -> T_max=3d/v0=1.5s, well below the rest-to-rest
    # trapezoidal estimate (2.5s) this config would otherwise use, so the
    # v0-aware cap is guaranteed to actually change segment 0.
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5, max_accel=1.0)
    without_v0 = allocator.allocate(waypoints)
    with_v0 = allocator.allocate(waypoints, v0=[2.0, 0.0, 0.0])
    assert not np.isclose(with_v0[0], without_v0[0])
    assert np.isclose(with_v0[1], without_v0[1])


def test_v0_raises_time_above_naive_estimate():
    # d=1, target_speed=5.0, a_max=1.0: rest-to-rest triangle time = 2.0
    # (see test_max_accel_triangular_profile_for_short_segment's formula),
    # independent of v0 in this small-v0 regime. But the v0-aware T_min at
    # a near-zero forward speed converges to the *exact* rest-to-rest bound
    # sqrt(6d/a_max)=sqrt(6)~=2.449 -- strictly larger than the 2.0 the
    # (approximate) rest-to-rest model above would give -- so the naive
    # value must be raised. v0 is kept tiny (not exactly 0) so this also
    # exercises the v_parallel>0 code path, not the v_parallel<=0 fallback.
    allocator = HeuristicSegmentTimeAllocator(target_speed=5.0, max_accel=1.0)
    segment_times = allocator.allocate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[1e-6, 0.0, 0.0]
    )
    assert np.isclose(segment_times[0], np.sqrt(6.0), atol=1e-3)


def test_v0_caps_time_below_naive_estimate():
    # d=1, target_speed=0.1 (slow cruise -> long rest-to-rest trapezoid time
    # ~10.1s), but v0=2.0 m/s residual speed -> T_max=3d/v0=1.5s. The naive
    # (v0-unaware) estimate would badly overshoot the target; it must be
    # capped down to T_max, not just floored.
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.1, max_accel=1.0)
    segment_times = allocator.allocate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[2.0, 0.0, 0.0]
    )
    assert np.isclose(segment_times[0], 1.5, atol=1e-6)


def test_v0_negative_parallel_component_falls_back_to_exact_rest_to_rest():
    # v0 points away from the target (e.g. TF noise, or a residual velocity
    # from an overshoot) -- the parallel component is clamped to 0, and the
    # exact rest-to-rest bound sqrt(6d/a_max) is used (design doc 2-4 節: this
    # is *not* the same number as the old approximate trapezoidal triangle
    # formula, which underestimates by a factor of sqrt(1.5)).
    allocator = HeuristicSegmentTimeAllocator(target_speed=5.0, max_accel=1.0)
    segment_times = allocator.allocate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[-1.0, 0.0, 0.0]
    )
    assert np.isclose(segment_times[0], np.sqrt(6.0), atol=1e-6)


# --- v_perp (start velocity component perpendicular to delta0) ----------
# docs/2026-08-25_v0_aware_time_allocation_lateral_velocity_fix.md: T_min
# must also account for v0's component orthogonal to delta0, via the
# closed-form T_min_perp = 4*v_perp/a_max (no matching T_max -- that channel
# never overshoots).


def test_v0_perpendicular_component_raises_time_above_naive_estimate():
    # delta0 along +x, v0 purely along +y (v_parallel=0 -> T_max=inf).
    # T_min_perp = 4*v_perp/a_max = 4*10/1 = 40, far above both the naive
    # distance/target_speed estimate (1.0) and the v_parallel=0 fallback
    # bound sqrt(6*d/a_max) ~= 2.449.
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    segment_times = allocator.allocate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[0.0, 10.0, 0.0]
    )
    assert np.isclose(segment_times[0], 40.0, atol=1e-6)


def test_v0_perpendicular_component_can_trigger_infeasible():
    # v_parallel=1.0 -> T_max=3d/v_parallel=3.0. v_perp=1.0 ->
    # T_min_perp=4*v_perp/a_max=4.0, which exceeds T_max: no segment time
    # satisfies both the overshoot bound (parallel) and the acceleration
    # bound (perpendicular) simultaneously.
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    with pytest.raises(SegmentTimeInfeasibleError):
        allocator.allocate(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[1.0, 1.0, 0.0]
        )


def test_v0_perpendicular_extraction_uses_raw_not_clamped_projection():
    # v0=[-1, 2, 0], delta0=[1, 0, 0]: the raw projection onto delta0 is
    # -1 (clamped to 0 for v_parallel/T_max purposes), but extracting
    # v_perp must subtract the *raw* -1, not the clamped 0 -- otherwise
    # v_perp_vec would wrongly be v0 itself (norm sqrt(5)) instead of the
    # true perpendicular residual [0, 2, 0] (norm 2). With a_max=1.0,
    # T_min_perp=4*2/1=8.0, above the v_parallel=0 fallback bound
    # sqrt(6*1/1)~=2.449, so this distinguishes the two computations.
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, max_accel=1.0)
    segment_times = allocator.allocate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], v0=[-1.0, 2.0, 0.0]
    )
    assert np.isclose(segment_times[0], 8.0, atol=1e-6)


def test_v0_aware_bounds_never_infeasible_for_positive_v_parallel():
    # docs/archive/achieved/session_2026-08-24_heuristic_segment_time_allocator_v0_extension.md 3 節/
    # _v0_aware_bounds's docstring: T_min < T_max is provable for any
    # d > 0, a_max > 0, v_parallel > 0 -- spot-check across a wide range
    # (including regimes prone to floating-point cancellation) that no
    # SegmentTimeInfeasibleError-worthy T_min > T_max case slips through.
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d = 10.0 ** rng.uniform(-6, 2)
        v_parallel = 10.0 ** rng.uniform(-6, 2)
        a_max = 10.0 ** rng.uniform(-6, 2)
        t_min, t_max = HeuristicSegmentTimeAllocator._v0_aware_bounds(
            d, v_parallel, a_max
        )
        assert t_min <= t_max
