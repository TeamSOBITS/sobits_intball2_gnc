"""Tests for guidance/trajectory/min_snap_trajectory.py (static_min_snap)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.trajectory.min_snap_trajectory import (
    MinSnapInfeasibleError,
    MinSnapTrajectory,
)

WAYPOINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5], [0.0, 2.0, 0.0]]
Q0 = np.array([0.0, 0.0, 0.3826834, 0.9238795])
V_MAX = 0.5
A_MAX = 0.031


def _sample_all(traj, n=2000):
    ts = np.linspace(0.0, traj.global_total_duration, n)
    samples = [traj.sample(t) for t in ts]
    return ts, samples


def test_passes_through_every_waypoint():
    traj = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX)
    times = np.concatenate([[0.0], np.cumsum(traj.segment_times)])
    for t, wp in zip(times, WAYPOINTS):
        p, _v, _a, _q = traj.sample(t)
        assert np.allclose(p, wp, atol=1e-6)
    assert np.isclose(times[-1], traj.global_total_duration)


def test_starts_and_ends_at_rest():
    traj = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX)
    for t in (0.0, traj.global_total_duration):
        _p, v, a, _q = traj.sample(t)
        assert np.allclose(v, 0.0, atol=1e-9)
        assert np.allclose(a, 0.0, atol=1e-9)


def test_holds_the_final_state_past_the_end():
    traj = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX)
    p, v, a, _q = traj.sample(traj.global_total_duration + 5.0)
    assert np.allclose(p, WAYPOINTS[-1], atol=1e-6)
    assert np.allclose(v, 0.0, atol=1e-9)
    assert np.allclose(a, 0.0, atol=1e-9)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("optimize", [True, False])
def test_respects_the_limits_and_reaches_one(order, optimize):
    traj = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX, order=order,
                             optimize_segment_times=optimize)
    _ts, samples = _sample_all(traj)
    v_peak = max(np.linalg.norm(v) for _p, v, _a, _q in samples)
    a_peak = max(np.linalg.norm(a) for _p, _v, a, _q in samples)
    assert v_peak <= V_MAX * (1.0 + 1e-3)
    assert a_peak <= A_MAX * (1.0 + 1e-3)
    # time scaling makes one of the two limits active
    assert v_peak >= V_MAX * 0.99 or a_peak >= A_MAX * 0.99


def test_optimizing_segment_times_keeps_the_ratio_free_but_lowers_snap_cost():
    from sobits_intball2_gnc.guidance.trajectory.generation.min_snap_trajectory_generator import (
        solve_min_snap,
    )
    plain = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX, optimize_segment_times=False)
    opt = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX, optimize_segment_times=True)
    # compare the snap cost at the same total time (cost scales with T^-7)
    total = plain.global_total_duration
    plain_cost = solve_min_snap(WAYPOINTS, plain.segment_times).cost
    opt_times = opt.segment_times * total / opt.global_total_duration
    opt_cost = solve_min_snap(WAYPOINTS, opt_times).cost
    assert opt_cost < plain_cost


def test_attitude_is_fixed_at_q0():
    traj = MinSnapTrajectory(WAYPOINTS, Q0, V_MAX, A_MAX)
    _ts, samples = _sample_all(traj, n=50)
    for _p, _v, _a, q in samples:
        assert np.allclose(q, Q0)
    for t in (0.0, 0.5 * traj.global_total_duration, traj.global_total_duration + 1.0):
        omega, omega_dot = traj.sample_body_angular(t)
        assert np.allclose(omega, 0.0)
        assert np.allclose(omega_dot, 0.0)


def test_two_waypoints_is_a_straight_rest_to_rest_line():
    traj = MinSnapTrajectory([[0.0, 0.0, 0.0], [1.0, 0.5, 0.3]], Q0, V_MAX, A_MAX)
    direction = np.array([1.0, 0.5, 0.3]) / np.linalg.norm([1.0, 0.5, 0.3])
    _ts, samples = _sample_all(traj, n=100)
    for p, _v, _a, _q in samples:
        assert np.allclose(np.cross(p, direction), 0.0, atol=1e-9)


def test_coincident_waypoints_are_infeasible():
    with pytest.raises(MinSnapInfeasibleError):
        MinSnapTrajectory([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          Q0, V_MAX, A_MAX)


@pytest.mark.parametrize("kwargs", [
    {"order": 5}, {"order": 1}, {"max_vel": 0.0}, {"max_accel": -1.0},
])
def test_rejects_bad_arguments(kwargs):
    args = {"max_vel": V_MAX, "max_accel": A_MAX}
    args.update(kwargs)
    with pytest.raises(ValueError):
        MinSnapTrajectory(WAYPOINTS, Q0, **args)


def test_rejects_a_single_waypoint():
    with pytest.raises(ValueError):
        MinSnapTrajectory([[0.0, 0.0, 0.0]], Q0, V_MAX, A_MAX)
