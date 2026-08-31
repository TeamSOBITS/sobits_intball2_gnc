"""Unit tests for guidance/utils/trajectory.py (plain-value, no ROS).

Uses hand-built stub coefficients (straight-line, constant-velocity per
segment) rather than real min_snap output, per
docs/min_snap_interface_contract.md's plan to test Trajectory independently
of the (not-yet-implemented) min_snap solver.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory.trajectory import Trajectory

# Two segments, each 2s: (0,0,0) -> (1,0,0) -> (1,1,0), constant velocity.
WAYPOINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
SEGMENT_TIMES = [2.0, 2.0]


def _straight_line_coeffs():
    coeffs = np.zeros((2, 3, 8))
    coeffs[0, 0, 0], coeffs[0, 0, 1] = 0.0, 0.5  # segment 0: x(t) = 0.5t
    coeffs[1, 0, 0] = 1.0  # segment 1: x(t) = 1 (constant)
    coeffs[1, 1, 0], coeffs[1, 1, 1] = 0.0, 0.5  # segment 1: y(t) = 0.5t (local tau)
    return coeffs


def test_sample_at_start():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p, v, a, q = traj.sample(0.0)
    assert np.allclose(p, [0.0, 0.0, 0.0])
    assert np.allclose(v, [0.5, 0.0, 0.0])
    assert np.allclose(a, [0.0, 0.0, 0.0])


def test_sample_mid_first_segment():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p, v, _a, _q = traj.sample(1.0)
    assert np.allclose(p, [0.5, 0.0, 0.0])
    assert np.allclose(v, [0.5, 0.0, 0.0])


def test_sample_uses_local_tau_across_segment_boundary():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p_end_of_seg0, _, _, _ = traj.sample(2.0 - 1e-9)
    p_start_of_seg1, v_start_of_seg1, _, _ = traj.sample(2.0)
    # Continuous position across the boundary...
    assert np.allclose(p_end_of_seg0, p_start_of_seg1, atol=1e-6)
    # ...but segment 1's coeffs are evaluated at tau=0, not global t=2.
    assert np.allclose(p_start_of_seg1, [1.0, 0.0, 0.0])
    assert np.allclose(v_start_of_seg1, [0.0, 0.5, 0.0])


def test_sample_mid_second_segment():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p, v, _a, _q = traj.sample(3.0)
    assert np.allclose(p, [1.0, 0.5, 0.0])
    assert np.allclose(v, [0.0, 0.5, 0.0])


def test_sample_past_total_duration_holds_last_waypoint():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p, v, a, _q = traj.sample(4.0)
    assert np.allclose(p, [1.0, 1.0, 0.0])
    assert np.allclose(v, [0.0, 0.0, 0.0])
    assert np.allclose(a, [0.0, 0.0, 0.0])

    p2, _v2, _a2, _q2 = traj.sample(100.0)
    assert np.allclose(p2, [1.0, 1.0, 0.0])


def test_negative_time_is_clamped_to_start():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    p, v, _a, _q = traj.sample(-1.0)
    assert np.allclose(p, [0.0, 0.0, 0.0])
    assert np.allclose(v, [0.5, 0.0, 0.0])


def test_q_des_tracks_velocity_direction_and_holds_at_rest():
    traj = Trajectory(
        WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(), attitude_speed_threshold=0.05
    )
    _p0, _v0, _a0, q0 = traj.sample(0.0)  # v = [0.5, 0, 0], forward already aligned
    assert np.allclose(quat_rotate(q0, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-9)

    _p1, _v1, _a1, q1 = traj.sample(3.0)  # v = [0, 0.5, 0]
    assert np.allclose(quat_rotate(q1, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)

    _p2, _v2, _a2, q2 = traj.sample(4.0)  # v = 0 -> hold last attitude
    assert np.allclose(q2, q1)


def _geodesic_angle(q_a, q_b):
    dot = float(np.clip(abs(np.dot(q_a, q_b)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def test_max_angular_rate_caps_q_des_change_between_samples():
    # sample(1.9) is segment 0 (v = [0.5, 0, 0], forward already aligned ->
    # q_des = identity); sample(2.0), 0.1s later, is just past the segment
    # boundary (v = [0, 0.5, 0], a 90-degree direction change). The capped
    # step between the two q_des values should be far short of 90 degrees.
    traj = Trajectory(
        WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(),
        attitude_speed_threshold=0.05, max_angular_rate=np.radians(10.0),
    )
    _p0, _v0, _a0, q0 = traj.sample(1.9)
    _p1, _v1, _a1, q1 = traj.sample(2.0)
    stepped = _geodesic_angle(q0, q1)
    assert stepped <= np.radians(10.0) * 0.1 + 1e-6


def test_max_angular_rate_none_preserves_prior_instantaneous_behavior():
    traj_unlimited = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(),
                                 attitude_speed_threshold=0.05)
    traj_unlimited.sample(0.0)
    _p, _v, _a, q_unlimited = traj_unlimited.sample(3.0)  # v direction reverses to +y

    traj_limited = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(),
                               attitude_speed_threshold=0.05, max_angular_rate=None)
    traj_limited.sample(0.0)
    _p, _v, _a, q_limited = traj_limited.sample(3.0)
    assert np.allclose(q_unlimited, q_limited)


def _stub_coeffs_with_velocity(p0, v):
    """Single-segment stub starting at ``p0`` with constant velocity ``v`` --
    for exercising ``replace_coeffs``'s state-carryover behavior against a
    re-plan, like a real re-planning tracker would produce (docs/archive/
    achieved/2026-08-24_trajectory_state_carryover_design.md)."""
    coeffs = np.zeros((1, 3, 8))
    for axis in range(3):
        coeffs[0, axis, 0] = p0[axis]
        coeffs[0, axis, 1] = v[axis]
    return coeffs


def test_replace_coeffs_preserves_last_q_des_and_last_sample_t():
    traj = Trajectory(
        WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(),
        attitude_speed_threshold=0.05,
    )
    traj.sample(0.0)
    _p, _v, _a, q_before = traj.sample(3.0)  # v = [0, 0.5, 0] -> q_des rotated

    # Re-plan at global t=5.0: a fresh single-segment leg with the SAME
    # velocity direction traj was already tracking, so q_des shouldn't need
    # to move at all -- if replace_coeffs had reset _last_q_des to identity
    # (docs/guidance_realtime_replanning_design.md 6-6 節's failure mode),
    # this sample would jump back toward facing +X instead of staying at q_before.
    traj.replace_coeffs(
        [[1.0, 0.5, 0.0], [1.0, 2.0, 0.0]], [3.0],
        _stub_coeffs_with_velocity([1.0, 0.5, 0.0], [0.0, 0.5, 0.0]), 5.0,
    )
    _p, _v, _a, q_after = traj.sample(5.0)
    assert np.allclose(q_after, q_before, atol=1e-6)


def test_replace_coeffs_evaluates_new_polynomial_in_local_time():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    traj.replace_coeffs(
        [[1.0, 0.5, 0.0], [1.0, 2.0, 0.0]], [3.0],
        _stub_coeffs_with_velocity([1.0, 0.5, 0.0], [0.5, 0.0, 0.0]), 5.0,
    )
    # t_local = 0 at the moment of replan (global t == t_origin == 5.0).
    p_at_origin, v_at_origin, _a, _q = traj.sample(5.0)
    assert np.allclose(p_at_origin, [1.0, 0.5, 0.0])
    assert np.allclose(v_at_origin, [0.5, 0.0, 0.0])
    # One second later: local tau = 1.0, not global t = 6.0.
    p_later, _v, _a, _q = traj.sample(6.0)
    assert np.allclose(p_later, [1.5, 0.5, 0.0])


def test_global_total_duration_accounts_for_replan_origin():
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs())
    assert traj.global_total_duration == traj.total_duration == 4.0
    traj.replace_coeffs(
        [[1.0, 0.5, 0.0], [1.0, 2.0, 0.0]], [3.0],
        _stub_coeffs_with_velocity([1.0, 0.5, 0.0], [0.5, 0.0, 0.0]), 5.0,
    )
    assert traj.total_duration == 3.0
    assert traj.global_total_duration == 8.0


def test_initial_q_des_seeds_first_sample_instead_of_identity():
    # A min-jerk trajectory has v=0 at t=0 by construction, so without seeding,
    # sample(0) would fall back to compute_q_des's IDENTITY_QUAT default --
    # unrelated to the vehicle's actual attitude (docs/
    # trajectory_force_duration_investigation.md 6-3).
    seed = np.array([0.0, 0.0, 0.7071, 0.7071])
    traj = Trajectory(WAYPOINTS, SEGMENT_TIMES, _straight_line_coeffs(),
                       attitude_speed_threshold=1.0,  # so v=[0.5,0,0] still counts as "low speed"
                       initial_q_des=seed)
    _p, _v, _a, q0 = traj.sample(0.0)
    assert np.allclose(q0, seed)
