"""Unit tests for guidance/utils/attitude_reference.py (plain-value, no ROS)."""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    IDENTITY_QUAT,
    compute_camera_relative_quat,
    compute_look_at_quat,
    compute_q_des,
)

FORWARD = (1.0, 0.0, 0.0)


def test_low_speed_holds_previous_q_des():
    prev = np.array([0.0, 0.0, 0.7071, 0.7071])
    q = compute_q_des([0.001, 0.0, 0.0], prev, speed_threshold=0.05)
    assert np.allclose(q, prev)


def test_low_speed_with_no_previous_defaults_to_identity():
    q = compute_q_des([0.0, 0.0, 0.0], None, speed_threshold=0.05)
    assert np.allclose(q, IDENTITY_QUAT)


def test_already_aligned_gives_identity():
    q = compute_q_des([2.0, 0.0, 0.0], None, speed_threshold=0.05, forward_axis=FORWARD)
    assert np.allclose(q, IDENTITY_QUAT)


def test_points_forward_axis_along_v_des():
    v_des = [0.0, 1.0, 0.0]
    q = compute_q_des(v_des, None, speed_threshold=0.05, forward_axis=FORWARD)
    rotated = quat_rotate(q, np.array(FORWARD))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_diagonal_direction():
    v_des = [1.0, 1.0, 0.0]
    q = compute_q_des(v_des, None, speed_threshold=0.05, forward_axis=FORWARD)
    rotated = quat_rotate(q, np.array(FORWARD))
    expected = np.array(v_des) / np.linalg.norm(v_des)
    assert np.allclose(rotated, expected, atol=1e-9)


def test_antiparallel_direction_is_valid_unit_quaternion():
    v_des = [-1.0, 0.0, 0.0]
    q = compute_q_des(v_des, None, speed_threshold=0.05, forward_axis=FORWARD)
    assert np.isclose(np.linalg.norm(q), 1.0)
    rotated = quat_rotate(q, np.array(FORWARD))
    assert np.allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-9)


def test_speed_exactly_at_threshold_updates():
    q = compute_q_des([0.05, 0.0, 0.0], None, speed_threshold=0.05, forward_axis=FORWARD)
    assert np.allclose(q, IDENTITY_QUAT)


def _geodesic_angle(q_a, q_b):
    dot = float(np.clip(abs(np.dot(q_a, q_b)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def test_rate_limit_caps_a_large_direction_reversal():
    # Facing +X (identity); v_des reverses to -X, which would otherwise
    # demand a full 180-degree jump in one tick.
    prev = IDENTITY_QUAT.copy()
    q = compute_q_des([-1.0, 0.0, 0.0], prev, speed_threshold=0.05,
                       forward_axis=FORWARD, dt=0.02, max_angular_rate=np.radians(20.0))
    stepped = _geodesic_angle(prev, q)
    assert stepped <= np.radians(20.0) * 0.02 + 1e-9


def test_rate_limit_is_no_op_when_target_already_within_reach():
    prev = IDENTITY_QUAT.copy()
    v_des = [1.0, 0.001, 0.0]  # a tiny direction change
    unlimited = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD)
    limited = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD,
                             dt=0.02, max_angular_rate=np.radians(20.0))
    assert np.allclose(unlimited, limited, atol=1e-9)


def test_rate_limit_disabled_without_dt_or_max_rate():
    prev = IDENTITY_QUAT.copy()
    v_des = [-1.0, 0.0, 0.0]
    unlimited = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD)
    same = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD, dt=0.02)
    assert np.allclose(unlimited, same)


def test_rate_limit_converges_over_repeated_steps():
    # Repeatedly stepping toward a fixed reversed target should monotonically
    # close the gap and eventually reach it.
    prev = IDENTITY_QUAT.copy()
    target_v = [-1.0, 0.0, 0.0]
    dt = 0.02
    max_rate = np.radians(20.0)
    angles = []
    q = prev
    for _ in range(int(np.pi / (max_rate * dt)) + 5):
        q = compute_q_des(target_v, q, speed_threshold=0.05, forward_axis=FORWARD,
                           dt=dt, max_angular_rate=max_rate)
        full_target = compute_q_des(target_v, None, speed_threshold=0.05, forward_axis=FORWARD)
        angles.append(_geodesic_angle(q, full_target))
    assert angles[-1] < 1e-6
    assert all(a2 <= a1 + 1e-9 for a1, a2 in zip(angles, angles[1:]))


def test_preserves_current_roll_instead_of_shortest_arc_convention():
    # Facing +X with a 90-degree roll baked in (rotation about +X itself).
    prev = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    v_des = [0.0, 1.0, 0.0]
    q = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD)
    # Still points forward_axis along v_des (the actual task).
    rotated = quat_rotate(q, np.array(FORWARD))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)
    # But unlike the plain shortest-arc solution, some other body axis that
    # was fixed relative to `prev`'s roll (here, +Z) should have moved
    # continuously with the roll already present in `prev`, not jumped to
    # whatever the shortest-arc convention happens to produce.
    plain = compute_q_des(v_des, None, speed_threshold=0.05, forward_axis=FORWARD)
    z_axis = np.array([0.0, 0.0, 1.0])
    assert not np.allclose(quat_rotate(q, z_axis), quat_rotate(plain, z_axis), atol=1e-3)


def test_preserve_roll_is_noop_when_prev_has_no_twist_about_axis():
    # prev is a pure heading change from identity with zero roll about
    # forward_axis -- nothing to preserve, so behaves like the old plain
    # shortest-arc result.
    prev = IDENTITY_QUAT.copy()
    v_des = [0.0, 1.0, 0.0]
    q = compute_q_des(v_des, prev, speed_threshold=0.05, forward_axis=FORWARD)
    plain = compute_q_des(v_des, None, speed_threshold=0.05, forward_axis=FORWARD)
    assert np.allclose(q, plain, atol=1e-9)


def test_compute_look_at_quat_points_forward_axis_at_target():
    own_pos = [0.0, 0.0, 0.0]
    target_pos = [0.0, 2.0, 0.0]
    q = compute_look_at_quat(own_pos, target_pos, forward_axis=FORWARD)
    rotated = quat_rotate(q, np.array(FORWARD))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


def test_compute_look_at_quat_already_aligned_gives_identity():
    q = compute_look_at_quat([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], forward_axis=FORWARD)
    assert np.allclose(q, IDENTITY_QUAT)


def test_compute_look_at_quat_raises_when_positions_coincide():
    with np.testing.assert_raises(ValueError):
        compute_look_at_quat([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], forward_axis=FORWARD)


STEREO = (0.0, 1.0, 0.0)


def test_compute_camera_relative_quat_same_axis_returns_target_unchanged():
    q_target = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.1**2 - 0.2**2 - 0.3**2)])
    q = compute_camera_relative_quat(q_target, FORWARD, FORWARD)
    assert np.allclose(q, q_target)


def test_compute_camera_relative_quat_points_to_axis_at_what_from_axis_would_see():
    q_target = IDENTITY_QUAT.copy()
    q = compute_camera_relative_quat(q_target, FORWARD, STEREO)
    # Under q_target, FORWARD (body +X) points along world +X. The result
    # should instead point STEREO (body +Y) along that same world +X.
    rotated = quat_rotate(q, np.array(STEREO))
    expected = quat_rotate(q_target, np.array(FORWARD))
    assert np.allclose(rotated, expected, atol=1e-9)
