"""rotvec_rates_to_body_rates / sample_body_angular vs. the body rates of the
sampled quaternion itself (finite difference of q(t), no Jacobian involved)."""
import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    right_jacobian,
    rotvec_rates_to_body_rates,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import ToppraTrajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.constraints.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import IDENTITY_QUAT

H = 1e-4
# Multi-axis turn reaching ~130 deg over t in [0, 1], where r_dot != omega.
R0 = np.array([1.2, 0.9, -0.6])
R1 = np.array([0.4, -0.9, 0.5])
R2 = np.array([0.3, 0.2, -0.4])
TURNING_ROUTE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


def _body_omega_from_quat(q_of_t, t, h=H):
    return quat_log(quat_mul(quat_conj(q_of_t(t - h)), q_of_t(t + h))) / (2.0 * h)


def _body_omega_dot_from_quat(q_of_t, t, h=H):
    return (_body_omega_from_quat(q_of_t, t + h, h)
            - _body_omega_from_quat(q_of_t, t - h, h)) / (2.0 * h)


def _poly_r(t):
    return R0 + R1 * t + 0.5 * R2 * t ** 2


def _poly_q(t):
    return quat_exp(_poly_r(t))


@pytest.mark.parametrize("t", [0.0, 0.3, 0.7, 1.0])
def test_matches_quaternion_finite_difference_for_multi_axis_large_rotation(t):
    omega, omega_dot = rotvec_rates_to_body_rates(_poly_r(t), R1 + R2 * t, R2)
    assert np.allclose(omega, _body_omega_from_quat(_poly_q, t), atol=1e-6)
    assert np.allclose(omega_dot, _body_omega_dot_from_quat(_poly_q, t), atol=1e-4)


def test_raw_rotvec_derivatives_are_wrong_at_large_multi_axis_rotation():
    t = 0.5
    omega_true = _body_omega_from_quat(_poly_q, t)
    r_dot = R1 + R2 * t
    assert np.linalg.norm(r_dot - omega_true) > 0.2 * np.linalg.norm(omega_true)


def test_fixed_axis_rotation_needs_no_correction():
    axis = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    omega, omega_dot = rotvec_rates_to_body_rates(2.5 * axis, 0.7 * axis, -0.3 * axis)
    assert np.allclose(omega, 0.7 * axis, atol=1e-9)
    assert np.allclose(omega_dot, -0.3 * axis, atol=1e-6)


def test_right_jacobian_is_identity_at_zero():
    assert np.allclose(right_jacobian(np.zeros(3)), np.eye(3))


def _errors_vs_sampled_quat(traj):
    def q_of_t(t):
        return traj.sample(t)[3]

    total = traj.global_total_duration
    omega_err, omega_dot_err, omega_peak, omega_dot_peak = [], [], 0.0, 0.0
    for t in np.linspace(0.02 * total, 0.98 * total, 200):
        omega, omega_dot = traj.sample_body_angular(t)
        omega_err.append(np.linalg.norm(omega - _body_omega_from_quat(q_of_t, t)))
        omega_dot_err.append(
            np.linalg.norm(omega_dot - _body_omega_dot_from_quat(q_of_t, t)))
        omega_peak = max(omega_peak, np.linalg.norm(omega))
        omega_dot_peak = max(omega_dot_peak, np.linalg.norm(omega_dot))
    return np.array(omega_err), np.array(omega_dot_err), omega_peak, omega_dot_peak


def _toppra_turning():
    alloc = ThrustAllocator()
    return ToppraTrajectory(
        TURNING_ROUTE, IDENTITY_QUAT.copy(),
        max_vel=0.5, mass=3.216, inertia=0.0136,
        wrench_envelope=wrench_envelope_halfspaces(alloc.A, alloc.fj_max),
        max_angular_rate=np.radians(90.0), forward_axis=(1.0, 0.0, 0.0),
        face_travel=True,
    )


def test_toppra_sample_body_angular_matches_sampled_quaternion():
    traj = _toppra_turning()
    omega_err, omega_dot_err, omega_peak, omega_dot_peak = _errors_vs_sampled_quat(traj)
    assert omega_err.max() < 1e-3 * omega_peak
    # ParametrizeConstAccel makes r_ddot piecewise constant: the finite
    # difference straddling a knot is off there, so compare the median.
    assert np.median(omega_dot_err) < 1e-2 * omega_dot_peak


def test_minco_sample_body_angular_matches_sampled_quaternion():
    pytest.importorskip("minco_native_py")
    from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

    traj = MincoTrajectory(TURNING_ROUTE, IDENTITY_QUAT.copy())
    omega_err, omega_dot_err, omega_peak, omega_dot_peak = _errors_vs_sampled_quat(traj)
    assert omega_err.max() < 1e-3 * omega_peak
    assert omega_dot_err.max() < 1e-2 * omega_dot_peak


def test_sample_body_angular_is_zero_once_q_is_held_at_the_end():
    traj = _toppra_turning()
    for t in (traj.global_total_duration, traj.global_total_duration + 5.0):
        omega, omega_dot = traj.sample_body_angular(t)
        assert np.all(omega == 0.0) and np.all(omega_dot == 0.0)


def test_static_tracker_exposes_rates_of_its_last_sample():
    traj = _toppra_turning()
    tracker = StaticTrajectoryTracker(traj)
    t = 0.5 * traj.global_total_duration
    tracker.sample(t)
    expected = traj.sample_body_angular(t)
    assert np.allclose(tracker.last_body_angular[0], expected[0])
    assert np.allclose(tracker.last_body_angular[1], expected[1])
