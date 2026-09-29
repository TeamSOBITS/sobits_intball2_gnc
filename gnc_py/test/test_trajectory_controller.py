"""Unit tests for TrajectoryController (plain-value, no ROS)."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from sobits_intball2_gnc.control.utils.trajectory_controller import TrajectoryController

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
# Not identity / pure yaw / 180deg, so q vs conj(q) and multiplication order matter.
BODY = Rotation.from_euler("ZYX", [90.0, 0.0, 25.0], degrees=True)
DESIRED = Rotation.from_euler("ZYX", [30.0, -15.0, 10.0], degrees=True)


def test_feedforward_force_matches_mass_times_acceleration():
    ctrl = TrajectoryController(mass=2.0, kp_pos=[0, 0, 0], kd_pos=[0, 0, 0],
                                 vel_filter_alpha=1.0, max_force=100.0)
    # At the target already, zero velocity error -> only feedforward remains.
    force = ctrl.compute(
        stamp=0.0, pos_now=[1.0, 2.0, 3.0], quat_now=IDENTITY_QUAT,
        p_des=[1.0, 2.0, 3.0], v_des=[0.0, 0.0, 0.0], a_des=[0.5, 0.0, 0.0],
    )
    assert math.isclose(force[0], 1.0, abs_tol=1e-9)  # m * a_des = 2.0 * 0.5
    assert math.isclose(force[1], 0.0, abs_tol=1e-9)
    assert math.isclose(force[2], 0.0, abs_tol=1e-9)


def test_feedforward_force_is_rotated_into_non_symmetric_body_frame():
    ctrl = TrajectoryController(mass=2.0, kp_pos=[0, 0, 0], kd_pos=[0, 0, 0],
                                 vel_filter_alpha=1.0, max_force=100.0)
    a_des = np.array([0.3, -0.1, 0.2])
    force = ctrl.compute(
        stamp=0.0, pos_now=[1.0, 2.0, 3.0], quat_now=BODY.as_quat(),
        p_des=[1.0, 2.0, 3.0], v_des=[0.0, 0.0, 0.0], a_des=a_des,
    )
    assert np.allclose(force, BODY.inv().apply(2.0 * a_des), atol=1e-12)


def test_feedback_and_feedforward_share_the_non_symmetric_body_frame():
    ctrl = TrajectoryController(mass=2.0, kp_pos=[0.635] * 3, kd_pos=[0, 0, 0],
                                 vel_filter_alpha=1.0, max_force=100.0)
    p_err, a_des = np.array([0.2, -0.1, 0.05]), np.array([0.0, 0.05, -0.02])
    force = ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=BODY.as_quat(),
        p_des=p_err, v_des=[0.0, 0.0, 0.0], a_des=a_des,
    )
    expected = BODY.inv().apply(0.635 * p_err + 2.0 * a_des)
    assert np.allclose(force, expected, atol=1e-12)


def test_position_error_produces_feedback_force_toward_target():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[2.0, 2.0, 2.0],
                                 kd_pos=[0, 0, 0], vel_filter_alpha=1.0,
                                 max_force=100.0)
    force = ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[1.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    # kp * (p_des - p_now) = 2.0 * 1.0
    assert math.isclose(force[0], 2.0, abs_tol=1e-9)


def test_velocity_damping_uses_error_against_desired_velocity():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[0, 0, 0], kd_pos=[3.0, 0, 0],
                                 vel_filter_alpha=1.0, max_force=100.0)
    # Prime the finite-difference velocity estimate: moving +1 m/s in x.
    ctrl.compute(stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
                 p_des=[0.0, 0.0, 0.0], v_des=[1.0, 0.0, 0.0], a_des=[0, 0, 0])
    force = ctrl.compute(
        stamp=1.0, pos_now=[1.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[0.0, 0.0, 0.0], v_des=[1.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    # vel_now == v_des (both 1.0 m/s) -> velocity error is zero -> no damping force.
    assert math.isclose(force[0], 0.0, abs_tol=1e-6)


def test_output_clamped_to_max_force():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[1000.0, 1000.0, 1000.0],
                                 kd_pos=[0, 0, 0], vel_filter_alpha=1.0,
                                 max_force=0.1)
    force = ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[10.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    assert all(abs(f) <= 0.1 + 1e-12 for f in force)


def test_output_clamped_per_axis_when_max_force_is_a_vector():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[1000.0, 1000.0, 1000.0],
                                 kd_pos=[0, 0, 0], vel_filter_alpha=1.0,
                                 max_force=[0.181, 0.0996, 0.122])
    force = ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[10.0, 10.0, 10.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    assert math.isclose(force[0], 0.181, abs_tol=1e-12)
    assert math.isclose(force[1], 0.0996, abs_tol=1e-12)
    assert math.isclose(force[2], 0.122, abs_tol=1e-12)


def test_attitude_error_produces_opposing_torque():
    quat_off = [0.3, 0.0, 0.0, 0.954]  # small rotation about x, away from q_des
    ctrl = TrajectoryController(kp_att=[10.0, 10.0, 10.0], kd_att=[0, 0, 0],
                                att_filter_alpha=1.0, max_torque=100.0)
    torque = ctrl.compute_attitude(
        stamp=0.0, quat_now=quat_off, q_des=IDENTITY_QUAT,
    )
    # qe = quat_now (q_des is identity) -> torque = -kp_att * qe[:3]
    assert math.isclose(torque[0], -3.0, abs_tol=1e-6)
    assert math.isclose(torque[1], 0.0, abs_tol=1e-9)
    assert math.isclose(torque[2], 0.0, abs_tol=1e-9)


def test_attitude_zero_error_produces_zero_torque():
    ctrl = TrajectoryController(kp_att=[10.0, 10.0, 10.0], max_torque=100.0)
    torque = ctrl.compute_attitude(
        stamp=0.0, quat_now=IDENTITY_QUAT, q_des=IDENTITY_QUAT,
    )
    assert all(math.isclose(v, 0.0, abs_tol=1e-9) for v in torque)


def test_attitude_output_clamped_to_max_torque():
    quat_off = [0.3, 0.0, 0.0, 0.954]
    ctrl = TrajectoryController(kp_att=[1000.0, 1000.0, 1000.0], max_torque=0.01)
    torque = ctrl.compute_attitude(
        stamp=0.0, quat_now=quat_off, q_des=IDENTITY_QUAT,
    )
    assert all(abs(v) <= 0.01 + 1e-12 for v in torque)


def test_attitude_reset_forgets_rate_estimate():
    quat_off = [0.3, 0.0, 0.0, 0.954]
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[5.0, 0, 0],
                                att_filter_alpha=1.0, max_torque=100.0)
    ctrl.compute_attitude(stamp=0.0, quat_now=IDENTITY_QUAT, q_des=IDENTITY_QUAT)
    ctrl.reset()
    # No prior sample after reset -> rate estimate is zero again despite the
    # large jump in tracking error, matching the very first call's behavior.
    torque = ctrl.compute_attitude(
        stamp=10.0, quat_now=quat_off, q_des=IDENTITY_QUAT,
    )
    assert math.isclose(torque[0], 0.0, abs_tol=1e-9)


def test_reset_forgets_velocity_estimate():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[0, 0, 0], kd_pos=[3.0, 0, 0],
                                 vel_filter_alpha=1.0, max_force=100.0)
    ctrl.compute(stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
                 p_des=[0.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0, 0, 0])
    ctrl.reset()
    # Immediately after reset, no prior sample -> velocity estimate is zero
    # again, matching the very first call's behavior (no damping contribution
    # despite the position having jumped 5m, which would otherwise look like
    # a huge instantaneous velocity).
    force = ctrl.compute(
        stamp=10.0, pos_now=[5.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[5.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    assert math.isclose(force[0], 0.0, abs_tol=1e-9)


def test_last_force_raw_reflects_pre_clamp_value():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[1000.0, 1000.0, 1000.0],
                                 kd_pos=[0, 0, 0], vel_filter_alpha=1.0,
                                 max_force=0.1)
    force = ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[10.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    # Output is clamped to max_force, but the raw request (kp*10000) is not.
    assert math.isclose(force[0], 0.1, abs_tol=1e-12)
    assert ctrl.last_force_raw[0] > 0.1


def test_last_torque_raw_reflects_pre_clamp_value():
    quat_off = [0.3, 0.0, 0.0, 0.954]
    ctrl = TrajectoryController(kp_att=[1000.0, 1000.0, 1000.0], max_torque=0.01)
    torque = ctrl.compute_attitude(
        stamp=0.0, quat_now=quat_off, q_des=IDENTITY_QUAT,
    )
    assert math.isclose(torque[0], -0.01, abs_tol=1e-12)
    assert ctrl.last_torque_raw[0] < -0.01


def test_last_qe_vec_and_omega_err_expose_p_d_terms_separately():
    quat_off = [0.3, 0.0, 0.0, 0.954]
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[5.0, 0, 0],
                                att_filter_alpha=1.0, max_torque=100.0)
    ctrl.compute_attitude(stamp=0.0, quat_now=IDENTITY_QUAT, q_des=IDENTITY_QUAT)
    ctrl.compute_attitude(stamp=1.0, quat_now=quat_off, q_des=IDENTITY_QUAT)
    # qe_vec (P-term input) is the current error's vector part.
    assert math.isclose(ctrl.last_qe_vec[0], 0.3, abs_tol=1e-6)
    # omega_err (D-term input) is nonzero: the error jumped from 0 to 0.3
    # over dt=1.0s.
    assert math.isclose(ctrl.last_omega_err[0], 0.3, abs_tol=1e-6)


def test_raw_state_reset_by_reset():
    ctrl = TrajectoryController(mass=1.0, kp_pos=[1000.0, 1000.0, 1000.0],
                                 kd_pos=[0, 0, 0], vel_filter_alpha=1.0,
                                 max_force=0.1)
    ctrl.compute(
        stamp=0.0, pos_now=[0.0, 0.0, 0.0], quat_now=IDENTITY_QUAT,
        p_des=[10.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
    )
    assert ctrl.last_force_raw[0] > 0.1
    ctrl.reset()
    assert ctrl.last_force_raw == [0.0, 0.0, 0.0]
    assert ctrl.last_torque_raw == [0.0, 0.0, 0.0]


def test_repeated_tf_stamp_does_not_bias_velocity_estimate_low():
    # Control ticks faster than TF updates, so some ticks see the same stamp.
    ctrl = TrajectoryController(mass=1.0, kp_pos=[0, 0, 0], kd_pos=[1.0, 0, 0],
                                 vel_filter_alpha=0.3, max_force=100.0)
    v, tf_dt, force = 0.2, 0.024, None
    for tick in range(500):
        stamp = (tick * 0.02 // tf_dt) * tf_dt
        force = ctrl.compute(stamp=stamp, pos_now=[v * stamp, 0.0, 0.0],
                             quat_now=IDENTITY_QUAT, p_des=[v * stamp, 0.0, 0.0],
                             v_des=[v, 0.0, 0.0], a_des=[0.0, 0.0, 0.0])
    assert math.isclose(force[0], 0.0, abs_tol=1e-6)


def test_repeated_tf_stamp_does_not_bias_attitude_rate_low():
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[1.0, 0, 0],
                                att_filter_alpha=0.3, max_torque=100.0)
    rate, tf_dt = 0.1, 0.024
    torques = []
    for tick in range(500):
        stamp = (tick * 0.02 // tf_dt) * tf_dt
        half = 0.5 * rate * stamp
        quat_now = [math.sin(half), 0.0, 0.0, math.cos(half)]
        torques.append(ctrl.compute_attitude(stamp=stamp, quat_now=quat_now,
                                             q_des=IDENTITY_QUAT)[0])
    # Steady rate -> filtered d(qe_vec)/dt settles at 0.5*rate*cos(half).
    half = 0.5 * rate * stamp
    assert math.isclose(torques[-1], -0.5 * rate * math.cos(half), rel_tol=0.02)


def _rot_z_quat(angle):
    return [0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)]


def test_attitude_feedforward_off_ignores_alpha_des():
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[0, 0, 0],
                                max_torque=100.0, inertia=2.0,
                                attitude_feedforward=False)
    torque = ctrl.compute_attitude(stamp=0.0, quat_now=IDENTITY_QUAT,
                                   q_des=IDENTITY_QUAT, alpha_des=[1.0, 0.0, 0.0])
    assert torque == [0.0, 0.0, 0.0]


def test_attitude_feedforward_adds_inertia_times_alpha_at_zero_error():
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[0, 0, 0],
                                max_torque=100.0, inertia=2.0,
                                attitude_feedforward=True)
    torque = ctrl.compute_attitude(stamp=0.0, quat_now=IDENTITY_QUAT,
                                   q_des=IDENTITY_QUAT, alpha_des=[0.5, -1.0, 0.25])
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(torque, [1.0, -2.0, 0.5]))


def test_attitude_feedforward_rotates_alpha_into_current_body_frame():
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[0, 0, 0],
                                max_torque=100.0, inertia=1.0,
                                attitude_feedforward=True)
    # Body is yawed +90deg from q_des: q_des's x axis is the body's -y axis.
    torque = ctrl.compute_attitude(stamp=0.0, quat_now=_rot_z_quat(math.pi / 2),
                                   q_des=IDENTITY_QUAT, alpha_des=[1.0, 0.0, 0.0])
    assert all(math.isclose(a, b, abs_tol=1e-12)
               for a, b in zip(torque, [0.0, -1.0, 0.0]))


def _closed_loop_max_yaw_error_deg(attitude_feedforward):
    """Isotropic rigid body tracking a 100deg minimum-jerk yaw in 8s, with
    config/gnc_params.yaml's gains and the ~0.008Nm physical torque ceiling."""
    inertia, total, turn, dt, substeps = 0.0136, 8.0, math.radians(100.0), 0.02, 20
    ctrl = TrajectoryController(kp_att=[0.20] * 3, kd_att=[0.0626] * 3,
                                att_filter_alpha=1.0, max_torque=0.008,
                                inertia=inertia,
                                attitude_feedforward=attitude_feedforward)
    yaw, yaw_rate, max_err = 0.0, 0.0, 0.0
    for tick in range(int(1.5 * total / dt)):
        t = tick * dt
        s = min(t / total, 1.0)
        yaw_des = turn * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
        yaw_acc_des = (turn / total ** 2) * (60 * s - 180 * s ** 2 + 120 * s ** 3) \
            if s < 1.0 else 0.0
        max_err = max(max_err, abs(math.degrees(yaw - yaw_des)))
        torque = ctrl.compute_attitude(stamp=t, quat_now=_rot_z_quat(yaw),
                                       q_des=_rot_z_quat(yaw_des),
                                       alpha_des=[0.0, 0.0, yaw_acc_des])
        for _ in range(substeps):
            yaw_rate += torque[2] / inertia * (dt / substeps)
            yaw += yaw_rate * (dt / substeps)
    return max_err


def test_attitude_feedforward_cuts_closed_loop_tracking_lag():
    pd_only = _closed_loop_max_yaw_error_deg(attitude_feedforward=False)
    with_ff = _closed_loop_max_yaw_error_deg(attitude_feedforward=True)
    assert pd_only > 1.0
    assert with_ff < 0.05 * pd_only


def test_attitude_torque_opposes_body_frame_offset_from_non_identity_q_des():
    ctrl = TrajectoryController(kp_att=[3.0] * 3, kd_att=[0, 0, 0],
                                att_filter_alpha=1.0, max_torque=100.0)
    offset_rotvec_body = np.radians(20.0) * np.array([0.6, -0.48, 0.64])
    current = DESIRED * Rotation.from_rotvec(offset_rotvec_body)
    torque = ctrl.compute_attitude(stamp=0.0, quat_now=current.as_quat(),
                                   q_des=DESIRED.as_quat())
    angle = np.linalg.norm(offset_rotvec_body)
    expected = -3.0 * np.sin(0.5 * angle) * offset_rotvec_body / angle
    assert np.allclose(torque, expected, atol=1e-12)


def test_attitude_feedforward_rotates_alpha_from_non_identity_q_des_into_body():
    ctrl = TrajectoryController(kp_att=[0, 0, 0], kd_att=[0, 0, 0],
                                max_torque=100.0, inertia=2.0,
                                attitude_feedforward=True)
    offset = Rotation.from_rotvec(np.radians(40.0) * np.array([0.0, 0.6, 0.8]))
    current = DESIRED * offset
    alpha_des = np.array([1.0, -0.5, 0.25])
    torque = ctrl.compute_attitude(stamp=0.0, quat_now=current.as_quat(),
                                   q_des=DESIRED.as_quat(), alpha_des=alpha_des)
    assert np.allclose(torque, 2.0 * offset.inv().apply(alpha_des), atol=1e-12)
