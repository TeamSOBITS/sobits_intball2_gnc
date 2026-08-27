"""Unit tests for TrajectoryController (plain-value, no ROS)."""
import math

from sobits_intball2_gnc.control.utils.trajectory_controller import TrajectoryController

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


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
