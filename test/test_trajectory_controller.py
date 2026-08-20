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
