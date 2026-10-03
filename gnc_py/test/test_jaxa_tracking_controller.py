"""JaxaTrackingController (jaxa_control_node's logic) with the real ported JAXA controllers."""
import os

import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.jaxa_control_params import (
    load_jaxa_control,
    make_attitude_controller,
    make_position_controller,
    unflatten,
)
from sobits_intball2_gnc.control.utils.jaxa_tracking_controller import (
    JaxaTrackingController,
    ema_alpha,
)

CFG = load_jaxa_control(os.path.join(os.path.dirname(__file__), "..", "config", "jaxa_control.yaml"))
Q0 = [0.0, 0.0, 0.0, 1.0]
G0 = [0.0, 0.0, 0.0]


def make():
    return JaxaTrackingController(make_position_controller(CFG), make_attitude_controller(CFG),
                                  vel_filter_alpha=0.3, trajectory_timeout=0.2, tf_timeout=1.0)


def setpoint(p):
    return {"p_des": p, "v_des": G0, "a_des": G0, "q_des": Q0, "omega_des": G0}


def test_ema_alpha_keeps_the_50hz_time_constant():
    assert ema_alpha(0.3, 0.02) == pytest.approx(0.3)
    assert ema_alpha(0.3, 0.005) == pytest.approx(1 - 0.7 ** 0.25)
    # Two half steps decay like one full step.
    assert (1 - ema_alpha(0.3, 0.01)) ** 2 == pytest.approx(1 - ema_alpha(0.3, 0.02))


def test_missing_stale_tf_or_imu_gives_zero_wrench():
    c = make()
    assert c.step(1.0, None, G0, None, None) == ([0.0] * 3, [0.0] * 3)
    assert c.step(5.0, ([1, 0, 0], Q0, 3.0), G0, None, None) == ([0.0] * 3, [0.0] * 3)
    assert "stale" in c.status
    assert c.step(5.0, ([1, 0, 0], Q0, 5.0), None, None, None) == ([0.0] * 3, [0.0] * 3)


def test_first_pose_is_held_and_checkpoint_pulls_toward_it():
    c = make()
    f, t = c.step(1.0, ([1.0, 2.0, 3.0], Q0, 1.0), G0, None, None)
    assert np.allclose(f, 0.0) and np.allclose(t, 0.0) and c.status == "holding"
    c.set_checkpoints([([1.1, 2.0, 3.0], Q0)])
    f, _ = c.step(1.02, ([1.0, 2.0, 3.0], Q0, 1.02), G0, None, None)
    m, kp = CFG["ctl_body"]["mass"], CFG["pos_ctl"]["kp"]
    assert f[0] == pytest.approx(m * kp * 0.1)


def test_same_tf_stamp_reuses_the_previous_command():
    c = make()
    c.set_checkpoints([([0.1, 0.0, 0.0], Q0)])
    first = c.step(1.0, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    again = c.step(1.005, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    assert again == first


def test_trajectory_tracking_then_hold_where_it_ended():
    c = make()
    c.step(1.0, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    f, _ = c.step(1.02, ([0.0, 0.0, 0.0], Q0, 1.02), G0, setpoint([0.0, 0.2, 0.0]), 1.02)
    assert c.trajectory_active and f[1] > 0.0
    # Setpoint stops arriving: hold the pose at the falling edge.
    c.step(1.5, ([0.0, 0.15, 0.0], Q0, 1.5), G0, setpoint([0.0, 0.2, 0.0]), 1.02)
    assert not c.trajectory_active and c.status == "holding"
    f, _ = c.step(1.52, ([0.0, 0.15, 0.0], Q0, 1.52), G0, setpoint([0.0, 0.2, 0.0]), 1.02)
    # Only velocity damping remains: no position error toward the old start or the setpoint.
    m, kd = CFG["ctl_body"]["mass"], CFG["pos_ctl"]["kd"]
    assert f[1] == pytest.approx(-m * kd * c.velocity[1], abs=1e-9)


def test_checkpoint_received_during_trajectory_is_kept_after_it():
    c = make()
    c.step(1.0, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    c.step(1.02, ([0.0, 0.0, 0.0], Q0, 1.02), G0, setpoint([0.0, 0.2, 0.0]), 1.02)
    c.set_checkpoints([([0.0, 0.3, 0.0], Q0)])
    c.step(1.5, ([0.0, 0.0, 0.0], Q0, 1.5), G0, setpoint([0.0, 0.2, 0.0]), 1.02)
    f, _ = c.step(1.52, ([0.0, 0.0, 0.0], Q0, 1.52), G0, None, None)
    assert f[1] > 0.0  # pulled toward the align checkpoint, not held in place


def test_velocity_estimate_converges_on_constant_motion():
    c = make()
    for k in range(200):
        t = 1.0 + 0.024 * k  # ~42 Hz TF
        c.step(t, ([0.1 * (t - 1.0), 0.0, 0.0], Q0, t), G0, None, None)
    assert c.velocity[0] == pytest.approx(0.1, abs=1e-6)


def test_unflatten_rebuilds_ros_dotted_parameters():
    flat = {"pos_ctl.kp": 0.6, "fan.Wp.fan01.Fx": 0.0, "fan.n_saturation": 4}
    assert unflatten(flat) == {"pos_ctl": {"kp": 0.6}, "fan": {"Wp": {"fan01": {"Fx": 0.0}}, "n_saturation": 4}}


def test_updated_flags_new_commands_only():
    c = make()
    c.step(1.0, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    assert c.updated
    c.step(1.005, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    assert not c.updated
    c.set_checkpoints([([0.1, 0.0, 0.0], Q0)])
    c.step(1.02, ([0.0, 0.0, 0.0], Q0, 1.02), G0, None, None)
    assert c.updated
    c.step(1.03, None, G0, None, None)
    assert c.updated  # nonzero -> zero once
    c.step(1.04, None, G0, None, None)
    assert not c.updated


def test_release_drops_old_target_and_holds_the_pose_seen_next():
    c = make()
    c.set_checkpoints([([0.5, 0.0, 0.0], Q0)])
    f, _ = c.step(1.0, ([0.0, 0.0, 0.0], Q0, 1.0), G0, None, None)
    assert f[0] > 0.0
    c.release()
    f, t = c.step(2.0, ([3.0, 1.0, 0.0], Q0, 2.0), G0, None, None)
    assert np.allclose(f, 0.0) and np.allclose(t, 0.0) and c.status == "holding"
    assert c.velocity == [0.0, 0.0, 0.0]
