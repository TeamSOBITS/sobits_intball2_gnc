"""TeleopReference (keyboard-driven moving reference) and the key layout."""
import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.constraints.actuation_envelope import wrench_envelope_halfspaces
from sobits_intball2_gnc.teleop.keymap import axes_from_pressed
from sobits_intball2_gnc.teleop.reference import TeleopLimits, TeleopReference

DT = 0.02
Q0 = [0.0, 0.0, 0.0, 1.0]
P0 = np.zeros(3)
MASS, INERTIA = 3.216, 0.0136


def limits(**kw):
    base = dict(vmax=0.05, wmax=0.1, acc=(0.028, 0.0155, 0.019), alpha=(0.11, 0.17, 0.30),
                err_pos=0.02, err_att=np.radians(5.0))
    base.update(kw)
    return TeleopLimits(**base)


def make(envelope=None, **kw):
    ref = TeleopReference(limits(**kw), MASS, INERTIA, envelope=envelope)
    ref.reset(P0, Q0)
    return ref


def run(ref, key, seconds):
    """Step with the vehicle sitting exactly on the reference."""
    sp = None
    for _ in range(int(round(seconds / DT))):
        sp = ref.step(DT, key, ref.p, ref.q)
    return sp


def test_limits_reject_bad_values():
    with pytest.raises(ValueError):
        limits(vmax=0.0)
    with pytest.raises(ValueError):
        limits(acc=(0.1, 0.0, 0.1))
    with pytest.raises(ValueError):
        limits(resume=1.0)


def test_starts_at_rest_at_reset_pose():
    ref = TeleopReference(limits(), MASS, INERTIA)
    ref.reset([1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0])
    sp = ref.step(DT, np.zeros(6), [1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(sp.p, [1.0, 2.0, 3.0]) and np.allclose(sp.v, 0.0) and np.allclose(sp.a, 0.0)
    assert np.allclose(sp.q, [0.0, 0.0, 1.0, 0.0])
    assert ref.at_rest


def test_forward_accelerates_at_limit_then_holds_speed_cap():
    ref = make()
    sp = run(ref, [1, 0, 0, 0, 0, 0], 0.5)
    assert sp.v[0] == pytest.approx(0.028 * 0.5, rel=1e-6)   # slew-limited by acc_x
    sp = run(ref, [1, 0, 0, 0, 0, 0], 5.0)
    assert sp.v[0] == pytest.approx(0.05)                     # capped at vmax
    assert np.allclose(sp.a, 0.0, atol=1e-9)


def test_release_decelerates_to_rest_with_same_limit():
    ref = make()
    run(ref, [1, 0, 0, 0, 0, 0], 5.0)
    sp = run(ref, np.zeros(6), 0.5)
    assert sp.v[0] == pytest.approx(0.05 - 0.028 * 0.5, rel=1e-6)
    run(ref, np.zeros(6), 3.0)
    assert ref.at_rest


def test_direction_follows_reference_attitude():
    ref = make()
    run(ref, [0, 0, 0, 0, 0, 1], 5.0)  # yaw left a bit, then drive "forward" in the body frame
    yaw_q = ref.q
    ref.vb[:] = 0.0
    ref.wb[:] = 0.0
    sp = run(ref, [1, 0, 0, 0, 0, 0], 3.0)
    assert np.allclose(sp.v, quat_rotate(yaw_q, [np.linalg.norm(sp.v), 0, 0]), atol=1e-9)
    assert sp.v[1] > 0.0  # yawed left, so "forward" has a +y component in the reference frame


def test_signed_keys_map_to_expected_world_directions_at_identity():
    ref = make()
    sp = run(ref, [0, 1, 0, 0, 0, 0], 4.0)
    assert sp.v[1] > 0.0 and abs(sp.v[0]) < 1e-9
    ref = make()
    sp = run(ref, [0, 0, -1, 0, 0, 0], 4.0)
    assert sp.v[2] < 0.0


def test_centripetal_feedforward_matches_numeric_derivative_of_velocity():
    ref = make()
    run(ref, [1, 0, 0, 0, 0, 1], 6.0)   # steady arc
    v_prev = None
    errs = []
    for _ in range(50):
        sp = ref.step(DT, [1, 0, 0, 0, 0, 1], ref.p, ref.q)
        if v_prev is not None:
            errs.append(np.linalg.norm((sp.v - v_prev) / DT - sp.a))
        v_prev = sp.v
    assert max(errs) < 2e-4
    assert np.linalg.norm(sp.a) == pytest.approx(0.05 * 0.1, rel=1e-3)  # v * w


def test_arc_radius_is_speed_over_angular_rate():
    ref = make()
    run(ref, [1, 0, 0, 0, 0, 1], 8.0)   # reach steady speed and rate
    pts = []
    for _ in range(int(round(2 * np.pi / 0.1 / DT))):  # one full revolution
        pts.append(ref.step(DT, [1, 0, 0, 0, 0, 1], ref.p, ref.q).p)
    pts = np.array(pts)
    center = np.array([pts[:, 0].min() + pts[:, 0].max(), pts[:, 1].min() + pts[:, 1].max()]) / 2
    radii = np.linalg.norm(pts[:, :2] - center, axis=1)
    assert radii.mean() == pytest.approx(0.05 / 0.1, rel=0.02)
    assert radii.std() < 0.01


def test_stalls_when_error_exceeds_limit_and_resumes_below_hysteresis():
    ref = make()
    run(ref, [1, 0, 0, 0, 0, 0], 3.0)
    far = ref.p - np.array([0.03, 0.0, 0.0])           # 30 mm behind: over the 20 mm limit
    ref.step(DT, [1, 0, 0, 0, 0, 0], far, Q0)
    assert ref.stalled
    v_before = ref.vb[0]
    ref.step(DT, [1, 0, 0, 0, 0, 0], far, Q0)
    assert ref.vb[0] < v_before                        # decelerating although the key is held
    near = ref.p - np.array([0.0175, 0.0, 0.0])        # below 20 mm but above 0.8 * 20 mm
    ref.step(DT, [1, 0, 0, 0, 0, 0], near, Q0)
    assert ref.stalled
    ok = ref.p - np.array([0.010, 0.0, 0.0])
    ref.step(DT, [1, 0, 0, 0, 0, 0], ok, Q0)
    assert not ref.stalled


def test_stalls_on_attitude_error():
    ref = make()
    bad_q = [np.sin(np.radians(3.0)), 0.0, 0.0, np.cos(np.radians(3.0))]   # 6 deg about x > 5 deg
    ref.step(DT, np.zeros(6), P0, bad_q)
    assert ref.stalled


def test_envelope_shaping_keeps_wrench_inside_and_preserves_ratio():
    plant = ThrustAllocator()
    env = wrench_envelope_halfspaces(plant.A, plant.fj_max)
    fast = dict(acc=(0.5, 0.5, 0.5), alpha=(5.0, 5.0, 5.0))   # far beyond what the fans can do
    ref = make(envelope=env, **fast)
    ref.step(DT, [1, 0, 0, 0, 0, 1], P0, Q0)
    assert ref.scaled
    wrench = np.concatenate([MASS * ref.vb / DT, INERTIA * np.eye(3) @ ref.wb / DT])
    assert np.all(env[0] @ wrench <= env[1] + 1e-9)
    assert ref.vb[0] / (0.5 * DT) == pytest.approx(ref.wb[2] / (5.0 * DT), rel=1e-6)  # same scale on both


def test_envelope_not_used_at_default_limits():
    plant = ThrustAllocator()
    env = wrench_envelope_halfspaces(plant.A, plant.fj_max)
    ref = make(envelope=env)
    run(ref, [1, 0, 0, 0, 0, 1], 10.0)
    assert not ref.scaled


def test_key_values_are_clipped():
    ref = make()
    sp = run(ref, [5, 0, 0, 0, 0, 0], 5.0)
    assert sp.v[0] == pytest.approx(0.05)


def test_keymap_opposite_keys_cancel_and_combine():
    assert axes_from_pressed({"w"}) == (1.0, 0, 0, 0, 0, 0)
    assert axes_from_pressed({"w", "s"}) == (0, 0, 0, 0, 0, 0)
    assert axes_from_pressed({"w", "left"}) == (1.0, 0, 0, 0, 0, 1.0)
    assert axes_from_pressed({"q"}) == (0, 0, 0, -1.0, 0, 0)
    assert axes_from_pressed({"up", "d", "unknown"}) == (0, -1.0, 0, 0, 1.0, 0)


def test_make_limits_follow_the_fan_envelope():
    from sobits_intball2_gnc.teleop.reference import axis_maxima, make_limits
    plant = ThrustAllocator()
    env = wrench_envelope_halfspaces(plant.A, plant.fj_max)
    m = axis_maxima(env)
    assert m[:3] == pytest.approx([0.18096, 0.0996, 0.12216], abs=2e-5)
    assert m[3:] == pytest.approx([0.00302, 0.00455, 0.00819], abs=2e-5)
    lim = make_limits(env, MASS, INERTIA, 0.05, 0.1, acc_frac=0.5, alpha_frac=0.25, err_pos=0.02, err_att=0.087)
    assert lim.acc == pytest.approx([0.5 * 0.18096 / MASS, 0.5 * 0.0996 / MASS, 0.5 * 0.12216 / MASS], abs=1e-4)
    assert lim.alpha == pytest.approx([0.25 * 0.00302 / INERTIA, 0.25 * 0.00455 / INERTIA,
                                       0.25 * 0.00819 / INERTIA], abs=1e-3)


def test_link_releases_keys_when_gui_goes_quiet_and_delivers_estop_once():
    from sobits_intball2_gnc.teleop.link import TeleopLink
    from sobits_intball2_gnc.teleop.state import KeyState
    now = [0.0]
    link = TeleopLink(key_timeout=0.5, clock=lambda: now[0])
    assert link.take_key() == KeyState()                    # nothing received yet
    link.set_key(KeyState(axes=(1.0, 0, 0, 0, 0, 0), enable=True))
    assert link.take_key().axes[0] == 1.0 and link.take_key().enable
    now[0] = 0.6                                            # GUI silent for longer than the timeout
    assert link.take_key() == KeyState()
    link.set_key(KeyState(enable=True, estop=True))
    assert link.take_key().estop
    assert not link.take_key().estop                        # one-shot


def test_guidance_status_codes():
    from action_msgs.msg import GoalStatus
    from sobits_intball2_gnc.teleop.ros.guidance_status_subscriber import any_active
    assert any_active([GoalStatus.STATUS_EXECUTING])
    assert any_active([GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELING])
    assert not any_active([GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED])
    assert not any_active([])


def test_error_limits_follow_the_speed_caps():
    from sobits_intball2_gnc.teleop.reference import error_limits
    assert error_limits(0.05, 0.1)[0] == pytest.approx(0.020)
    assert error_limits(0.10, 0.2)[0] == pytest.approx(0.035)
    assert error_limits(0.15, 0.3)[0] == pytest.approx(0.050)
    assert np.degrees(error_limits(0.05, 0.1)[1]) == pytest.approx(5.0)
    assert np.degrees(error_limits(0.10, 0.2)[1]) == pytest.approx(5.0)
    assert np.degrees(error_limits(0.15, 0.3)[1]) == pytest.approx(8.0)


def test_limits_for_levels_scale_with_the_setting():
    from sobits_intball2_gnc.teleop.reference import limits_for_levels
    plant = ThrustAllocator()
    env = wrench_envelope_halfspaces(plant.A, plant.fj_max)
    slow = limits_for_levels(env, MASS, INERTIA, 0.05, 0.5, 2.0, 0.5)
    fast = limits_for_levels(env, MASS, INERTIA, 0.15, 0.7, 2.0, 0.5)
    assert slow.vmax == 0.05 and slow.wmax == pytest.approx(0.1)
    assert fast.vmax == 0.15 and fast.wmax == pytest.approx(0.3)
    assert fast.acc[0] == pytest.approx(slow.acc[0] * 0.7 / 0.5)
    assert fast.alpha[2] == pytest.approx(slow.alpha[2] * 0.7 / 0.5)
    assert fast.err_pos > slow.err_pos and fast.err_att > slow.err_att


def test_set_limits_changes_the_caps_from_the_next_step():
    ref = make()
    run(ref, [1, 0, 0, 0, 0, 0], 5.0)
    assert ref.vb[0] == pytest.approx(0.05)
    run(ref, np.zeros(6), 4.0)
    assert ref.at_rest
    ref.set_limits(limits(vmax=0.1, acc=(0.056, 0.031, 0.038)))
    sp = run(ref, [1, 0, 0, 0, 0, 0], 0.5)
    assert sp.v[0] == pytest.approx(0.056 * 0.5, rel=1e-6)   # the new, higher acceleration cap
    sp = run(ref, [1, 0, 0, 0, 0, 0], 5.0)
    assert sp.v[0] == pytest.approx(0.1)                      # the new speed cap
