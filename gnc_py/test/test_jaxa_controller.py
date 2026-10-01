"""Ported JAXA controller (sobits_intball2_gnc_cpp) built from config/jaxa_control.yaml.

The pybind results are compared with an independent numpy transcription of the
JAXA laws, the shared vehicle constants with gnc_params.yaml, and the JAXA
allocation with our fan geometry (the offline plant).
"""
import math
import os

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from sobits_intball2_gnc.control.utils.jaxa_control_params import (
    inertia_rows,
    load_jaxa_control,
    make_attitude_controller,
    make_position_controller,
    make_thrust_allocator,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")
CFG = load_jaxa_control(os.path.join(CONFIG, "jaxa_control.yaml"))
OURS = yaml.safe_load(open(os.path.join(CONFIG, "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
FANS = ["fan%02d" % i for i in range(1, 9)]
KEYS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
RNG = np.random.default_rng(0)


def rand_quat():
    return Rotation.random(random_state=int(RNG.integers(1 << 31))).as_quat()


def ref_force(cfg, r, v, q, r_ref, v_ref, a_ref):
    p = cfg["pos_ctl"]
    a = np.asarray(a_ref) - p["kp"] * (np.asarray(r) - r_ref) - p["kd"] * (np.asarray(v) - v_ref)
    return Rotation.from_quat(q).inv().apply(a * cfg["ctl_body"]["mass"])


def ref_torque(cfg, q, w, q_ref, w_ref):
    kp, kd = cfg["att_ctl"]["kp"], cfg["att_ctl"]["kd"]
    inertia = np.asarray(inertia_rows(cfg))
    qe = Rotation.from_quat(q_ref).inv() * Rotation.from_quat(q)
    x, y, z, qw = qe.as_quat()
    sign = 1.0 if qw >= 0 else -1.0
    wc = -2.0 * kp / kd * sign * np.array([x, y, z]) + qe.inv().apply(w_ref)
    w = np.asarray(w)
    return kd * inertia @ (wc - w) + np.cross(w, inertia @ w)


def ref_duty(cfg, force, torque):
    fan = cfg["fan"]
    wp = np.array([[fan["Wp"][f][k] for k in KEYS] for f in FANS])
    wm = np.array([[fan["Wm"][f][k] for k in KEYS] for f in FANS])
    y = np.concatenate((force, torque))
    f = wp @ np.maximum(y, 0.0) + wm @ np.maximum(-y, 0.0)
    f = f - f.min()
    pwm = np.array([fan["kj"][n] for n in FANS]) * np.sqrt(np.abs(f + [fan["fj0"][n] for n in FANS]))
    over = pwm > fan["PWMmax"]
    if over.sum() >= fan["n_saturation"]:
        out = np.zeros_like(pwm)
        out[np.argsort(-pwm, kind="stable")[:fan["n_saturation"]]] = fan["PWMmax"]
        return out, int(over.sum())
    return np.minimum(pwm, fan["PWMmax"]), int(over.sum())


def test_position_controller_matches_independent_formula():
    c = make_position_controller(CFG)
    for _ in range(100):
        r, v, r_ref, v_ref, a_ref = (RNG.normal(0, 0.3, 3) for _ in range(5))
        q = rand_quat()
        got = c.force_command(0.0, r, v, q, r_ref, v_ref, a_ref)
        np.testing.assert_allclose(got, ref_force(CFG, r, v, q, r_ref, v_ref, a_ref), atol=1e-12)


def test_attitude_controller_matches_independent_formula():
    c = make_attitude_controller(CFG)
    for _ in range(100):
        q, q_ref = rand_quat(), rand_quat()
        w, w_ref = RNG.normal(0, 0.3, 3), RNG.normal(0, 0.3, 3)
        got = c.torque_command(q, w, q_ref, w_ref)
        np.testing.assert_allclose(got, ref_torque(CFG, q, w, q_ref, w_ref), atol=1e-12)


def test_thrust_allocator_matches_independent_formula():
    a = make_thrust_allocator(CFG)
    for scale in (0.02, 0.1, 0.4):
        for _ in range(100):
            force = RNG.normal(0, scale, 3)
            torque = RNG.normal(0, scale / 20, 3)
            want, over = ref_duty(CFG, force, torque)
            np.testing.assert_allclose(a.duty(force, torque), want, atol=1e-12)
            assert a.last_saturated_count == over


def test_vehicle_constants_match_gnc_params():
    tc, ta = OURS["trajectory_controller"], OURS["thrust_allocator"]
    assert CFG["ctl_body"]["mass"] == tc["mass"]
    np.testing.assert_allclose(inertia_rows(CFG), np.eye(3) * tc["inertia"])
    assert all(CFG["fan"]["kj"][f] == ta["kj"] for f in FANS)
    assert CFG["fan"]["Fmax"] == ta["fj_max"]


def test_jaxa_allocation_reproduces_wrench_through_our_fan_geometry():
    jaxa, ours = make_thrust_allocator(CFG), ThrustAllocator()
    for i in range(6):
        for sign in (1.0, -1.0):
            y = np.zeros(6)
            y[i] = sign * (0.05 if i < 3 else 0.002)
            achieved = ours.A @ np.asarray(jaxa.allocate(y[:3], y[3:]))
            assert np.linalg.norm(achieved - y) <= 0.01 * abs(y[i])


def test_position_step_response_matches_second_order_design():
    """Point mass, 200 Hz, no saturation: overshoot exp(-zeta pi / sqrt(1 - zeta^2))."""
    c = make_position_controller(CFG)
    p = CFG["pos_ctl"]
    zeta = p["kd"] / (2.0 * math.sqrt(p["kp"]))
    expected = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta ** 2))
    dt, m = 0.005, CFG["ctl_body"]["mass"]
    x, v, peak, q = np.zeros(3), np.zeros(3), 0.0, [0.0, 0.0, 0.0, 1.0]
    target = np.array([0.1, 0.0, 0.0])
    for k in range(4000):
        f = np.asarray(c.force_command(k * dt, x, v, q, target, np.zeros(3), np.zeros(3)))
        v = v + f / m * dt
        x = x + v * dt
        peak = max(peak, x[0])
    assert abs((peak - 0.1) / 0.1 - expected) < 0.005
    assert abs(x[0] - 0.1) < 1e-4


def test_invalid_parameters_raise():
    bad = yaml.safe_load(yaml.safe_dump(CFG))
    bad["att_ctl"]["kd"] = 0.0
    with pytest.raises(ValueError):
        make_attitude_controller(bad)
    bad = yaml.safe_load(yaml.safe_dump(CFG))
    bad["fan"]["n_saturation"] = 9
    with pytest.raises(ValueError):
        make_thrust_allocator(bad)
