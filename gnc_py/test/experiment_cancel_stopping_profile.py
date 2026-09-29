#!/usr/bin/env python3
"""Offline A/B of cancel-time braking (docs/archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md).

Rigid body (zero-g, isotropic inertia) driven by the real control stack --
``HoverController`` (``HoverLaw`` + ``PoseCorrector`` + ``TrajectoryController``)
and ``ThrustAllocator`` -- all built from ``config/gnc_params.yaml`` through
their own ``from_node`` paths. The vehicle first tracks a constant-velocity
setpoint, then the goal is canceled:

  A (current): setpoints stop; control falls back to holding the current
     pose after ``trajectory_controller.timeout``.
  B (JAXA stoppingProfile): setpoints follow ``StoppingProfile`` built from
     the measured state; afterwards the profile's end pose is held.

Not modeled: fan lag, TF noise/rate, estimator lag -> confirm in sim.

Not a pytest test (no test_ prefix):
    python3 test/experiment_cancel_stopping_profile.py
"""
import argparse
import math
import os

import numpy as np
import yaml

from sobits_intball2_gnc.common.utils.stopping_profile import StoppingProfile
from sobits_intball2_gnc.control.utils.hover_controller import HoverController
from sobits_intball2_gnc.control.utils.quat_math import (
    geodesic_angle,
    quat_exp,
    quat_mul,
    quat_rotate,
    quat_conj,
)
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

PARAMS = os.path.join(os.path.dirname(__file__), "..", "config", "gnc_params.yaml")
CONTROL_DT = 0.02
PHYSICS_SUBSTEPS = 10
PREROLL_S = 10.0
SAT_DUTY = 0.95
SETTLE_SPEED = 0.005
SETTLE_POS = 0.02
# JAXA ctl.yaml
TOLERANCE_POS_STOP = 0.30
TOLERANCE_ATT_STOP = 1.0
DURATION_GOAL = 3.0
WAIT_CANCEL = 10.0


class _Param:
    def __init__(self, value):
        self.value = value


class _YamlNode:
    def __init__(self, path):
        with open(path) as f:
            root = yaml.safe_load(f)["/**"]["ros__parameters"]
        self._params = {}
        self._flatten("", root)

    def _flatten(self, prefix, d):
        for k, v in d.items():
            name = prefix + k
            if isinstance(v, dict):
                self._flatten(name + ".", v)
            else:
                self._params[name] = v

    def has_parameter(self, name):
        return name in self._params

    def declare_parameter(self, name, default, descriptor=None):
        self._params.setdefault(name, default)

    def get_parameter(self, name):
        return _Param(self._params[name])


class _Imu:
    gyro = None
    acc = None


class _Fan:
    duties = []

    def set_duty_array(self, duties):
        self.duties = list(duties)


class _Tf:
    pose = None

    def get_pose(self):
        return self.pose


class _TrajectorySub:
    ready = False
    last_received_t = None
    p_des = v_des = a_des = q_des = omega_des = alpha_des = None

    def publish(self, t, p, v, a, q, omega, alpha):
        self.ready = True
        self.last_received_t = t
        self.p_des, self.v_des, self.a_des = list(p), list(v), list(a)
        self.q_des, self.omega_des, self.alpha_des = list(q), list(omega), list(alpha)


class Vehicle:
    def __init__(self, mass, inertia, p, v, q, w):
        self.m, self.I = mass, inertia
        self.p, self.v = np.array(p, float), np.array(v, float)
        self.q, self.w = np.array(q, float), np.array(w, float)
        self.a_body = np.zeros(3)

    def step(self, force_body, torque_body, dt):
        h = dt / PHYSICS_SUBSTEPS
        force_body, torque_body = np.asarray(force_body), np.asarray(torque_body)
        for _ in range(PHYSICS_SUBSTEPS):
            a_world = quat_rotate(self.q, force_body) / self.m
            self.p += self.v * h + 0.5 * a_world * h * h
            self.v += a_world * h
            self.w += torque_body / self.I * h
            self.q = quat_mul(self.q, quat_exp(self.w * h))
            self.q /= np.linalg.norm(self.q)
        self.a_body = force_body / self.m


def run(case, method, node_path=PARAMS, post_s=90.0):
    node = _YamlNode(node_path)
    allocator = ThrustAllocator.from_node(node)
    imu, fan, tf, traj = _Imu(), _Fan(), _Tf(), _TrajectorySub()
    hover = HoverController.from_node(node, imu, fan, allocator,
                                      tf_client=tf, trajectory_subscriber=traj)
    mass = float(node.get_parameter("trajectory_controller.mass").value)
    inertia = float(node.get_parameter("trajectory_controller.inertia").value)
    eta = float(node.get_parameter("guidance.wrench_envelope_safety_margin").value)
    max_axis_force = float(node.get_parameter("hover_control.max_force").value)

    d = np.asarray(case["dir"], float) / np.linalg.norm(case["dir"])
    v_cruise = case["speed"] * d
    w_cruise = np.asarray(case.get("w", [0.0, 0.0, 0.0]), float)
    q_start = np.array([0.0, 0.0, 0.0, 1.0])
    p_start = -v_cruise * PREROLL_S
    veh = Vehicle(mass, inertia, p_start, v_cruise, q_start, w_cruise)

    def cruise_ref(t):
        return (p_start + v_cruise * t, v_cruise, np.zeros(3),
                quat_mul(q_start, quat_exp(w_cruise * t)), w_cruise, np.zeros(3))

    def control_tick(t):
        imu.gyro, imu.acc = veh.w.tolist(), veh.a_body.tolist()
        tf.pose = (veh.p.tolist(), veh.q.tolist(), t)
        hover.step(t)
        force, torque = allocator.achieved_wrench(fan.duties)
        veh.step(force, torque, CONTROL_DT)

    t = 0.0
    while t < PREROLL_S:
        traj.publish(t, *cruise_ref(t))
        control_tick(t)
        t += CONTROL_DT

    t_cancel, p_cancel = t, veh.p.copy()
    v_at_cancel = np.linalg.norm(veh.v)
    profile = None
    if method == "B":
        profile = StoppingProfile(veh.p, veh.v, veh.q, veh.w, allocator,
                                  mass, inertia, eta, max_axis_force=max_axis_force)
        p_hold, _, _, q_hold, _, _ = profile.sample(profile.duration)
        braking, stay_since = True, None

    log = []
    while t < t_cancel + post_s:
        tp = t - t_cancel
        q_ref = None
        if method == "B" and braking:
            ref = profile.sample(tp)
            traj.publish(t, *ref)
            q_ref = ref[3]
            near = (np.linalg.norm(veh.p - profile.r1) < TOLERANCE_POS_STOP
                    and geodesic_angle(veh.q, profile.q1) < TOLERANCE_ATT_STOP)
            stay_since = (stay_since if stay_since is not None else t) if near else None
            if ((stay_since is not None and t - stay_since >= DURATION_GOAL)
                    or tp >= profile.duration + WAIT_CANCEL):
                hover.set_checkpoints([(p_hold, q_hold)], is_align=False)
                braking = False
        control_tick(t)
        log.append((tp, veh.p.copy(), veh.v.copy(), veh.q.copy(),
                    list(fan.duties), q_ref))
        t += CONTROL_DT

    return _metrics(log, p_cancel, d, v_at_cancel, profile)


def _metrics(log, p_cancel, d, v0, profile):
    along = np.array([np.dot(p - p_cancel, d) for _, p, *_ in log])
    speeds = np.array([np.linalg.norm(v) for _, _, v, *_ in log])
    positions = np.array([p for _, p, *_ in log])
    p_final = positions[-1]
    settled = (speeds < SETTLE_SPEED) & (np.linalg.norm(positions - p_final, axis=1) < SETTLE_POS)
    unsettled = np.flatnonzero(~settled)
    settle_idx = 0 if unsettled.size == 0 else unsettled[-1] + 1
    settle_t = log[settle_idx][0] if settle_idx < len(log) else None
    stop_idx = next((i for i, s in enumerate(speeds) if s < SETTLE_SPEED), len(log) - 1)
    window = log[:stop_idx + 1]
    sat = np.mean([max(dd) >= SAT_DUTY for *_, dd, _ in window]) if window else 0.0
    q_end = log[-1][3]
    att_dev = max(geodesic_angle(q, q_end) for _, _, _, q, *_ in log)
    return {
        "v0": v0,
        "max_excursion": along.max(),
        "final_offset": np.dot(p_final - p_cancel, d),
        "back_travel": along.max() - np.dot(p_final - p_cancel, d),
        "first_stop_t": log[stop_idx][0],
        "settle_t": settle_t,
        "sat_frac": sat,
        "att_dev_deg": math.degrees(att_dev),
        "planned_stop": (np.dot(profile.sample(profile.duration)[0] - p_cancel, d)
                         if profile is not None else float("nan")),
    }


CASES = [
    {"name": "x 0.2", "dir": [1, 0, 0], "speed": 0.2},
    {"name": "y 0.2", "dir": [0, 1, 0], "speed": 0.2},
    {"name": "diag 0.2", "dir": [1, 1, 1], "speed": 0.2},
    {"name": "x 0.5", "dir": [1, 0, 0], "speed": 0.5},
    {"name": "y 0.5", "dir": [0, 1, 0], "speed": 0.5},
    {"name": "diag 0.5", "dir": [1, 1, 1], "speed": 0.5},
    {"name": "x 0.3 +wz0.1", "dir": [1, 0, 0], "speed": 0.3, "w": [0, 0, 0.1]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="subset of case names")
    ap.add_argument("--post", type=float, default=90.0, help="seconds after cancel")
    args = ap.parse_args()
    cases = [c for c in CASES if not args.cases or c["name"] in args.cases]
    hdr = ("case", "m", "v0", "plan", "maxExc", "final", "back", "stop_t", "settle_t",
           "sat%", "attDev")
    print("%-14s %-2s %5s %6s %7s %7s %6s %7s %8s %5s %6s" % hdr)
    for c in cases:
        for m in ("A", "B"):
            r = run(c, m, post_s=args.post)
            st = "-" if r["settle_t"] is None else "%.1f" % r["settle_t"]
            print("%-14s %-2s %5.2f %6.2f %7.2f %7.2f %6.2f %7.1f %8s %5.0f %6.1f" % (
                c["name"], m, r["v0"], r["planned_stop"], r["max_excursion"],
                r["final_offset"], r["back_travel"], r["first_stop_t"], st,
                100 * r["sat_frac"], r["att_dev_deg"]), flush=True)


if __name__ == "__main__":
    main()
