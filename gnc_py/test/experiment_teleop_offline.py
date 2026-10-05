#!/usr/bin/env python3
"""Offline check: can a keyboard-driven moving reference be tracked by the JAXA controller?

Drives ``teleop_reference_py.TeleopReference`` (docs/archive/achieved/2026-10-01_teleop_keyboard_design.md):
held keys give a body-frame velocity command, slew-limited and shaped to the wrench envelope,
integrated into a reference pose published at the guidance rate (50 Hz); the reference stalls when
the tracking error exceeds a limit. The plant is sim-matched, as ``experiment_jaxa_baseline_offline.py
--controller jaxa``: ported JAXA position/attitude controllers at the TF rate (~42 Hz), JAXA
thrust allocation, fan layout to a full wrench, rigid-body rotation.

No obstacles, depth or planner: the reference is free flight in an empty world.
"""
import argparse
import csv
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from sobits_intball2_gnc.control.utils.jaxa_control_params import (
    inertia_rows, load_jaxa_control, make_attitude_controller, make_position_controller,
    make_thrust_allocator)
from sobits_intball2_gnc.control.utils.jaxa_tracking_controller import ema_alpha
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_exp, quat_mul, quat_rotate
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.control.utils.trajectory_controller import DEFAULT_TRAJECTORY
from teleop_reference_py import TeleopLimits, TeleopReference

HERE = os.path.dirname(__file__)
CONFIG = os.path.join(HERE, "..", "config")
PARAMS = yaml.safe_load(open(os.path.join(CONFIG, "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
TC = PARAMS["trajectory_controller"]
JAXA_CFG = load_jaxa_control(os.path.join(CONFIG, "jaxa_control.yaml"))
INERTIA = np.array(inertia_rows(JAXA_CFG), dtype=float)
MASS = float(TC["mass"])
VEL_ALPHA = TC.get("vel_filter_alpha", DEFAULT_TRAJECTORY["vel_filter_alpha"])
FORCE_AXIS_MAX = np.array(TC["max_force"], dtype=float)  # per-axis theoretical max [N]

ENVELOPE = np.loadtxt(os.path.join(HERE, "..", "..", "gnc_cpp", "config", "wrench_envelope.csv"), skiprows=1)
ENV_A, ENV_B = ENVELOPE[:, :6], ENVELOPE[:, 6]

GUIDANCE_DT = 0.02  # guidance.rate 50 Hz
PHYS_DT = 0.005
CONTROL_RATE = 42.0
TRACE_PERIOD = 0.1  # 10 Hz: trajectory/transient CSV (AGENTS.md)
FAN_SIGMA = 0.012
NAV_POS_MEAN = np.array([-0.0021, -0.0018, -0.000067304])
NAV_POS_SIGMA = np.array([0.00084609, 0.00085207, 0.00017358])
NAV_DELAY = 0.1

KEYS = ("vx", "vy", "vz", "wx", "wy", "wz")


def axis_torque_max():
    """Per-axis max torque from the wrench envelope (LP over the 24 faces)."""
    from scipy.optimize import linprog
    out = []
    for i in (3, 4, 5):
        c = np.zeros(6)
        c[i] = -1.0
        out.append(-linprog(c, A_ub=ENV_A, b_ub=ENV_B, bounds=[(None, None)] * 6).fun)
    return np.array(out)


class JaxaController:
    """JAXA position + attitude control on one TF update (as in the baseline experiment)."""

    def __init__(self):
        self.pos = make_position_controller(JAXA_CFG)
        self.att = make_attitude_controller(JAXA_CFG)
        self.last, self.vel = None, np.zeros(3)
        self.out = (np.zeros(3).tolist(), np.zeros(3).tolist())

    def compute(self, stamp, p, q, gyro, p_des, v_des, a_des, q_des, w_des):
        p = np.asarray(p, dtype=float)
        if self.last is not None and stamp - self.last[0] <= 1e-9:
            return self.out  # repeated TF stamp: production skips the update and keeps the last wrench
        if self.last is not None:
            dt = stamp - self.last[0]
            a = ema_alpha(VEL_ALPHA, dt)
            self.vel = a * (p - self.last[1]) / dt + (1.0 - a) * self.vel
        self.last = (stamp, p.copy())
        force = self.pos.force_command(stamp, p.tolist(), self.vel.tolist(), list(q),
                                       list(p_des), list(v_des), list(a_des))
        torque = self.att.torque_command(list(q), list(gyro), list(q_des), list(w_des))
        self.out = (force, torque)
        return force, torque


def schedule(*segments):
    """``(t_start, t_end, {key: value})`` segments -> f(t) -> 6-vector of key values."""
    def f(t):
        out = np.zeros(6)
        for t0, t1, keys in segments:
            if t0 <= t < t1:
                for k, v in keys.items():
                    out[KEYS.index(k)] = v
        return out
    return f


def pulse_train(keys_list, on, off, n):
    segs, t = [], 0.0
    for i in range(n):
        segs.append((t, t + on, keys_list[i % len(keys_list)]))
        t += on + off
    return segs


SCENARIOS = {
    "hold": ("no keys (hold only; noise/saturation reference)", 30.0, schedule()),
    "fwd": ("forward 6 s, release", 30.0, schedule((0, 6, {"vx": 1}))),
    "yaw": ("yaw 5 s, release", 30.0, schedule((0, 5, {"wz": 1}))),
    "pitch": ("pitch 5 s, release", 30.0, schedule((0, 5, {"wy": 1}))),
    "roll": ("roll 5 s, release", 30.0, schedule((0, 5, {"wx": 1}))),
    "arc": ("forward + yaw 10 s, release", 40.0, schedule((0, 10, {"vx": 1, "wz": 1}))),
    "arc_pitch": ("forward + pitch 10 s, release", 40.0, schedule((0, 10, {"vx": 1, "wy": 1}))),
    "repeat": ("1 s on / 1.5 s off x6, alternating fwd+yaw / back+yaw(-)", 40.0, schedule(
        *pulse_train([{"vx": 1, "wz": 1}, {"vx": -1, "wz": -1}, {"vy": 1}], 1.0, 1.5, 6))),
}


def run(name, args, csv_dir=None):
    desc, duration, key_fn = SCENARIOS[name]
    tau_max = axis_torque_max()
    inertia_diag = np.diag(INERTIA)
    acc = args.acc_frac * FORCE_AXIS_MAX / MASS
    alpha = args.alpha_frac * tau_max / inertia_diag
    ref = TeleopReference(
        TeleopLimits(vmax=args.vmax, wmax=args.wmax, acc=tuple(acc), alpha=tuple(alpha),
                     err_pos=args.err_pos, err_att=np.radians(args.err_att_deg)),
        MASS, INERTIA, envelope=None if args.no_shape else (ENV_A, ENV_B),
        centripetal=not args.no_centripetal)
    ref.reset(np.zeros(3), [0.0, 0.0, 0.0, 1.0])
    ctrl, alloc, plant = JaxaController(), make_thrust_allocator(JAXA_CFG), ThrustAllocator()
    inertia_inv = np.linalg.inv(INERTIA)
    rng = np.random.RandomState(args.seed)
    p, v, w = np.zeros(3), np.zeros(3), np.zeros(3)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    history = [(0.0, p.copy())]
    duties = np.zeros(plant.fan_count)
    t = next_guidance = next_control = next_trace = 0.0
    control_period = 1.0 / CONTROL_RATE
    p_des, v_des, a_des, q_des, w_des = np.zeros(3), np.zeros(3), np.zeros(3), q.copy(), np.zeros(3)
    rows = []
    n_g = n_stall = n_scaled = n_c = n_sat = 0
    max_duty = 0.0
    max_ep = 0.0
    max_lat = np.zeros(3)
    ep_sum = 0.0
    att_err, att_max = [], 0.0
    max_dev_dist = 0.0
    t_release = max((t1 for _, t1, _ in _segments(name)), default=0.0)
    t_settle = None
    while t < duration:
        if args.noise:
            history.append((t, p.copy()))
            while len(history) > 1 and history[1][0] <= t - NAV_DELAY + 1e-9:
                history.pop(0)
            stamp, p_meas = history[0][0], history[0][1] + NAV_POS_MEAN + rng.normal(0.0, NAV_POS_SIGMA)
        else:
            stamp, p_meas = t, p
        if t + 1e-9 >= next_guidance:
            next_guidance += GUIDANCE_DT
            p_des, v_des, a_des, q_des, w_des = ref.step(GUIDANCE_DT, key_fn(t), p_meas, q)
            n_g += 1
            n_stall += ref.stalled
            n_scaled += ref.scaled
        if t + 1e-9 >= next_control:
            next_control += control_period
            force, torque = ctrl.compute(stamp, p_meas, q, w, p_des, v_des, a_des, q_des, w_des)
            duties = np.asarray(alloc.duty(list(force), list(torque)))
            n_c += 1
            n_sat += alloc.last_saturated_count > 0
            max_duty = max(max_duty, float(duties.max()))
            att_err.append(geodesic_angle(q, q_des))
            att_max = max(att_max, att_err[-1])
        e_world = p_des - p
        e_body = quat_rotate(np.array([-q[0], -q[1], -q[2], q[3]]), e_world)
        ep = float(np.linalg.norm(e_world))
        max_ep, ep_sum = max(max_ep, ep), ep_sum + ep
        max_lat = np.maximum(max_lat, np.abs(e_body))
        if t >= t_release and t_settle is None and ep < 0.005 and np.linalg.norm(v) < 0.002 \
                and np.linalg.norm(w) < np.radians(0.2):
            t_settle = t - t_release
        if csv_dir is not None and t + 1e-9 >= next_trace:
            next_trace += TRACE_PERIOD
            rows.append(dict(t=t, px=p[0], py=p[1], pz=p[2], rx=p_des[0], ry=p_des[1], rz=p_des[2],
                             speed=float(np.linalg.norm(v)), ref_speed=float(np.linalg.norm(v_des)),
                             pos_err=ep, att_err_deg=float(np.degrees(geodesic_angle(q, q_des))),
                             stalled=int(ref.stalled), scaled=int(ref.scaled),
                             saturated=alloc.last_saturated_count))
        thrust = (duties / plant.kj) ** 2
        if args.noise:
            thrust = np.maximum(thrust + rng.normal(0.0, FAN_SIGMA, len(thrust)), 0.0)
        wrench = plant.A @ thrust
        v = v + quat_rotate(q, wrench[:3]) / MASS * PHYS_DT
        w = w + inertia_inv @ (wrench[3:] - np.cross(w, INERTIA @ w)) * PHYS_DT
        q = quat_mul(q, quat_exp(w * PHYS_DT))
        q = q / np.linalg.norm(q)
        p, t = p + v * PHYS_DT, t + PHYS_DT
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)
        with open(os.path.join(csv_dir, "teleop_%s.csv" % name), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    steps = max(n_c, 1)
    return dict(
        scenario=name, max_pos_err_mm=1e3 * max_ep, mean_pos_err_mm=1e3 * ep_sum / max(duration / PHYS_DT, 1),
        max_lat_body_mm="/".join("%.0f" % (1e3 * x) for x in max_lat),
        att_err_max_deg=np.degrees(att_max), att_err_mean_deg=np.degrees(np.mean(att_err)),
        final_hold_err_mm=1e3 * float(np.linalg.norm(p_des - p)),
        settle_s=t_settle if t_settle is not None else float("nan"),
        stalled_frac=n_stall / max(n_g, 1), scaled_frac=n_scaled / max(n_g, 1),
        sat_frac=n_sat / steps, max_duty=max_duty, ref_dist_m=float(np.linalg.norm(ref.p)),
        act_dist_m=float(np.linalg.norm(p)))


def _segments(name):
    f = SCENARIOS[name][2]
    ts = np.arange(0.0, SCENARIOS[name][1], 0.01)
    on = [bool(np.any(f(t))) for t in ts]
    segs, start = [], None
    for t, o in zip(ts, on):
        if o and start is None:
            start = t
        if not o and start is not None:
            segs.append((start, t, None))
            start = None
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenarios", nargs="*", default=["fwd", "yaw", "arc", "repeat"], help="subset of: %s" % ", ".join(SCENARIOS))
    ap.add_argument("--vmax", type=float, default=0.05, help="translational speed cap [m/s]")
    ap.add_argument("--wmax", type=float, default=0.1, help="angular speed cap [rad/s]")
    ap.add_argument("--acc-frac", type=float, default=0.5, help="accel limit as a fraction of per-axis max force / mass")
    ap.add_argument("--alpha-frac", type=float, default=0.5, help="angular accel limit as a fraction of per-axis max torque / inertia")
    ap.add_argument("--err-pos", type=float, default=0.05, help="stall the reference above this position error [m]")
    ap.add_argument("--err-att-deg", type=float, default=5.0, help="stall the reference above this attitude error [deg]")
    ap.add_argument("--no-centripetal", action="store_true", help="omit w x v from the feed-forward acceleration")
    ap.add_argument("--no-shape", action="store_true", help="do not scale rate changes to the wrench envelope")
    ap.add_argument("--noise", action="store_true", help="nav delay/noise and fan noise as the baseline experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv-dir", default=None)
    args = ap.parse_args()
    tm = axis_torque_max()
    print("# acc=%s m/s^2 alpha=%s rad/s^2 (torque max %s Nm, I diag %s) vmax=%.3f wmax=%.3f err=%.3f m/%.1f deg "
          "centripetal=%s shape=%s noise=%s" % (
              np.round(args.acc_frac * FORCE_AXIS_MAX / MASS, 4), np.round(args.alpha_frac * tm / np.diag(INERTIA), 4),
              np.round(tm, 4), np.round(np.diag(INERTIA), 4), args.vmax, args.wmax, args.err_pos, args.err_att_deg,
              not args.no_centripetal, not args.no_shape, args.noise))
    for name in args.scenarios:
        r = run(name, args, args.csv_dir)
        print("%-9s %s" % (name, SCENARIOS[name][0]))
        print("  " + "  ".join("%s=%s" % (k, ("%.3f" % x) if isinstance(x, float) else x)
                               for k, x in r.items() if k != "scenario"))


if __name__ == "__main__":
    main()
