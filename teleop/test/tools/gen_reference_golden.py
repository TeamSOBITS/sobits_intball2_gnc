#!/usr/bin/env python3
"""Write the Python TeleopReference's output for fixed scenarios (golden data for the C++ port).

Run from the repo root (needs gnc_py/test/teleop_reference_py.py, scipy):
    python3 teleop/test/tools/gen_reference_golden.py
Writes teleop/test/data/{reference_golden.csv, limits_golden.csv, envelope_golden.csv}.
"""
import csv
import os

import numpy as np
import yaml

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.constraints.actuation_envelope import wrench_envelope_halfspaces
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "gnc_py", "test"))
from teleop_reference_py import TeleopReference, limits_for_levels  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "..", "..", "gnc_py", "config", "gnc_params.yaml")
DATA = os.path.join(ROOT, "data")
DT = 0.02
Q0 = [0.0, 0.0, 0.0, 1.0]


def load_params():
    with open(PARAMS) as f:
        raw = yaml.safe_load(f)
    params = raw["/**"]["ros__parameters"] if "/**" in raw else next(iter(raw.values()))["ros__parameters"]
    return params


def main():
    params = load_params()
    ta = params["thrust_allocator"]
    alloc = ThrustAllocator(kj=ta["kj"], fj_max=ta["fj_max"], cg=ta["cg"], fan_positions=ta["fan_positions"],
                            fan_vectors=ta["fan_vectors"])
    env = wrench_envelope_halfspaces(alloc.A, alloc.fj_max, safety_margin=1.0)
    mass = params["trajectory_controller"]["mass"]
    inertia = params["trajectory_controller"]["inertia"]
    tp = params["teleop"]
    speeds, accels = tp["speed_levels"], tp["acc_frac_levels"]

    with open(os.path.join(DATA, "envelope_golden.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for row, g in zip(env[0], env[1]):
            w.writerow(list(row) + [g])

    with open(os.path.join(DATA, "limits_golden.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for si, v in enumerate(speeds):
            for ai, a in enumerate(accels):
                lim = limits_for_levels(env, mass, inertia, v, a, tp["wmax_per_vmax"], tp["alpha_per_acc"],
                                        tp["resume_ratio"])
                w.writerow([si, ai, lim.vmax, lim.wmax, *lim.acc, *lim.alpha, lim.err_pos, lim.err_att, lim.resume])

    lim = limits_for_levels(env, mass, inertia, speeds[1], accels[2], tp["wmax_per_vmax"], tp["alpha_per_acc"],
                            tp["resume_ratio"])
    p0 = np.array([1.0, 2.0, 3.0])
    q0 = np.array([0.1, -0.2, 0.3, 0.9])
    q0 = q0 / np.linalg.norm(q0)
    z = [0.0] * 6
    scenarios = [([1, 0, 0, 0, 0, 0], 3.0), (z, 2.0),                                # forward, stop
                 ([0, 1, 0, 0, 0, 0.5], 3.0), (z, 2.0),                              # arc
                 ([1, 1, 1, 1, 1, 1], 3.0), (z, 2.0),                                # everything: envelope shaping
                 ([-1, 0, 0, 0, 0, 0], 2.0), ([0, 0, 0, 0, 1, 0], 2.0), (z, 2.0)]   # back, pitch
    ref2 = TeleopReference(lim, mass, inertia, envelope=env)
    ref2.reset(p0, q0)
    out = []
    for key, seconds in scenarios:
        for _ in range(int(round(seconds / DT))):
            pm, qm = ref2.p, ref2.q
            sp = ref2.step(DT, key, pm, qm)
            out.append((key, pm, qm, sp, snapshot(ref2)))
    stuck_p, stuck_q = ref2.p, ref2.q
    for i in range(300):
        pm = stuck_p - np.array([0.03, 0.0, 0.0]) if i < 150 else ref2.p
        qm = stuck_q if i < 150 else ref2.q
        sp = ref2.step(DT, [1, 0, 0, 0, 0, 0], pm, qm)
        out.append(([1, 0, 0, 0, 0, 0], pm, qm, sp, snapshot(ref2)))
    with open(os.path.join(DATA, "reference_golden.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["I", *p0, *q0, *lim_row(lim), mass, inertia, DT])
        for key, pm, qm, sp, r in out:
            w.writerow(["S", *key, *pm, *qm, *sp.p, *sp.v, *sp.a, *sp.q, *sp.w, *r["vb"], *r["wb"],
                        int(r["stalled"]), int(r["scaled"]), r["pos_err"], r["att_err"]])
    print("rows:", len(out), "stalled rows:", sum(1 for *_, r in out if r["stalled"]),
          "scaled rows:", sum(1 for *_, r in out if r["scaled"]))


def snapshot(ref):
    return {"vb": ref.vb.copy(), "wb": ref.wb.copy(), "stalled": ref.stalled, "scaled": ref.scaled,
            "pos_err": ref.pos_err, "att_err": ref.att_err}


def lim_row(lim):
    return [lim.vmax, lim.wmax, *lim.acc, *lim.alpha, lim.err_pos, lim.err_att, lim.resume]


if __name__ == "__main__":
    main()
