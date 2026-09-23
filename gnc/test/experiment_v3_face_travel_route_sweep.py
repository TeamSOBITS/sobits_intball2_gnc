"""v3 face travel across route shapes, with a body-frame wrench check.

minco_solver.cpp checks MASS*acc in the reference frame against the
body-frame wrench envelope, so the body-frame ratio here is computed
independently: F_ENV @ [m R^T a, I w_dot] / G_ENV (1.0 = physical limit,
the solver itself caps its own reference-frame view at MARGIN).
"""
import sys

import numpy as np
import yaml

from experiment_v3_face_travel_feasibility import (
    FWD, MARGIN, SPEED_MIN, angle_deg, build_global, compute_q_des, run_local,
)
from experiment_v3_face_travel_lead_diagnosis import fwd_of
from experiment_v3_face_travel_replan_period_sweep import POSITION_ON_GLOBAL_DEG, global_headings
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_exp, quat_mul, quat_rotate
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoInfeasibleError

MASS = 3.216
INERTIA = 0.0136
ENV = np.loadtxt("/root/colcon_ws/src/sobits_intball2_gnc/minco_native_py/config/wrench_envelope.csv",
                 skiprows=1)
F_ENV, G_ENV = ENV[:, :6], ENV[:, 6]
LOC = yaml.safe_load(open(
    "/root/colcon_ws/src/sobits_intball2_gnc/gnc/maps/iss_location.yaml"))["location_pose"]
Q_ID = np.array([0.0, 0.0, 0.0, 1.0])
Q_YAW120 = np.array([0.0, 0.0, np.sin(np.radians(60)), np.cos(np.radians(60))])


def loc(*names):
    return [np.array([LOC[n]["translation"][k] for k in "xyz"]) for n in names]


def loc_q(name):
    r = LOC[name]["rotation"]
    return np.array([r["x"], r["y"], r["z"], r["w"]])


# (name, waypoints, raw start attitude, pre_align, global via_half_width)
ROUTES = [
    ("S1 straight 5m", [[0, 0, 0], [5, 0, 0]], Q_ID, True, 0.0),
    ("S2 straight 1.5m", [[0, 0, 0], [1.5, 0, 0]], Q_ID, True, 0.0),
    ("S3 turn 90", [[0, 0, 0], [3, 0, 0], [3, 3, 0]], Q_ID, True, 0.0),
    ("S4 turn 135", [[0, 0, 0], [3, 0, 0], [3 - 2.12, 2.12, 0]], Q_ID, True, 0.0),
    ("S5 hairpin 170", [[0, 0, 0], [3, 0, 0], [3 - 2.99, 0.26, 0]], Q_ID, True, 0.0),
    ("S6 U-turn 2x90 (0.5m)", [[0, 0, 0], [3, 0, 0], [3, 0.5, 0], [0, 0.5, 0]], Q_ID, True, 0.0),
    ("S7 zigzag 5x120 (1m)", [[0, 0, 0], [1, 0, 0], [1.5, 0.87, 0], [2.5, 0.87, 0], [3, 0, 0],
                              [4, 0, 0], [4.5, 0.87, 0]], Q_ID, True, 0.0),
    ("S8 square loop 5x90 (cum 450)", [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0.3, 0],
                                       [1.7, 0.3, 0], [1.7, 1.7, 0]], Q_ID, True, 0.0),
    ("S9 3D climb corner", [[0, 0, 0], [3, 0, 0], [3, 0, 2], [5, 0, 2]], Q_ID, True, 0.0),
    ("S10 3D skew corners", [[0, 0, 0], [2, 1, 0.5], [3, 3, 0], [2, 4, 1.5]], Q_ID, True, 0.0),
    ("S11 = S3 curved (vhw 0.25)", [[0, 0, 0], [3, 0, 0], [3, 3, 0]], Q_ID, True, 0.25),
    ("S12 = S7 curved (vhw 0.25)", [[0, 0, 0], [1, 0, 0], [1.5, 0.87, 0], [2.5, 0.87, 0], [3, 0, 0],
                                    [4, 0, 0], [4.5, 0.87, 0]], Q_ID, True, 0.25),
    ("S13 = S1 no pre_align, start yaw 120", [[0, 0, 0], [5, 0, 0]], Q_YAW120, False, 0.0),
    ("S14 = S3 no pre_align, start yaw 120", [[0, 0, 0], [3, 0, 0], [3, 3, 0]], Q_YAW120, False, 0.0),
    ("M1 above_dock->nav_entry->insp2->cap3",
     loc("above_dock", "nav_entry", "inspection_entry_2", "capture_point_3"), loc_q("above_dock"), True, 0.0),
    ("M2 insp1->insp3->insp2->cap2",
     loc("inspection_entry_1", "inspection_entry_3", "inspection_entry_2", "capture_point_2"),
     loc_q("inspection_entry_1"), True, 0.0),
    ("M3 nav_entry->insp1->nav_entry (U)",
     loc("nav_entry", "inspection_entry_1", "nav_entry"), loc_q("nav_entry"), True, 0.0),
    ("M4 near_dock->above_dock_2->nav_entry->cap1",
     loc("near_dock", "above_dock_2", "nav_entry", "capture_point_1"), loc_q("near_dock"), True, 0.0),
]

BOTH = {"carry_rot_accel": True, "rot_rate_tail": True}
LOCAL_FT = {"carry_rot_accel": True, "local_face_travel": True}
LOCAL_FT_GVIA = {**LOCAL_FT, "local_via_from_global": True}
VARIANT_SETS = {
    "boundary": [
        ("prod ft=off", False, {}),
        ("ft rest", True, {}),
        ("ft both", True, BOTH),
        ("ft both 0.5s", True, {**BOTH, "period": 0.5}),
        ("ft both fail/3", True, {**BOTH, "fail_every": 3}),
    ],
    "local_ft": [
        ("prod ft=off", False, {}),
        ("ft both", True, BOTH),
        ("localft vhw0", True, LOCAL_FT),
        ("localft vhw0.3", True, {**LOCAL_FT, "local_vhw": 0.3}),
        ("localft gvia", True, LOCAL_FT_GVIA),
        ("localft gvia vhw0.3", True, {**LOCAL_FT_GVIA, "local_vhw": 0.3}),
        ("localft gvia 0.5s", True, {**LOCAL_FT_GVIA, "period": 0.5}),
        ("localft gvia f/3", True, {**LOCAL_FT_GVIA, "fail_every": 3}),
    ],
    "self_path": [
        ("prod ft=off", False, {}),
        ("self path v0.2", True, {"carry_rot_accel": True, "local_self_path": True, "max_vel": 0.2}),
    ],
    "body": [
        ("prod ft=off body", False, {"body_frame": True}),
        ("self v0.2 body", True, {"carry_rot_accel": True, "local_self_path": True, "max_vel": 0.2,
                                  "body_frame": True}),
    ],
    "gvia": [
        ("localft gvia", True, LOCAL_FT_GVIA),
        ("localft gvia vhw0.3", True, {**LOCAL_FT_GVIA, "local_vhw": 0.3}),
    ],
}


def wrench_ratios(q0, rv, a, wd):
    q = quat_mul(q0, quat_exp(rv))
    torque = INERTIA * wd
    body = F_ENV @ np.concatenate([MASS * quat_rotate(quat_conj(q), a), torque]) / G_ENV
    ref = F_ENV @ np.concatenate([MASS * a, torque]) / G_ENV
    return body.max(), ref.max()


def evaluate(r, q0, gp, gf):
    fv, lag, body, ref = [], [], [], []
    for (_t, p, v, rv, _w, wd), a in zip(r["recs"], r["accs"]):
        b, rf = wrench_ratios(q0, rv, a, wd)
        body.append(b)
        ref.append(rf)
        if np.linalg.norm(v) <= SPEED_MIN:
            continue
        f = fwd_of(q0, rv)
        k = int(np.argmin(np.linalg.norm(gp - p, axis=1)))
        fv.append(angle_deg(f, v))
        if angle_deg(v, gf[k]) < POSITION_ON_GLOBAL_DEG:
            lag.append(angle_deg(f, gf[k]))
    body = np.array(body)
    lag_p95 = np.percentile(lag, 95) if lag else float("nan")
    vmax = max(np.linalg.norm(rec[2]) for rec in r["recs"])
    return ("%6.1f %3d %3d | %5.1f %5.1f | %5.1f %5.1f | %5.2f | %4.1f %5.2f | %5.2f %5.2f %5.1f%% | vmax %.3f solve_max %.0fms"
            % (r["dur"], r["n_replans"], r["fails"], lag_p95, max(lag) if lag else float("nan"),
               np.percentile(fv, 95), max(fv), r["wd_jump_max"], r["w_max"], r["wd_max"],
               max(ref), body.max(), 100 * np.mean(body > MARGIN), vmax, r["solve_max_ms"]))


HEADER = ("  variant          dur[s] rep fail | lag p95  max | fwd-vel p95 max | wdjump | "
          "|w|max |wd|max | wrench ref-frame / body-frame max ratio, body>%.1f %%" % MARGIN)


def main(variants, route_prefixes=()):
    for name, wps, q_raw, pre_align, vhw in ROUTES:
        if route_prefixes and not any(name.startswith(p + " ") for p in route_prefixes):
            continue
        wps = np.array(wps, dtype=float)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD) if pre_align else q_raw
        print("\n=== %s" % name)
        print(HEADER)
        for label, ft, kwargs in variants:
            try:
                g, _ = build_global(wps, q0, ft, True, via_half_width=vhw,
                                    body_frame=kwargs.get("body_frame", False))
            except MincoInfeasibleError as e:
                print("  %-16s GLOBAL FAIL %s" % (label, e))
                continue
            gp, gf = global_headings(g, q0)
            try:
                r = run_local(g, wps, q0, ft, **kwargs)
            except MincoInfeasibleError as e:
                print("  %-16s LOCAL FIRST FAIL %s" % (label, e))
                continue
            print("  %-18s %s" % (label, evaluate(r, q0, gp, gf)))


if __name__ == "__main__":
    main(VARIANT_SETS[sys.argv[1] if len(sys.argv) > 1 else "boundary"], sys.argv[2:])
