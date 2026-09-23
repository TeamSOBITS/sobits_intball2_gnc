"""Offline check of the candidate v3 face-travel config before sim.

Candidate: local position solved freely (1 segment), then re-solved on its own
path with face-travel attitude waypoints; rot_a0 carried; max_vel 0.2;
body-frame wrench; 4 m look-ahead. Runs one route at a time.
"""
import sys
import time

import numpy as np

from experiment_v3_face_travel_feasibility import build_global, compute_q_des, FWD, run_local
from experiment_v3_face_travel_replan_period_sweep import global_headings
from experiment_v3_face_travel_route_sweep import HEADER, ROUTES, Q_ID, evaluate, loc

CANDIDATE = {"carry_rot_accel": True, "local_self_path": True, "max_vel": 0.2,
             "body_frame": True, "horizon": 4.0}
SIM_START_P = [10.5026, -3.6347, 4.5033]
SIM_START_Q = np.array([-0.7071, 0.7071, -0.0001, 0.0])
EXTRA_ROUTES = [
    ("SIM cur->nav_entry->insp1", [np.array(SIM_START_P)] + loc("nav_entry", "inspection_entry_1"),
     SIM_START_Q, True, 0.0),
    ("S15 spiral 4x90 (cum 360)", [[0, 0, 0], [3, 0, 0], [3, 3, 0], [-1, 3, 0], [-1, -2, 0], [5, -2, 0]],
     Q_ID, True, 0.0),
]
VARIANTS = {
    "base": {},
    "fail3": {"fail_every": 3},
    "latency": {"solve_latency": True},
}


def main(variant, prefixes):
    print(HEADER + " | final dist | max ref step")
    for name, wps, q_raw, pre_align, vhw in ROUTES + EXTRA_ROUTES:
        if prefixes and not any(name.startswith(p + " ") for p in prefixes):
            continue
        wps = np.array(wps, dtype=float)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD) if pre_align else q_raw
        t0 = time.perf_counter()
        g, _ = build_global(wps, q0, True, True, via_half_width=vhw, body_frame=True)
        gp, gf = global_headings(g, q0)
        r = run_local(g, wps, q0, True, **CANDIDATE, **VARIANTS[variant])
        print("  %-38s %s | %.3f m | %.3f m | wall %.1fs" % (
            name[:38], evaluate(r, q0, gp, gf), np.linalg.norm(r["recs"][-1][1] - wps[-1]),
            r["max_step"], time.perf_counter() - t0))
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
