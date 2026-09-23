"""Does carrying rotation state across v3 local replans remove the attitude lag?

Period stays 1s. Compares the current rest-to-rest rotation boundaries with
rot_a0 (head rotvec accel from the previous local) and/or rot_v_tail (tail
rotvec rate from the global trajectory at the local target).
"""
import numpy as np

from experiment_v3_face_travel_feasibility import (
    FWD, ROUTES, build_global, compute_q_des, run_local,
)
from experiment_v3_face_travel_replan_period_sweep import HEADER, global_headings, summarize

VARIANTS = [
    ("rest-to-rest", {}),
    ("rot_a0", {"carry_rot_accel": True}),
    ("rot_v_tail", {"rot_rate_tail": True}),
    ("both", {"carry_rot_accel": True, "rot_rate_tail": True}),
]


def main():
    for name, (wps, q_raw) in ROUTES.items():
        wps = np.array(wps)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD)
        g, _ = build_global(wps, q0, True, True)
        gp, gf = global_headings(g, q0)
        print("\n=== %s" % name)
        print(HEADER)
        for label, kwargs in VARIANTS:
            print("  %-12s %s" % (label, summarize(run_local(g, wps, q0, True, **kwargs), q0, gp, gf)))


if __name__ == "__main__":
    main()
