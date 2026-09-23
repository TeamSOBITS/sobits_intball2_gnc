"""Why v3 face travel fails on sharp turns / large start misalignment.

Prints the global trajectory's own face error, then a local ("both") time
series: forward vs velocity, vs nearest global heading, the local target's
attitude vs the current one, and body rate.
"""
import sys

import numpy as np

from experiment_v3_face_travel_feasibility import (
    FWD, SPEED_MIN, angle_deg, build_global, compute_q_des, global_sample_full, run_local,
)
from experiment_v3_face_travel_lead_diagnosis import fwd_of
from experiment_v3_face_travel_route_sweep import BOTH, LOCAL_FT, LOCAL_FT_GVIA, ROUTES


def main(names, local_kwargs):
    for name, wps, q_raw, pre_align, vhw in ROUTES:
        if not any(name.startswith(n + " ") for n in names):
            continue
        wps = np.array(wps, dtype=float)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD) if pre_align else q_raw
        g, _ = build_global(wps, q0, True, True, via_half_width=vhw)
        ts = np.arange(0.0, g.global_total_duration, 0.05)
        gs = [global_sample_full(g, t) for t in ts]
        gp = np.array([s[0] for s in gs])
        gf = np.array([fwd_of(q0, s[3]) for s in gs])
        gerr = [angle_deg(gf[i], gs[i][1]) for i in range(len(ts)) if np.linalg.norm(gs[i][1]) > SPEED_MIN]
        print("\n=== %s  global dur=%.1fs face_err med/p95/max=%.1f/%.1f/%.1f" % (
            name, g.global_total_duration, np.median(gerr), np.percentile(gerr, 95), max(gerr)))
        print("  global: t  |v|  face_err  |w|[deg/s]")
        for i in range(0, len(ts), int(len(ts) / 15) or 1):
            w = np.degrees(np.linalg.norm(g.sample_body_angular(ts[i])[0]))
            print("   %6.1f %.3f %6.1f %6.2f" % (ts[i], np.linalg.norm(gs[i][1]), angle_deg(gf[i], gs[i][1]), w))
        r = run_local(g, wps, q0, True, **local_kwargs)
        print("  local %s:" % local_kwargs, " t  |v|  fwd-vel  fwd-g_near  vel-g_near  |w|[deg/s]  nearest_global_t")
        recs = r["recs"]
        for rec in recs[::max(1, len(recs) // 20)]:
            t, p, v, rv, w, _wd = rec
            k = int(np.argmin(np.linalg.norm(gp - p, axis=1)))
            f = fwd_of(q0, rv)
            sp = np.linalg.norm(v)
            print("   %6.1f %.3f %6.1f %6.1f %6.1f %6.2f %6.1f" % (
                t, sp, angle_deg(f, v) if sp > 1e-6 else float("nan"), angle_deg(f, gf[k]),
                angle_deg(v, gf[k]) if sp > 1e-6 else float("nan"), np.degrees(np.linalg.norm(w)), ts[k]))
        sys.stdout.flush()


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "both"
    main(sys.argv[2:] or ["S3", "S5", "S13"], {"both": BOTH, "local_ft": LOCAL_FT, "local_ft_gvia": LOCAL_FT_GVIA}[variant])
