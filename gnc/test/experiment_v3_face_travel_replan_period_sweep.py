"""Does v3's post-corner attitude lag shrink when local replans are rarer?

If the lag comes from each replan restarting rotation at zero rotvec accel
(and ending at zero rotvec rate) while only the first period is used, a
longer period / replanning only at each local's end should shrink it.
"""
import numpy as np

from experiment_v3_face_travel_feasibility import (
    ROUTES, SPEED_MIN, FWD, angle_deg, build_global, compute_q_des,
    global_sample_full, run_local,
)
from experiment_v3_face_travel_lead_diagnosis import fwd_of

POSITION_ON_GLOBAL_DEG = 10.0
VARIANTS = [("1s", 1.0, False), ("2s", 2.0, False), ("3s", 3.0, False),
            ("5s", 5.0, False), ("local_end", None, True)]


def global_headings(g, q0):
    gs = [global_sample_full(g, t) for t in np.arange(0.0, g.global_total_duration, 0.05)]
    return np.array([s[0] for s in gs]), np.array([fwd_of(q0, s[3]) for s in gs])


def summarize(r, q0, gp, gf):
    fv, lag = [], []
    for _t, p, v, rv, _w, _wd in r["recs"]:
        if np.linalg.norm(v) <= SPEED_MIN:
            continue
        f = fwd_of(q0, rv)
        k = int(np.argmin(np.linalg.norm(gp - p, axis=1)))
        fv.append(angle_deg(f, v))
        if angle_deg(v, gf[k]) < POSITION_ON_GLOBAL_DEG:
            lag.append(angle_deg(f, gf[k]))
    lag_s = ("%5.1f/%5.1f/%5.1f %4d" % (np.median(lag), np.percentile(lag, 95), max(lag),
                                        len(lag))) if lag else "  n/a"
    return "%6.1f %5d %3d | %5.1f/%5.1f/%5.1f | %s | %.2f | %.1f %.2f" % (
        r["dur"], r["n_replans"], r["fails"], np.median(fv), np.percentile(fv, 95), max(fv),
        lag_s, r["wd_jump_max"], r["w_max"], r["wd_max"])


HEADER = ("  variant       dur[s] replans fails | fwd-vel med/p95/max | attitude lag (vel on global) "
          "med/p95/max n | wd_jump max[deg/s2] | |w|max[deg/s] |wd|max[deg/s2]")


def main():
    for name, (wps, q_raw) in ROUTES.items():
        wps = np.array(wps)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD)
        g, _ = build_global(wps, q0, True, True)
        gp, gf = global_headings(g, q0)
        print("\n=== %s" % name)
        print(HEADER)
        for label, period, at_end in VARIANTS:
            kwargs = {"replan_at_local_end": at_end}
            if period is not None:
                kwargs["period"] = period
            print("  %-12s %s" % (label, summarize(run_local(g, wps, q0, True, **kwargs), q0, gp, gf)))


if __name__ == "__main__":
    main()
