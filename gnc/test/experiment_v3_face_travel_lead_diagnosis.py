"""Does v3's local attitude turn ahead of its own velocity near corners?

Reuses experiment_v3_face_travel_feasibility's local replica (heuristic-time
global, pre_align q0, face_travel=True) and compares each local sample's
forward axis against its velocity and against the global heading at the
nearest global point / at the local target 2m ahead.
"""
import numpy as np

from experiment_v3_face_travel_feasibility import (
    FWD, HORIZON, ROUTES, SPEED_MIN, angle_deg, build_global, compute_q_des,
    global_sample_full, run_local,
)
from sobits_intball2_gnc.control.utils.quat_math import quat_exp, quat_mul, quat_rotate


def fwd_of(q0, rv):
    return quat_rotate(quat_mul(q0, quat_exp(rv)), FWD)


def turn_phase(x, e1, e2):
    return np.degrees(np.arctan2(x @ e2, x @ e1))


def main():
    for name, (wps, q_raw) in ROUTES.items():
        wps = np.array(wps)
        q0 = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD)
        g, _ = build_global(wps, q0, True, True)
        ts = np.arange(0.0, g.global_total_duration, 0.05)
        gs = [global_sample_full(g, t) for t in ts]
        gp = np.array([s[0] for s in gs])
        gf = np.array([fwd_of(q0, s[3]) for s in gs])
        garc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])

        corners = []
        for i in range(1, len(wps) - 1):
            e1 = (wps[i] - wps[i - 1]) / np.linalg.norm(wps[i] - wps[i - 1])
            out = wps[i + 1] - wps[i]
            e2 = out - (out @ e1) * e1
            corners.append((wps[i], e1, e2 / np.linalg.norm(e2)))

        recs = run_local(g, wps, q0, True)["recs"]
        rows = []
        for t, p, v, rv, _w, _wd in recs:
            if np.linalg.norm(v) <= SPEED_MIN:
                continue
            f = fwd_of(q0, rv)
            k = int(np.argmin(np.linalg.norm(gp - p, axis=1)))
            k_ahead = min(int(np.searchsorted(garc, garc[k] + HORIZON)), len(gp) - 1)
            c_pos, e1, e2 = min(corners, key=lambda c: np.linalg.norm(c[0] - p))
            rows.append((
                t, np.linalg.norm(p - c_pos), angle_deg(f, v),
                turn_phase(f, e1, e2) - turn_phase(v, e1, e2),
                angle_deg(f, gf[k]), angle_deg(f, gf[k_ahead]), angle_deg(v, gf[k]),
            ))
        rows = np.array(rows)
        big = rows[rows[:, 2] > 10.0]
        print("\n=== %s" % name)
        print("  samples with |fwd-vel|>10deg: %d/%d" % (len(big), len(rows)))
        if len(big):
            print("    lead>0 (fwd turned further toward next leg than vel): %.0f%%"
                  % (100 * np.mean(big[:, 3] > 0)))
            print("    median angle fwd vs global-heading nearest=%.1f  2m-ahead=%.1f  "
                  "vel vs global-heading nearest=%.1f" % (
                      np.median(big[:, 4]), np.median(big[:, 5]), np.median(big[:, 6])))
        print("  t[s]  dist_to_corner[m]  fwd-vel  lead  fwd-g_near  fwd-g_ahead  vel-g_near")
        for r in rows[::20]:
            print("  %5.1f  %5.2f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f" % tuple(r))


if __name__ == "__main__":
    main()
