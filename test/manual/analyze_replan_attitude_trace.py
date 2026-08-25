#!/usr/bin/env python3
"""Post-process ``log_replanning_attitude_trace.py``'s CSV pair (no ROS).

Reuses the project's own canonical quaternion functions
(``control/utils/quat_math.py``, the same ``quat_rotate``/``geodesic_angle``
``attitude_reference.py`` itself is built on) rather than a hand-rolled
rotation matrix, to avoid a convention-transcription bug -- an earlier ad hoc
version of this analysis rolled its own rotation matrix and got a nonsense,
sign-flipped "pointing" result on one leg out of four (see the investigation
this was written for).

For each leg, per setpoint sample nearest in time to a TF sample (both
``t_sim`` columns MUST be on the same clock -- see ``--tf-uses-sim-time``
below), reports:

- ``q_des <-> q_act`` geodesic angle [deg] (the actual tracking error the
  replanning-mode sim verification found degraded, ``docs/archive/achieved/
  2026-08-25_guidance_realtime_replanning_sim_verification.md`` 6 節)
- ``v_des <-> v_act`` direction cosine (commanded vs actual velocity
  direction -- 6-4 節's "is translation tracking itself healthy" check)
- ``fwd_act <-> v_act`` direction cosine (does the vehicle's OWN current
  attitude already point along its OWN current motion -- a pointing-quality
  check independent of what was commanded)

split by ``likely_replan_tick`` to check whether error concentrates at
re-plan boundaries (Guidance-side jump) or is uniform (Control-side lag).
"""
import argparse
import csv
import sys

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_rotate

FORWARD_AXIS_BODY = np.array([1.0, 0.0, 0.0])  # matches attitude_reference.py's default


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def as_vec(row, prefix):
    return np.array([float(row[prefix + "x"]), float(row[prefix + "y"]), float(row[prefix + "z"])])


def as_quat(row, prefix="q"):
    return np.array([float(row[prefix + "x"]), float(row[prefix + "y"]),
                      float(row[prefix + "z"]), float(row[prefix + "w"])])


def cosine(a, b, min_norm=1e-4):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < min_norm or nb < min_norm:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def analyze_leg(sp_rows, tf_rows, moving_speed_threshold):
    tf_t = np.array([float(r["t_sim"]) for r in tf_rows])
    tf_pos = np.array([as_vec(r, "p") for r in tf_rows])
    tf_quat = np.array([as_quat(r) for r in tf_rows])

    dt = np.diff(tf_t)
    dt[dt <= 0] = np.nan
    v_act = np.zeros_like(tf_pos)
    v_act[1:] = (tf_pos[1:] - tf_pos[:-1]) / dt[:, None]

    out = []
    for r in sp_rows:
        t = float(r["t_sim"])
        idx = int(np.searchsorted(tf_t, t))
        idx = min(max(idx, 1), len(tf_t) - 1)
        if abs(tf_t[idx - 1] - t) < abs(tf_t[idx] - t):
            idx -= 1
        dt_match = abs(tf_t[idx] - t)

        v_des = as_vec(r, "v")
        q_des = as_quat(r)
        v_a = v_act[idx]
        q_act = tf_quat[idx]
        fwd_act = quat_rotate(q_act, FORWARD_AXIS_BODY)

        out.append({
            "leg": r["leg"],
            "leg_local_idx": int(r["leg_local_idx"]),
            "likely_replan_tick": r["likely_replan_tick"] == "True",
            "speed_des": float(np.linalg.norm(v_des)),
            "speed_act": float(np.linalg.norm(v_a)),
            "match_dt": dt_match,
            "q_err_deg": np.degrees(geodesic_angle(q_des, q_act)),
            "cos_v_des_v_act": cosine(v_des, v_a),
            "cos_fwd_act_v_act": cosine(fwd_act, v_a),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("setpoint_csv")
    ap.add_argument("tf_csv")
    ap.add_argument("--moving-speed-threshold", type=float, default=0.05,
                     help="exclude samples with |v_des| below this [m/s] from "
                          "the direction-cosine stats (direction is ill-"
                          "defined near rest) (default: %(default)s)")
    ap.add_argument("--max-match-dt", type=float, default=0.5,
                     help="warn if any setpoint<->TF time match exceeds this "
                          "many seconds -- a large value usually means the "
                          "two CSVs are not on the same clock (default: "
                          "%(default)s)")
    args = ap.parse_args()

    sp = load(args.setpoint_csv)
    tf = load(args.tf_csv)
    legs = sorted(set(r["leg"] for r in sp))

    for leg in legs:
        sp_leg = [r for r in sp if r["leg"] == leg]
        tf_leg = [r for r in tf if r["leg"] == leg]
        if len(tf_leg) < 2:
            print(f"=== {leg}: skipped (only {len(tf_leg)} TF samples) ===")
            continue

        rows = analyze_leg(sp_leg, tf_leg, args.moving_speed_threshold)
        max_dt = max(r["match_dt"] for r in rows)
        if max_dt > args.max_match_dt:
            print(f"WARNING: {leg}: worst setpoint<->TF time match is "
                  f"{max_dt:.3f}s -- check the two CSVs share the same clock "
                  f"(e.g. guidance_node's use_sim_time)", file=sys.stderr)

        moving = [r for r in rows if r["speed_des"] > args.moving_speed_threshold]
        replan = [r for r in moving if r["likely_replan_tick"]]
        non_replan = [r for r in moving if not r["likely_replan_tick"]]

        def stat(rs, key):
            vals = [r[key] for r in rs if not np.isnan(r[key])]
            return (np.mean(vals), np.max(vals), len(vals)) if vals else (np.nan, np.nan, 0)

        q_mean, q_max, n = stat(moving, "q_err_deg")
        q_mean_r, q_max_r, n_r = stat(replan, "q_err_deg")
        q_mean_nr, q_max_nr, n_nr = stat(non_replan, "q_err_deg")
        cv_mean, cv_min, _ = stat(moving, "cos_v_des_v_act")
        cf_mean, cf_min, _ = stat(moving, "cos_fwd_act_v_act")

        print(f"=== {leg} (n={len(rows)}, moving={len(moving)}, "
              f"worst_match_dt={max_dt:.4f}s) ===")
        print(f"  q_des<->q_act error [deg]: mean={q_mean:.2f} max={q_max:.2f} (n={n})")
        print(f"    at replan ticks:     mean={q_mean_r:.2f} max={q_max_r:.2f} (n={n_r})")
        print(f"    at non-replan ticks: mean={q_mean_nr:.2f} max={q_max_nr:.2f} (n={n_nr})")
        print(f"  v_des<->v_act cosine:      mean={cv_mean:.3f} min={cv_min:.3f}")
        print(f"  fwd_act<->v_act cosine:    mean={cf_mean:.3f} min={cf_min:.3f}")


if __name__ == "__main__":
    sys.exit(main())
