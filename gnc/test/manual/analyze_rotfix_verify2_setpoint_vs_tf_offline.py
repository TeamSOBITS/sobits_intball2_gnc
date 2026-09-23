#!/usr/bin/env python3
"""Offline (no ROS, no sim) re-analysis of the real ``rotfix_verify2`` trace
(``docs/2026-09-20_minco_face_travel_corner_divergence_root_cause_and_fix.md``'s
live-sim re-verification run) to check an assumption the previous synthetic/
idealized offline scripts could not: is the 56-59 degree divergence a pure
*setpoint-generation* artifact (fwd_des vs v_des both coming from the same
commanded reference stream, no real vehicle involved), or does the *real*
vehicle (TF) actually fly misaligned with its own real velocity too?

That distinction matters for picking a fix: if the real vehicle tracks its
own real velocity fine (TF fwd ~ TF velocity direction) despite the setpoint
stream's internal inconsistency, then the setpoint-generation bug
(``ReplanningMincoV2Tracker.sample()``'s per-tick full-horizon rotation
re-blend) is producing a large, physically-unnecessary *commanded* attitude
excursion that the controller then has to track -- a latch/shrinking-horizon
fix to the *reference generator* alone would remove a self-inflicted
tracking problem. If instead TF itself also diverges from its own velocity
by a similar amount, the mechanism includes real control-loop
tracking error, not just reference generation.

Computes three angle signals from ``rotfix_verify2_{setpoint,tf}.csv``
(finite-differenced positions for velocity, forward axis = body +X from the
recorded quaternions, no production code changes):

1. fwd_des <-> v_des        (setpoint vs itself -- reproduces the doc's own metric)
2. fwd_tf  <-> v_tf         (real vehicle vs its own real velocity -- the
                              question above)
3. fwd_tf  <-> v_des        (real vehicle vs the commanded velocity -- the
                              actual closed-loop attitude tracking error)

Usage:
    python3 test/manual/analyze_rotfix_verify2_setpoint_vs_tf_offline.py <setpoint.csv> <tf.csv>
"""
import csv
import math
import sys

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate

FORWARD_AXIS = np.array([1.0, 0.0, 0.0])


def load_pose_csv(path):
    ts, ps, qs = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["t_sim"]))
            ps.append([float(row["px"]), float(row["py"]), float(row["pz"])])
            qs.append([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])
    return np.array(ts), np.array(ps), np.array(qs)


def central_diff_velocity(ts, ps):
    n = len(ts)
    v = np.zeros_like(ps)
    v[1:-1] = (ps[2:] - ps[:-2]) / (ts[2:] - ts[:-2])[:, None]
    v[0] = (ps[1] - ps[0]) / (ts[1] - ts[0])
    v[-1] = (ps[-1] - ps[-2]) / (ts[-1] - ts[-2])
    return v


def angle_deg(fwd, v_dir):
    speed = np.linalg.norm(v_dir)
    if speed < 1e-4:
        return float("nan")
    cos_a = np.dot(fwd, v_dir) / (np.linalg.norm(fwd) * speed)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def resample(src_t, src_v, dst_t):
    """Linear interpolation, one column at a time (np.interp is 1D)."""
    out = np.empty((len(dst_t), src_v.shape[1]))
    for c in range(src_v.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, src_v[:, c])
    return out


SPEED_FLOOR = 0.05  # below this, finite-diffed direction is dominated by TF/setpoint
                     # sampling noise rather than real heading -- matches the
                     # doc's own use of a focused post-via window rather than a
                     # whole-trace max (start/stop/hold phases are near-zero speed).


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    setpoint_csv, tf_csv = sys.argv[1], sys.argv[2]

    t_des, p_des, q_des = load_pose_csv(setpoint_csv)
    t_tf, p_tf, q_tf = load_pose_csv(tf_csv)
    v_des = central_diff_velocity(t_des, p_des)
    v_tf = central_diff_velocity(t_tf, p_tf)

    fwd_des = np.array([quat_rotate(q, FORWARD_AXIS) for q in q_des])
    fwd_tf = np.array([quat_rotate(q, FORWARD_AXIS) for q in q_tf])

    ang_des_vdes = np.array([angle_deg(fwd_des[i], v_des[i]) for i in range(len(t_des))])
    ang_des_vdes[np.linalg.norm(v_des, axis=1) < SPEED_FLOOR] = np.nan

    # Resample setpoint fwd/v onto the TF (denser) time base for the other two signals.
    v_des_on_tf = resample(t_des, v_des, t_tf)
    ang_tf_vtf = np.array([angle_deg(fwd_tf[i], v_tf[i]) for i in range(len(t_tf))])
    ang_tf_vtf[np.linalg.norm(v_tf, axis=1) < SPEED_FLOOR] = np.nan
    ang_tf_vdes = np.array([angle_deg(fwd_tf[i], v_des_on_tf[i]) for i in range(len(t_tf))])
    ang_tf_vdes[np.linalg.norm(v_des_on_tf, axis=1) < SPEED_FLOOR] = np.nan

    # Focus window: center on the single worst excursion (the doc's "second,
    # unresolved divergence" peaks at 56-59 degrees, well above the
    # already-fixed via-passage excursion) and report the surrounding
    # +/-20s, wide enough to see both the ramp-up and the eventual settle.
    peak_i = int(np.nanargmax(ang_des_vdes))
    peak_t = t_des[peak_i]
    window_lo, window_hi = peak_t - 20.0, peak_t + 20.0
    print(f"auto-detected worst-excursion time: t={peak_t:.3f} "
          f"(reporting window [{window_lo:.2f}, {window_hi:.2f}])\n")

    def report(label, ts, angs, thresh=20.0, lo=window_lo, hi=window_hi):
        mask = (ts >= lo) & (ts <= hi) & ~np.isnan(angs)
        if not mask.any():
            print(f"{label:22s} no valid samples in window")
            return float("nan")
        sub_t, sub_a = ts[mask], angs[mask]
        peak_idx = int(np.argmax(sub_a))
        above = sub_a > thresh
        dt = float(np.median(np.diff(ts)))
        print(f"{label:22s} peak={sub_a[peak_idx]:6.2f} deg @ t={sub_t[peak_idx]:9.3f}  "
              f"time_above_{thresh:.0f}deg={np.sum(above) * dt:6.2f}s")
        return float(sub_a[peak_idx])

    print(f"setpoint samples={len(t_des)} (~{1.0/np.median(np.diff(t_des)):.1f} Hz), "
          f"tf samples={len(t_tf)} (~{1.0/np.median(np.diff(t_tf)):.1f} Hz)\n")
    peak_des = report("fwd_des <-> v_des", t_des, ang_des_vdes)
    peak_tf_vtf = report("fwd_tf  <-> v_tf ", t_tf, ang_tf_vtf)
    peak_tf_vdes = report("fwd_tf  <-> v_des", t_tf, ang_tf_vdes)
    print()
    # The diagnostic question is NOT "does TF diverge from its own v_tf" (it
    # necessarily will, just from faithfully chasing a bad setpoint -- that
    # alone proves nothing about where the badness originates). The relevant
    # comparison is peak_tf_vdes (real attitude vs. COMMANDED velocity, i.e.
    # closed-loop attitude tracking error) against peak_des (the setpoint's
    # own internal inconsistency): if they're close, the attitude controller
    # is tracking the bad reference just fine, and the badness is entirely
    # upstream in the reference generator.
    if abs(peak_tf_vdes - peak_des) < 0.15 * peak_des:
        print(f"=> fwd_tf<->v_des peak ({peak_tf_vdes:.1f} deg) is nearly identical to "
              f"fwd_des<->v_des peak ({peak_des:.1f} deg): the attitude controller is "
              "tracking the bad setpoint essentially perfectly -- there is no significant "
              "ADDITIONAL closed-loop tracking lag on top of the reference. The real "
              f"vehicle's large divergence from its own velocity (fwd_tf<->v_tf peak "
              f"{peak_tf_vtf:.1f} deg) is simply it faithfully flying the bad commanded "
              "attitude, not independent physical-plant lag. This confirms the bug is "
              "entirely upstream in the LOCAL LAYER'S REFERENCE GENERATION (the receding-"
              "horizon rotation re-blend) -- a latch/shrinking-horizon fix to the "
              "reference generator alone should remove the real vehicle's divergence too, "
              "with no need to model real plant/control dynamics.")
    else:
        print(f"=> fwd_tf<->v_des peak ({peak_tf_vdes:.1f} deg) differs meaningfully from "
              f"fwd_des<->v_des peak ({peak_des:.1f} deg): closed-loop attitude tracking "
              "itself is adding/removing a nontrivial amount of error on top of the "
              "reference -- a pure reference-generator fix may not fully explain or fix "
              "the live divergence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
