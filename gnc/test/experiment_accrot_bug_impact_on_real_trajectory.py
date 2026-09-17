#!/usr/bin/env python3
"""Offline, sim-free check: on an ACTUAL trajectory produced by the unmodified,
already-built minco_native_py (no source edits, no rebuild), how much does the
accRot bug (see experiment_accrot_jacobian_bug_quantification.py) actually
matter?

Calls minco_native_py.plan_minco() exactly as-is (production behavior,
untouched) on the same 3-waypoint / large-rotation scenario already used in
test_minco_native_regression.py (RV1 has magnitude ~95deg across 2 axes --
a real, already-validated "large-angle, multi-axis" case, not a synthetic
worst case). Then, post-hoc, re-evaluates the wrench envelope constraint
along the resulting rotation-vector trajectory two ways:

  - "buggy" : tau = INERTIA * r_ddot(t)         (what minco_solver.cpp used
                                                  while solving -- this is
                                                  what determined the shape
                                                  and duration of the plan
                                                  actually returned)
  - "true"  : tau = INERTIA * omega_dot(t)      (the physically correct
                                                  torque this trajectory
                                                  actually demands)

This does NOT re-run the optimizer with the fix (that needs a source change,
out of scope here) -- it only asks "given the plan the buggy solver already
committed to, how far off was its own torque bookkeeping, and did the real
torque demand secretly exceed the envelope it thought it was respecting?".
That bounds how much headroom fixing the bug could plausibly recover,
without touching minco_solver.cpp.

Usage: python3 gnc/test/experiment_accrot_bug_impact_on_real_trajectory.py
"""
import numpy as np

import minco_native_py

MASS = 3.216
INERTIA = 0.0136
ENVELOPE_CSV = "minco_native_py/config/wrench_envelope.csv"
WRENCH_SAFETY_MARGIN = 1.0  # matches plan_minco's default used below

NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5
RV1 = np.array([0.0, 0.35864857, 1.6255471])


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def right_jacobian(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    K = skew(r)
    a = (1 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) - a * K + b * (K @ K)


def load_envelope(path):
    with open(path) as f:
        n, cols = (int(x) for x in f.readline().split())
        rows = [list(map(float, f.readline().split())) for _ in range(n)]
    rows = np.array(rows)
    return rows[:, :-1], rows[:, -1]


def beta_vectors(t):
    beta0 = np.array([1, t, t**2, t**3, t**4, t**5])
    beta1 = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
    beta2 = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])
    return beta0, beta1, beta2


def flatten_waypoint(pos, rot):
    return list(pos) + list(rot)


def analyze(label, waypoints):
    success, error_code, segment_times, coeffs_flat, duration = minco_native_py.plan_minco(
        waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert success and error_code == 0, f"{label}: unmodified solver didn't converge"
    print(f"\n=== {label} ===")
    print(f"plan_minco (unmodified): K={len(segment_times)} segments, duration={duration:.4f}s")

    K = len(segment_times)
    # layout per minco_solver.cpp:435-453: [seg][dim(px,py,pz,rx,ry,rz)][degree(0..5)]
    coeffs = np.array(coeffs_flat).reshape(K, 6, 6)

    F_env, g_env = load_envelope(ENVELOPE_CSV)

    tau_buggy_all, tau_true_all = [], []
    max_true_violation = -np.inf
    max_buggy_violation = -np.inf
    h = 1e-6

    for seg in range(K):
        cPos = coeffs[seg, 0:3, :]  # (3 dims, 6 degrees)
        cRot = coeffs[seg, 3:6, :]
        T_seg = segment_times[seg]
        for t in np.linspace(0.02 * T_seg, 0.98 * T_seg, 60):
            beta0, beta1, beta2 = beta_vectors(t)
            accPos = cPos @ beta2
            r_ddot = cRot @ beta2

            def omega(tt):
                b0, b1, _ = beta_vectors(tt)
                return right_jacobian(cRot @ b0) @ (cRot @ b1)

            omega_dot_true = (omega(t + h) - omega(t - h)) / (2 * h)

            tau_buggy = INERTIA * r_ddot
            tau_true = INERTIA * omega_dot_true
            tau_buggy_all.append(tau_buggy)
            tau_true_all.append(tau_true)

            wrench_buggy = np.concatenate([MASS * accPos, tau_buggy])
            wrench_true = np.concatenate([MASS * accPos, tau_true])
            viol_buggy = np.max(F_env @ wrench_buggy - WRENCH_SAFETY_MARGIN * g_env)
            viol_true = np.max(F_env @ wrench_true - WRENCH_SAFETY_MARGIN * g_env)
            max_buggy_violation = max(max_buggy_violation, viol_buggy)
            max_true_violation = max(max_true_violation, viol_true)

    # Peak-normalized, not instantaneous-relative -- an instantaneous-relative
    # metric spikes to spurious huge numbers near tau_true's zero-crossings
    # (verified in experiment_accrot_jacobian_bug_quantification.py).
    tau_buggy_all = np.array(tau_buggy_all)
    tau_true_all = np.array(tau_true_all)
    peak_true = np.max(np.linalg.norm(tau_true_all, axis=1))
    rel_errs = np.linalg.norm(tau_buggy_all - tau_true_all, axis=1) / peak_true
    print(f"torque error vs. peak true torque across all segments: mean={np.mean(rel_errs):.1%}"
          f", max={np.max(rel_errs):.1%}")
    print(f"max envelope violation using solver's own (buggy) torque bookkeeping: {max_buggy_violation:+.4f}"
          " (<=0 means the solver correctly believed it stayed inside the envelope)")
    print(f"max envelope violation using the PHYSICALLY TRUE torque on that same plan: {max_true_violation:+.4f}"
          " (>0 would mean the real vehicle secretly exceeds the envelope on this plan)")


def main():
    midpoint = 0.5 * (NEAR_DOCK + ABOVE_DOCK)
    bulge = NAV_ENTRY - midpoint
    q1_given = midpoint + BULGE_SCALE * bulge

    analyze(
        "regression-test scenario (via and end share the same RV1 -> near-fixed rotation axis)",
        flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
        + flatten_waypoint(q1_given, RV1)
        + flatten_waypoint(ABOVE_DOCK, RV1),
    )

    analyze(
        "genuine axis-change scenario (via=1.2rad about x, end=1.2rad about z -- "
        "same positions, rotations chosen to force the rotvec path to bend)",
        flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
        + flatten_waypoint(q1_given, [1.2, 0.0, 0.0])
        + flatten_waypoint(ABOVE_DOCK, [0.0, 0.0, 1.2]),
    )


if __name__ == "__main__":
    main()
