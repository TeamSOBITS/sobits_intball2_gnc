#!/usr/bin/env python3
"""Offline validation of a design in docs/archive/achieved/
2026-09-17_replanning_minco_v2_local_layer_feasibility_governor_design.md
(deferred -- based on a misdiagnosed incident, see that doc's correction note):
give ReplanningMincoV2Tracker's local layer real wrench-envelope
feasibility awareness (the same ``(F, g)`` halfspace polytope the
static/TOPP-RA path already uses) instead of blindly committing to a fixed
``t_local`` window, by growing the local segment's duration (mirroring
``experiment_replanning_closed_loop.py``'s existing T-update rule: x1.5 per
retry, capped by time remaining until the next global replan) until the
worst-case wrench along the segment is feasible.

This is a standalone follow-up to the first offline check (a rotation-only,
noise-free toy in the design doc); it covers the three items that doc's
"次のアクション" left open:
  1. combined translation+rotation (a single 6D wrench, not rotation alone)
  2. how often the duration-growth loop falls back to "infeasible even at
     the cap" across scenarios of varying severity, and whether that
     degrades tracking
  3. interaction with MODEL_KF estimation noise (using the real
     ``ModelKfEstimator``, not an idealized true-state readout)

Does NOT touch guidance/ or minco_native_py -- calls the real, unmodified
``solve_quintic_hermite_coeffs`` and ``ModelKfEstimator``, and loads the
real (dense, 9951-facet) fan wrench envelope from
``gnc/test/experiment_minco_native/wrench_envelope.csv``.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_local_layer_feasibility_governor.py
"""
import numpy as np

from sobits_intball2_gnc.guidance.utils.model_kf_estimator import ModelKfEstimator
from sobits_intball2_gnc.guidance.utils.quintic_hermite import (
    solve_quintic_hermite_coeffs,
)

MASS = 4.5        # trajectory_controller.mass default
INERTIA = 0.0136  # trajectory_controller.inertia default
T_LOCAL_NOMINAL = 1.0   # ReplanningMincoV2Tracker DEFAULT_T_LOCAL_S
GLOBAL_REPLAN_PERIOD = 2.0  # DEFAULT_GLOBAL_REPLAN_PERIOD_S
DT = 0.05
GROWTH_FACTOR = 1.5
MAX_RETRIES = 5
ENVELOPE_PATH = "gnc/test/experiment_minco_native/wrench_envelope.csv"

# ModelKfEstimator defaults (replanning_minco_v2_tracker.py)
DEFAULT_Q_ACCEL_STD = 0.01
DEFAULT_R_STD = 0.001


def load_envelope(path):
    with open(path) as f:
        n, dim = map(int, f.readline().split())
        data = np.loadtxt(f, max_rows=n)
    return data[:, :dim], data[:, dim]


F, G = load_envelope(ENVELOPE_PATH)


def smooth_ramp(tau):
    """Quintic 0->1 ramp with zero end velocity/accel (rest-to-rest)."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def make_profile(total, duration):
    def profile(t):
        t = min(max(t, 0.0), duration)
        s, ds, dds = smooth_ramp(t / duration)
        return s * total, ds * total / duration, dds * total / duration**2
    return profile


def accel_at(c, tau):
    return 2 * c[2] + 6 * c[3] * tau + 12 * c[4] * tau**2 + 20 * c[5] * tau**3


def worst_case_wrench_ok(coeffs_pos, coeffs_rot, T, n_samples=25, margin=1.0,
                          check_horizon=None):
    """check_horizon caps how far into [0, T] the feasibility check looks.
    None means the full segment (the original, overly conservative
    design); a finite value models that only the near-term portion of a
    freshly re-solved segment is ever actually executed before the next
    tick discards it and re-solves from the (by-then-updated) real state."""
    horizon = T if check_horizon is None else min(T, check_horizon)
    taus = np.linspace(0.0, horizon, n_samples)
    for tau in taus:
        a_pos = np.array([accel_at(coeffs_pos[i], tau) for i in range(3)])
        a_rot = np.array([accel_at(coeffs_rot[i], tau) for i in range(3)])
        w = np.concatenate([MASS * a_pos, INERTIA * a_rot])
        if not np.all(F @ w <= G * margin + 1e-9):
            return False
    return True


def scale_to_feasible(w_des):
    """Realistic fan-saturation model for the *ungoverned* baseline's true
    dynamics: uniformly scale the desired 6D wrench toward the origin
    (which the envelope always contains) until it's feasible, rather than
    an arbitrary per-axis clip. Returns (w_applied, scale)."""
    if np.all(F @ w_des <= G):
        return w_des, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if np.all(F @ (mid * w_des) <= G):
            lo = mid
        else:
            hi = mid
    return lo * w_des, lo


def solve_feasible_segment(p0, v0, rv0, w0, pos_profile, rot_profile,
                            global_elapsed, t_max, check_horizon=None):
    """T-update rule: grow T x1.5 up to MAX_RETRIES, capped at t_max (time
    left until the next global replan), stop as soon as the combined
    (position + rotation) wrench is feasible over [0, check_horizon]
    (None = the whole freshly-solved segment)."""
    T = T_LOCAL_NOMINAL
    coeffs_pos = coeffs_rot = None
    for _ in range(MAX_RETRIES + 1):
        T = min(T, t_max)
        tgt_p, tgt_v, tgt_a = pos_profile(global_elapsed + T)
        tgt_rv, tgt_rv_v, tgt_rv_a = rot_profile(global_elapsed + T)
        coeffs_pos = solve_quintic_hermite_coeffs(
            p0, v0, np.zeros(3),
            np.array([tgt_p, 0, 0]), np.array([tgt_v, 0, 0]), np.array([tgt_a, 0, 0]),
            T,
        )
        coeffs_rot = solve_quintic_hermite_coeffs(
            rv0, w0, np.zeros(3),
            np.array([tgt_rv, 0, 0]), np.array([tgt_rv_v, 0, 0]), np.array([tgt_rv_a, 0, 0]),
            T,
        )
        ok = worst_case_wrench_ok(coeffs_pos, coeffs_rot, T, check_horizon=check_horizon)
        if ok or T >= t_max:
            return coeffs_pos, coeffs_rot, T, ok
        T *= GROWTH_FACTOR
    return coeffs_pos, coeffs_rot, T, False


def run_combined(label, governed, noisy, duration_s=None,
                  pos_total=2.0, pos_duration=30.0,
                  rot_total_deg=90.0, rot_duration=8.0, seed=0,
                  check_horizon=None):
    if duration_s is None:
        duration_s = max(pos_duration, rot_duration) + 10.0
    rng = np.random.default_rng(seed)
    pos_profile = make_profile(pos_total, pos_duration)
    rot_profile = make_profile(np.radians(rot_total_deg), rot_duration)

    x, v = 0.0, 0.0
    rv, w = 0.0, 0.0
    global_elapsed = 0.0
    max_pos_err = 0.0
    max_rot_err = 0.0
    fallback_count = 0
    n_ticks = int(duration_s / DT)

    pos_kf = ModelKfEstimator(3, DEFAULT_Q_ACCEL_STD, DEFAULT_R_STD)
    rot_kf = ModelKfEstimator(3, DEFAULT_Q_ACCEL_STD, DEFAULT_R_STD)
    last_a_pos = np.zeros(3)
    last_a_rot = np.zeros(3)

    for step in range(n_ticks):
        if noisy:
            z_pos = np.array([x, 0, 0]) + rng.normal(0, DEFAULT_R_STD, 3)
            z_rot = np.array([rv, 0, 0]) + rng.normal(0, DEFAULT_R_STD, 3)
            pos_est, vel_est = pos_kf.step(z_pos, last_a_pos, DT)
            rot_est, angvel_est = rot_kf.step(z_rot, last_a_rot, DT)
            p0, v0 = pos_est, vel_est
            rv0, w0 = rot_est, angvel_est
        else:
            p0, v0 = np.array([x, 0, 0]), np.array([v, 0, 0])
            rv0, w0 = np.array([rv, 0, 0]), np.array([w, 0, 0])

        if governed:
            t_max = GLOBAL_REPLAN_PERIOD - (global_elapsed % GLOBAL_REPLAN_PERIOD)
            t_max = max(t_max, T_LOCAL_NOMINAL)
            coeffs_pos, coeffs_rot, T_used, ok = solve_feasible_segment(
                p0, v0, rv0, w0, pos_profile, rot_profile, global_elapsed, t_max,
                check_horizon=check_horizon,
            )
            if not ok:
                fallback_count += 1
            a_pos_cmd = np.array([accel_at(coeffs_pos[i], DT) for i in range(3)])
            a_rot_cmd = np.array([accel_at(coeffs_rot[i], DT) for i in range(3)])
            w_des = np.concatenate([MASS * a_pos_cmd, INERTIA * a_rot_cmd])
            w_applied, _scale = scale_to_feasible(w_des)  # should already be ~feasible
        else:
            tgt_p, tgt_v, tgt_a = pos_profile(global_elapsed + T_LOCAL_NOMINAL)
            tgt_rv, tgt_rv_v, tgt_rv_a = rot_profile(global_elapsed + T_LOCAL_NOMINAL)
            coeffs_pos = solve_quintic_hermite_coeffs(
                p0, v0, np.zeros(3),
                np.array([tgt_p, 0, 0]), np.array([tgt_v, 0, 0]), np.array([tgt_a, 0, 0]),
                T_LOCAL_NOMINAL,
            )
            coeffs_rot = solve_quintic_hermite_coeffs(
                rv0, w0, np.zeros(3),
                np.array([tgt_rv, 0, 0]), np.array([tgt_rv_v, 0, 0]), np.array([tgt_rv_a, 0, 0]),
                T_LOCAL_NOMINAL,
            )
            a_pos_cmd = np.array([accel_at(coeffs_pos[i], DT) for i in range(3)])
            a_rot_cmd = np.array([accel_at(coeffs_rot[i], DT) for i in range(3)])
            w_des = np.concatenate([MASS * a_pos_cmd, INERTIA * a_rot_cmd])
            w_applied, _scale = scale_to_feasible(w_des)  # real fan saturation

        a_pos_applied = w_applied[:3] / MASS
        a_rot_applied = w_applied[3:] / INERTIA
        last_a_pos, last_a_rot = a_pos_applied, a_rot_applied

        v += a_pos_applied[0] * DT
        x += v * DT
        w += a_rot_applied[0] * DT
        rv += w * DT

        true_x, _, _ = pos_profile(global_elapsed)
        true_rv, _, _ = rot_profile(global_elapsed)
        max_pos_err = max(max_pos_err, abs(x - true_x))
        max_rot_err = max(max_rot_err, abs(rv - true_rv))
        global_elapsed += DT

    final_true_x, _, _ = pos_profile(global_elapsed)
    final_true_rv, _, _ = rot_profile(global_elapsed)
    print(
        f"{label}: max_pos_err={max_pos_err*1000:8.1f}mm  "
        f"max_rot_err={np.degrees(max_rot_err):7.2f}deg  "
        f"final_pos_err={abs(x-final_true_x)*1000:7.2f}mm  "
        f"final_rot_err={np.degrees(abs(rv-final_true_rv)):6.2f}deg  "
        f"fallback_ticks={fallback_count}/{n_ticks}"
    )


print("=== 0. does checking the WHOLE [0,T] window over-trigger on an already-feasible profile? ===")
print("(every tick re-solves from scratch and only ever executes ~[0,dt] before")
print(" the next tick discards it -- checking far-future tau the segment will")
print(" never actually be run at is a plausible source of spurious fallbacks)")
run_combined("baseline (ungoverned)          ", governed=False, noisy=False)
run_combined("governed, full-window check    ", governed=True, noisy=False, check_horizon=None)
for h in (2 * DT, 5 * DT, 10 * DT):
    run_combined(f"governed, check_horizon={h:.2f}s      ", governed=True, noisy=False, check_horizon=h)

CHECK_HORIZON = 5 * DT  # chosen from the sweep above

print("\n=== 1. combined translation+rotation, no noise (check_horizon=%.2fs) ===" % CHECK_HORIZON)
run_combined("baseline (ungoverned)", governed=False, noisy=False)
run_combined("feasibility-governed  ", governed=True, noisy=False, check_horizon=CHECK_HORIZON)

print("\n=== 2. combined translation+rotation, with MODEL_KF + realistic noise ===")
run_combined("baseline (ungoverned)", governed=False, noisy=True)
run_combined("feasibility-governed  ", governed=True, noisy=True, check_horizon=CHECK_HORIZON)

print("\n=== 3. fallback-rate sensitivity across scenario severity (governed, no noise) ===")
for rot_total_deg, rot_duration in [(30, 8), (60, 8), (90, 8), (90, 4), (120, 4), (164, 3)]:
    run_combined(
        f"turn={rot_total_deg:5.1f}deg/{rot_duration}s",
        governed=True, noisy=False, check_horizon=CHECK_HORIZON,
        rot_total_deg=rot_total_deg, rot_duration=rot_duration,
    )

print("\n=== 4. fallback-rate sensitivity across scenario severity (governed, with noise) ===")
for rot_total_deg, rot_duration in [(30, 8), (60, 8), (90, 8), (90, 4), (120, 4), (164, 3)]:
    run_combined(
        f"turn={rot_total_deg:5.1f}deg/{rot_duration}s",
        governed=True, noisy=True, check_horizon=CHECK_HORIZON,
        rot_total_deg=rot_total_deg, rot_duration=rot_duration,
    )

print("\n=== 5. re-run the ORIGINAL incident-scale scenario (design-doc rotation-only)")
print("    with the horizon-limited check, to confirm it still catches the real failure ===")
run_combined("baseline (ungoverned)         ", governed=False, noisy=False,
             pos_total=0.0, pos_duration=8.0, rot_total_deg=90.0, rot_duration=8.0)
run_combined("governed, full-window check   ", governed=True, noisy=False, check_horizon=None,
             pos_total=0.0, pos_duration=8.0, rot_total_deg=90.0, rot_duration=8.0)
run_combined("governed, check_horizon=%.2fs  " % CHECK_HORIZON, governed=True, noisy=False,
             check_horizon=CHECK_HORIZON,
             pos_total=0.0, pos_duration=8.0, rot_total_deg=90.0, rot_duration=8.0)
