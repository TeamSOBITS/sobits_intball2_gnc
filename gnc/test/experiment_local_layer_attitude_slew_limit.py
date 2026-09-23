#!/usr/bin/env python3
"""Offline validation of a quick-fix for the duty-saturation incident in
docs/archive/achieved/2026-09-17_replanning_minco_v2_local_layer_duty_saturation_incident.md:
``ReplanningMincoV2Tracker``'s local layer (``guidance/utils/quintic_hermite.py``)
feeds whatever rotvec look-ahead target the global MINCO trajectory produces
directly into ``solve_quintic_hermite_coeffs`` with no feasibility check, so
a large attitude-reference jump (observed mid-route, ~100+deg within one
``t_local``=1.0s window) produces a huge peak angular acceleration -- more
torque than the real fans can deliver, which starves translation thrust
(same 8 fans) and was the root cause of the ~2.8m position tracking failure
seen in the Phase 5 sim run (2026-09-17).

This script does NOT touch guidance/ or minco_native_py -- it only calls the
real, unmodified ``solve_quintic_hermite_coeffs`` (the actual local-layer
primitive) with representative boundary conditions, to answer: does clamping
the rotvec target fed into that solve (a slew-rate limit applied *before*
the Hermite fit, not inside it) bring the peak required angular acceleration
back under a known-safe bound?

Known-safe reference: guidance.align_angular_accel_deg=2.4 deg/s^2 (guidance.py
default) -- the accel the vehicle's *pure-rotation* align phase already uses
without saturating duty (its comment notes control_node's gains are tuned
around this value). Using it here as the bound is conservative in the
attitude-only sense but optimistic in the coupled sense (real budget is
lower once translation thrust is also drawn from the same fans) -- good
enough to show whether the mitigation direction is worth prototyping in
guidance/, not a final feasibility proof.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_local_layer_attitude_slew_limit.py
"""
import numpy as np

from sobits_intball2_gnc.guidance.utils.quintic_hermite import (
    solve_quintic_hermite_coeffs,
)

T_LOCAL_S = 1.0  # ReplanningMincoV2Tracker DEFAULT_T_LOCAL_S
ALIGN_ANGULAR_ACCEL_DEG = 2.4  # guidance.align_angular_accel_deg default
SAFE_ACCEL_RAD = np.radians(ALIGN_ANGULAR_ACCEL_DEG)

# Observed incident: attitude error climbed from ~4deg to ~82deg within
# roughly one global-replan tick around t=42-50s (see the incident doc's
# time-series table). Reproduced here as a rest-to-rest rotvec jump on a
# single axis -- current state at rest (rotvec_obs=0, angvel_est=0), target
# likewise at rest at the look-ahead point (tgt_rv_vel=tgt_rv_accel=0),
# same simplification ReplanningMincoV2Tracker uses whenever the global
# trajectory's attitude reference is itself near-static at the look-ahead
# time (e.g. approaching a waypoint's fixed target attitude).
OBSERVED_JUMP_DEG = 90.0


def peak_angular_accel(rotvec_delta_deg, t_local=T_LOCAL_S):
    """Peak |accel| (rad/s^2) of the rest-to-rest quintic Hermite segment
    that moves a single rotation axis by ``rotvec_delta_deg`` in ``t_local``
    seconds -- the exact call ReplanningMincoV2Tracker._sample_locked makes
    for the rotation boundary conditions."""
    delta = np.radians(rotvec_delta_deg)
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    a0 = np.zeros(3)
    p1 = np.array([delta, 0.0, 0.0])
    v1 = np.zeros(3)
    a1 = np.zeros(3)
    coeffs = solve_quintic_hermite_coeffs(p0, v0, a0, p1, v1, a1, t_local)
    # coeffs[axis] ascending power; acceleration poly = 2*c2 + 6*c3*t + 12*c4*t^2 + 20*c5*t^3
    c = coeffs[0]
    taus = np.linspace(0.0, t_local, 200)
    accel = 2 * c[2] + 6 * c[3] * taus + 12 * c[4] * taus**2 + 20 * c[5] * taus**3
    return float(np.max(np.abs(accel)))


def clamp_rotvec_delta(rotvec_delta_deg, t_local=T_LOCAL_S, max_accel_rad=SAFE_ACCEL_RAD):
    """Quick-fix candidate: binary-search the largest delta (<=
    rotvec_delta_deg) whose rest-to-rest quintic peak accel fits under
    max_accel_rad, and use that as the look-ahead target instead of the
    unclamped one. This is the "slew-rate limit before the Hermite solve"
    mitigation described in the incident doc, applied here in its simplest
    single-axis form."""
    if peak_angular_accel(rotvec_delta_deg, t_local) <= max_accel_rad:
        return rotvec_delta_deg
    lo, hi = 0.0, rotvec_delta_deg
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if peak_angular_accel(mid, t_local) <= max_accel_rad:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    print("=== baseline: unclamped look-ahead rotvec target ===")
    baseline_peak = peak_angular_accel(OBSERVED_JUMP_DEG)
    print(
        f"jump={OBSERVED_JUMP_DEG:.1f}deg over t_local={T_LOCAL_S:.1f}s -> "
        f"peak accel={np.degrees(baseline_peak):.1f}deg/s^2 "
        f"(safe bound={ALIGN_ANGULAR_ACCEL_DEG:.1f}deg/s^2, "
        f"{baseline_peak / SAFE_ACCEL_RAD:.1f}x over)"
    )

    print("\n=== mitigation: clamp look-ahead rotvec target pre-Hermite ===")
    clamped_deg = clamp_rotvec_delta(OBSERVED_JUMP_DEG)
    clamped_peak = peak_angular_accel(clamped_deg)
    print(
        f"clamped target={clamped_deg:.2f}deg (of {OBSERVED_JUMP_DEG:.1f}deg requested) -> "
        f"peak accel={np.degrees(clamped_peak):.2f}deg/s^2 "
        f"({'OK' if clamped_peak <= SAFE_ACCEL_RAD * 1.01 else 'FAIL'}, "
        f"bound={ALIGN_ANGULAR_ACCEL_DEG:.1f}deg/s^2)"
    )

    print(
        "\nNote: with the target clamped to "
        f"{clamped_deg:.2f}deg per {T_LOCAL_S:.1f}s tick, closing the full "
        f"{OBSERVED_JUMP_DEG:.1f}deg gap takes about "
        f"{OBSERVED_JUMP_DEG / clamped_deg * T_LOCAL_S:.1f}s of successive "
        "global-replan ticks instead of one -- attitude convergence gets "
        "slower, but duty stays available for translation throughout, which "
        "is the actual trade-off being proposed."
    )

    print("\n=== sweep: peak accel vs. jump size (unclamped) ===")
    for jump_deg in (5.0, 15.0, 30.0, 60.0, 90.0, 120.0, 164.0):
        peak = peak_angular_accel(jump_deg)
        ratio = peak / SAFE_ACCEL_RAD
        flag = "OK" if ratio <= 1.0 else f"{ratio:.1f}x OVER"
        print(f"  jump={jump_deg:6.1f}deg -> peak accel={np.degrees(peak):8.1f}deg/s^2  [{flag}]")


if __name__ == "__main__":
    main()
