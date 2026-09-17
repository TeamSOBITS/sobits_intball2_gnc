#!/usr/bin/env python3
"""Offline, sim-free follow-up to
experiment_fan_second_order_lag_wrench_rate_constraint.py: that script
compared the fan-lag-consistent safe force-rate against a *synthetic*
quintic min-jerk proxy, not a real MINCO trajectory.

This script closes that gap by measuring, on an ACTUAL trajectory from the
unmodified minco_native_py, how far today's (unconstrained) plan already
exceeds the fan-lag-consistent safe rate, and gives an *estimated*
duration-increase from a simple uniform-time-dilation argument (not a real
re-optimization).

A real force_rate_limit parameter was added to minco_solver.cpp at one point
to measure the actual (re-optimized) duration and its effect in sim -- see
docs/archive/achieved/2026-09-17_fan_lag_wrench_rate_constraint_investigation.md
sections 5-6 for those numbers. The task was ultimately shelved (the sim
verification found the effect on tracking error/duty saturation was
non-monotonic in both path shape and constraint strength, i.e. not reliably
beneficial) and the production wiring was reverted; this script's estimate
below no longer has a production API to compare against directly.

Usage: python3 gnc/test/experiment_wrench_rate_constraint_duration_impact.py
"""
import numpy as np

import minco_native_py
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass
INERTIA = 0.0136  # kg*m^2, isotropic scalar currently used

# From experiment_fan_second_order_lag_wrench_rate_constraint.py: real
# coeff_a/coeff_b read live from the sim (rosparam), Ts=0.02s (50 Hz).
TAU_S = 0.19983322207648327
ERROR_TOLERANCE_FRACTIONS = (0.05, 0.10, 0.20)

NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5
RV1 = np.array([0.0, 0.35864857, 1.6255471])

SAMPLE_DT = 0.02  # s, matches the real fan control tick (Ts above)


def flatten_waypoint(pos, rot):
    return list(pos) + list(rot)


def beta_vectors(t):
    beta0 = np.array([1, t, t**2, t**3, t**4, t**5])
    beta1 = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
    beta2 = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])
    return beta0, beta1, beta2


def plan(waypoints):
    success, error_code, segment_times, coeffs_flat, duration = minco_native_py.plan_minco(
        waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert success and error_code == 0, "unmodified solver didn't converge"
    K = len(segment_times)
    coeffs = np.array(coeffs_flat).reshape(K, 6, 6)
    return segment_times, coeffs, duration


def sample_wrench_and_fan_forces(segment_times, coeffs, alloc):
    """Returns (t_all, fan_force_all) sampled at SAMPLE_DT across the whole
    trajectory, using the REAL ThrustAllocator on the REAL planned wrench."""
    t_all, fan_force_all = [], []
    t_offset = 0.0
    n_segs = len(segment_times)
    for seg, T_seg in enumerate(segment_times):
        cPos = coeffs[seg, 0:3, :]
        cRot = coeffs[seg, 3:6, :]
        n = max(2, int(round(T_seg / SAMPLE_DT)))
        # Drop the boundary sample except on the final segment, so it isn't
        # duplicated at dt=0 against the next segment's t=0 sample.
        ks = range(n) if seg == n_segs - 1 else range(n - 1)
        for k in ks:
            t = k * T_seg / (n - 1)
            _, _, beta2 = beta_vectors(t)
            accPos = cPos @ beta2
            r_ddot = cRot @ beta2
            force = MASS * accPos
            torque = INERTIA * r_ddot
            duties = np.array(alloc.allocate(force, torque))
            fan_force = (duties / alloc.kj) ** 2
            t_all.append(t_offset + t)
            fan_force_all.append(fan_force)
        t_offset += T_seg
    return np.array(t_all), np.array(fan_force_all)


def peak_fan_force_rate(t_all, fan_force_all):
    """Central-difference d(force)/dt per fan, peak magnitude over all fans
    and all samples."""
    dt = np.diff(t_all)
    rates = np.diff(fan_force_all, axis=0) / dt[:, None]
    return float(np.max(np.abs(rates)))


def analyze_current_plan(label, waypoints):
    print(f"\n=== {label} ===")
    segment_times, coeffs, duration = plan(waypoints)
    print(f"plan_minco (unconstrained): K={len(segment_times)} segments, "
          f"duration={duration:.4f}s")

    alloc = ThrustAllocator()
    t_all, fan_force_all = sample_wrench_and_fan_forces(segment_times, coeffs, alloc)
    peak_rate = peak_fan_force_rate(t_all, fan_force_all)
    print(f"peak per-fan force-rate actually demanded by this (unconstrained) "
          f"plan: {peak_rate:.4f} N/s")

    print(f"{'tol frac':>9} {'safe rate (N/s)':>16} {'lambda (time-stretch)':>22} "
          f"{'ESTIMATED duration (s)':>24}")
    for frac in ERROR_TOLERANCE_FRACTIONS:
        safe_rate = frac * alloc.fj_max / TAU_S
        lam = (peak_rate / safe_rate) ** (1.0 / 3.0)
        print(f"{frac:9.2f} {safe_rate:16.4f} {lam:22.2f} "
              f"{duration * max(lam, 1.0):24.2f}")


def main():
    midpoint = 0.5 * (NEAR_DOCK + ABOVE_DOCK)
    bulge = NAV_ENTRY - midpoint
    q1_given = midpoint + BULGE_SCALE * bulge

    scenarios = {
        "regression-test scenario (near_dock -> bulge via -> above_dock, "
        "shared rotation RV1)":
            flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
            + flatten_waypoint(q1_given, RV1)
            + flatten_waypoint(ABOVE_DOCK, RV1),
        "genuine axis-change scenario (via=1.2rad about x, end=1.2rad about z)":
            flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
            + flatten_waypoint(q1_given, [1.2, 0.0, 0.0])
            + flatten_waypoint(ABOVE_DOCK, [0.0, 0.0, 1.2]),
        "short pure-translation move, no via/rotation (near_dock -> "
        "above_dock direct, 0.88m) -- faster/snappier than the two above, "
        "which both carry a large (~95-137deg) rotation that dominates "
        "duration":
            flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
            + flatten_waypoint(ABOVE_DOCK, [0.0, 0.0, 0.0]),
    }

    for label, wp in scenarios.items():
        analyze_current_plan(label, wp)


if __name__ == "__main__":
    main()
