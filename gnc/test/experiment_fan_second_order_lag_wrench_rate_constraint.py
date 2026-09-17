#!/usr/bin/env python3
"""Verification-only check for [G] "ファンの二次遅れをMINCOのレンチ変化率制約に
反映" (docs/issue.md). No production code is touched -- this only measures
whether the idea would matter, using the real fan-response filter coefficients
read live from the JAXA sim (ROS1 side, via the bridge host) with:

    source /opt/ros/noetic/setup.bash
    rosparam get /thr_parameter/filter/coeff_a/a0   # 1.0
    rosparam get /thr_parameter/filter/coeff_a/a1   # -19/21
    rosparam get /thr_parameter/filter/coeff_a/a2   # 0.0
    rosparam get /thr_parameter/filter/coeff_b/b0   # 1/21
    rosparam get /thr_parameter/filter/coeff_b/b1   # 1/21
    rosparam get /thr_parameter/filter/coeff_b/b2   # 0.0
    rosparam get /prop/pub_fan_status_duration      # 0.02 (50 Hz)

a2 == b2 == 0, so despite the name ("biQuadFilter") this is a first-order
discrete IIR, not a true second-order one. Difference equation
(a0 == 1 normalizes it):

    y[n] = -a1*y[n-1] + b0*u[n] + b1*u[n-1]

which is the bilinear-transform discretization of a first-order lag with
continuous time constant tau = -Ts / ln(-a1). See MEASURED_TAU_S below.

Question this answers: given that per-fan thrust response lags a commanded
step by tau ~= 0.2s, does MINCO's *current* (unconstrained) wrench-rate
already imply per-fan force-rate changes fast enough for that lag to matter?
If yes, capping the wrench rate of change at the value derived here would be
a real, physically-grounded constraint worth adding to minco_solver.cpp. If
typical segments stay well under it, the constraint would rarely bind and is
lower priority than the other [G] items.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 gnc/test/experiment_fan_second_order_lag_wrench_rate_constraint.py
"""
import math

import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

# Measured live from the sim (see module docstring), not guessed.
COEFF_A = (1.0, -19.0 / 21.0, 0.0)
COEFF_B = (1.0 / 21.0, 1.0 / 21.0, 0.0)
TS = 0.02  # s, /prop/pub_fan_status_duration (50 Hz)

MEASURED_TAU_S = -TS / math.log(-COEFF_A[1])

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass

# Fraction of a fan's full-scale thrust (fj_max) we're willing to tolerate as
# steady-state per-fan tracking error while ramping at the derived max rate.
# This is a design choice, not a measured value -- swept below instead of
# hardcoded to one number, since the "right" tolerance is itself undecided.
ERROR_TOLERANCE_FRACTIONS = (0.05, 0.10, 0.20)


class FanLagFilter:
    """Discrete IIR reproducing the sim's per-fan thrust response exactly
    (same coefficients, same tick rate). Single scalar input/output."""

    def __init__(self, a=COEFF_A, b=COEFF_B):
        self.a0, self.a1, self.a2 = a
        self.b0, self.b1, self.b2 = b
        self.u1 = self.u2 = 0.0
        self.y1 = self.y2 = 0.0

    def step(self, u):
        y = (
            self.b0 * u + self.b1 * self.u1 + self.b2 * self.u2
            - self.a1 * self.y1 - self.a2 * self.y2
        ) / self.a0
        self.u2, self.u1 = self.u1, u
        self.y2, self.y1 = self.y1, y
        return y


def simulate(commanded, ts=TS):
    """Run a per-fan commanded-force sequence through FanLagFilter."""
    f = FanLagFilter()
    return np.array([f.step(u) for u in commanded])


def step_response_tracking_error(target, n_ticks=400, ts=TS):
    """Baseline: MINCO today places no bound on wrench rate, so in the limit
    it could ask for an instantaneous step. Returns (peak_error, actual)."""
    commanded = np.full(n_ticks, target)
    commanded[0] = 0.0  # step applied at tick 1
    actual = simulate(commanded, ts)
    err = target - actual
    return float(np.max(np.abs(err))), actual


def ramp_tracking_error(target, rate, ts=TS):
    """Ramp commanded force to `target` at `rate` [N/s], then hold. Returns
    (steady_state_error, ramp_duration_s, actual)."""
    ramp_duration = target / rate
    n_ramp = max(1, int(round(ramp_duration / ts)))
    n_hold = int(round(2 * MEASURED_TAU_S / ts)) + 10
    ramp = np.linspace(0.0, target, n_ramp + 1)
    commanded = np.concatenate([ramp, np.full(n_hold, target)])
    actual = simulate(commanded, ts)
    steady_err = float(target - actual[n_ramp])
    return steady_err, ramp_duration, actual


def quintic_min_jerk_peak_force_rate(distance, duration, mass=MASS):
    """Peak |dF/dt| = mass * peak|jerk| for a rest-to-rest quintic min-jerk
    1D move of `distance` over `duration`. Peak jerk of this profile occurs
    at the endpoints: jerk(0) = jerk(T) = 60*distance/duration**3."""
    peak_jerk = 60.0 * distance / duration ** 3
    return mass * peak_jerk


def main():
    print(f"Measured tau (first-order fan lag) = {MEASURED_TAU_S:.4f} s "
          f"(Ts={TS}s, pole={-COEFF_A[1]:.6f})")

    alloc = ThrustAllocator()
    fj_max = alloc.fj_max
    print(f"fj_max = {fj_max} N per fan\n")

    print("--- Per-fan step vs. rate-limited-ramp tracking, at fj_max ---")
    peak_step_err, _ = step_response_tracking_error(fj_max)
    print(f"Unconstrained step to fj_max: peak per-fan tracking error = "
          f"{peak_step_err:.4f} N ({100*peak_step_err/fj_max:.1f}% of fj_max) "
          f"immediately after the step.\n")

    print(f"{'tol frac':>9} {'tol (N)':>9} {'rate limit (N/s)':>18} "
          f"{'ramp time to fj_max (s)':>24} {'actual steady err (N)':>22}")
    for frac in ERROR_TOLERANCE_FRACTIONS:
        tol = frac * fj_max
        rate_limit = tol / MEASURED_TAU_S  # r such that r*tau == tol
        steady_err, ramp_t, _ = ramp_tracking_error(fj_max, rate_limit)
        print(f"{frac:9.2f} {tol:9.4f} {rate_limit:18.4f} {ramp_t:24.2f} "
              f"{steady_err:22.4f}")

    print("\n--- Does a typical (unconstrained) MINCO segment already "
          "exceed that rate? ---")
    print("(quintic min-jerk 1D rest-to-rest move, peak |dF/dt| at segment "
          "boundaries)")
    print(f"{'distance (m)':>13} {'duration (s)':>13} "
          f"{'peak |dF/dt| (N/s)':>19}")
    scenarios = [
        (0.88, 8.0),
        (1.0, 5.0),
        (1.0, 3.0),
        (4.7, 12.0),
        (4.7, 6.0),
    ]
    for d, T in scenarios:
        peak_rate = quintic_min_jerk_peak_force_rate(d, T)
        print(f"{d:13.2f} {T:13.2f} {peak_rate:19.4f}")

    print("\nCompare the peak |dF/dt| column above against the 'rate limit "
          "(N/s)' column: if typical segments already sit far above every "
          "tolerance row's rate limit, the fan lag is currently the binding "
          "constraint MINCO ignores, and adding the cap would matter.")


if __name__ == "__main__":
    main()
