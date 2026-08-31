#!/usr/bin/env python3
"""Standalone (no sim) numerical verification for
``ToppraTrajectory.retime()`` (``docs/2026-08-31_toppra_retiming_
implementation_plan.md`` 実装順序1), ahead of building
``ToppraRetimingTrajectoryTracker`` on top of it.

Two questions this answers:

1. Does repeatedly calling ``retime(sd_start=...)`` on the same fixed path
   actually update the velocity profile correctly (initial speed matches
   the requested ``sd_start``, and the path still reaches the target at
   rest for ``sd_end=0.0``)? Exercised on both a straight leg and the
   hairpin route already used elsewhere for TOPP-RA benchmarks
   (``experiment_toppra_reduced_envelope_speed.py``).
2. Is each ``retime()`` call fast enough for a 10Hz replanning tick
   (<100ms), since it must run inline in ``sample()``? Reports both the
   first call (includes any one-time warmup) and the steady-state mean of
   the following calls, across a ``sd_start`` sweep meant to mimic a
   vehicle's speed varying tick to tick as it re-approaches the same fixed
   path.

Does NOT modify toppra_trajectory.py (production) or
replanning_trajectory_tracker.py.

Usage: python3 test/experiment_toppra_retiming.py
"""
import time

import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import ToppraTrajectory
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import IDENTITY_QUAT

MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)
FORWARD = (1.0, 0.0, 0.0)

STRAIGHT_ROUTE = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
HAIRPIN_ROUTE = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.8, 0.0]]

# Mimics a vehicle re-approaching the same fixed path tick to tick: starts
# near (but below, see below) max speed, decelerates toward the target,
# revisits a mid speed -- not monotonic, since a real re-plan tick's
# sd_start (live speed) is not guaranteed monotonic either (disturbances,
# tracking error). Stays a safety margin below MAX_VEL rather than sweeping
# all the way to it: TOPP-RA's forward-pass feasibility check at s=0 has
# essentially zero numerical slack right at the velocity limit itself (an
# sd_start of MAX_VEL, or even 98% of it, comes back genuinely infeasible
# here) -- a separate, real numerical-robustness question of its own, not
# something this experiment's correctness/timing check needs to resolve.
SD_START_SWEEP = [0.0, 0.1, 0.3, 0.35, 0.25, 0.15, 0.32, 0.05, 0.0]

_TICK_BUDGET_S = 0.1  # 10Hz


def _build(waypoints):
    alloc = ThrustAllocator()
    wrench_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max)
    return ToppraTrajectory(
        waypoints, IDENTITY_QUAT.copy(),
        max_vel=MAX_VEL, mass=MASS, inertia=INERTIA,
        wrench_envelope=wrench_envelope, max_angular_rate=MAX_ANGULAR_RATE,
        forward_axis=FORWARD, face_travel=True,
    )


def _check_and_time(label, waypoints):
    print(f"=== {label}: retime() correctness + per-call timing ===")
    traj = _build(waypoints)
    target = np.asarray(waypoints[-1], dtype=float)
    durations = []
    for i, sd_start in enumerate(SD_START_SWEEP):
        t0 = time.perf_counter()
        traj.retime(sd_start=sd_start, sd_end=0.0)
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)

        _p0, v0, _a0, _q0 = traj.sample(0.0)
        p_end, v_end, _a_end, _q_end = traj.sample(traj.global_total_duration)
        speed0_ok = np.isclose(np.linalg.norm(v0), sd_start, atol=1e-6)
        end_ok = (np.allclose(p_end, target, atol=1e-2)
                  and np.allclose(v_end, [0.0, 0.0, 0.0], atol=1e-2))
        budget_ok = elapsed < _TICK_BUDGET_S
        status = "OK" if (speed0_ok and end_ok and budget_ok) else "FAIL"
        print(f"  [{i}] sd_start={sd_start:.2f}: speed0={np.linalg.norm(v0):.4f} "
              f"dur={traj.global_total_duration:.2f}s call={elapsed*1000:.1f}ms "
              f"{status}")
        assert speed0_ok, f"sample(0) speed {np.linalg.norm(v0)} != sd_start {sd_start}"
        assert end_ok, f"did not reach target at rest: p={p_end} v={v_end}"

    first, rest = durations[0], durations[1:]
    print(f"  first call: {first*1000:.1f}ms, "
          f"steady-state mean of remaining {len(rest)}: "
          f"{np.mean(rest)*1000:.1f}ms (max {np.max(rest)*1000:.1f}ms), "
          f"10Hz budget={_TICK_BUDGET_S*1000:.0f}ms")
    assert np.max(rest) < _TICK_BUDGET_S, (
        f"steady-state retime() call exceeded 10Hz budget: {np.max(rest)*1000:.1f}ms"
    )
    print()


def main():
    _check_and_time("straight", STRAIGHT_ROUTE)
    _check_and_time("hairpin", HAIRPIN_ROUTE)
    print("all checks passed")


if __name__ == "__main__":
    main()
