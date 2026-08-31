#!/usr/bin/env python3
"""Benchmark: does the wrench-envelope facet-reduction trick (already proven
for MINCO in test/experiment_minco_native/gen_reduced_envelope.py) also speed
up TOPP-RA's ``ToppraTrajectory`` construction enough to fit the
``replanning`` 10Hz budget (0.1s)? See docs/2026-08-30_toppra_replanning_
sd_start_speed_investigation.md for context/results.

Usage: python3 test/experiment_toppra_reduced_envelope_speed.py
"""
import time
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import IDENTITY_QUAT
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
)

SAFETY_MARGIN = 0.7  # matches guidance.wrench_envelope_safety_margin default
MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)
FORWARD = (1.0, 0.0, 0.0)
REPLAN_BUDGET_S = 0.1
N_REPEATS = 5

# Sharp 3-waypoint turn, same shape as main_plan.md's validated 143.99deg
# hairpin case (docs/main_plan.md "90°超waypointでの分離型機動").
WAYPOINTS = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.8, 0.0]]


def _true_vertices(A, fj_max, margin):
    n = A.shape[1]
    corners = np.array(list(product([0.0, float(fj_max)], repeat=n)))
    return margin * (corners @ A.T)


def _farthest_point_sample(points, m, seed):
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    chosen = [int(rng.integers(n))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < m:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def _reduced_envelope(verts, m, seed):
    idx = np.unique(_farthest_point_sample(verts, m, seed))
    hull = ConvexHull(verts[idx])
    F = hull.equations[:, :-1]
    g = -hull.equations[:, -1]
    return F, g


def _time_construction(wrench_envelope, n_repeats=N_REPEATS):
    times = []
    duration = None
    infeasible = 0
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        try:
            traj = ToppraTrajectory(
                WAYPOINTS, IDENTITY_QUAT.copy(),
                max_vel=MAX_VEL, mass=MASS, inertia=INERTIA,
                wrench_envelope=wrench_envelope,
                max_angular_rate=MAX_ANGULAR_RATE,
                forward_axis=FORWARD,
            )
        except TrajectoryInfeasibleError:
            infeasible += 1
            continue
        times.append(time.perf_counter() - t0)
        duration = traj.global_total_duration
    return times, duration, infeasible


def main():
    alloc = ThrustAllocator()
    full_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max, SAFETY_MARGIN)
    n_facets_full = full_envelope[0].shape[0]

    full_times, full_duration, full_infeasible = _time_construction(full_envelope)
    print(f"full envelope: {n_facets_full} facets")
    print(f"  construct time: mean={np.mean(full_times):.3f}s "
          f"max={np.max(full_times):.3f}s (n={len(full_times)}, "
          f"infeasible={full_infeasible})")
    print(f"  trajectory duration: {full_duration:.3f}s")
    print()

    verts = _true_vertices(alloc.A, alloc.fj_max, SAFETY_MARGIN)
    for m in (32, 48, 64):
        for seed in (0, 1, 2):
            F, g = _reduced_envelope(verts, m, seed)
            times, duration, infeasible = _time_construction((F, g))
            budget_ok = "OK" if times and np.max(times) <= REPLAN_BUDGET_S else "OVER"
            conservatism = (
                f"{(duration / full_duration - 1.0) * 100:+.1f}%"
                if duration is not None else "n/a"
            )
            print(f"m={m} seed={seed}: {F.shape[0]} facets, "
                  f"mean={np.mean(times) if times else float('nan'):.3f}s "
                  f"max={np.max(times) if times else float('nan'):.3f}s "
                  f"[{budget_ok} vs {REPLAN_BUDGET_S}s budget], "
                  f"duration {conservatism}, infeasible={infeasible}")


if __name__ == "__main__":
    main()
