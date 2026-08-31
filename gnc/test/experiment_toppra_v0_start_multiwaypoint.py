#!/usr/bin/env python3
"""Follow-up to ``experiment_toppra_v0_start_tangent.py``: does routing
through additional waypoints (the mitigation named but never tried in
``docs/archive/achieved/2026-08-30_toppra_replanning_sd_start_speed_
investigation.md`` 追記5, and now available via ``ReplanningTrajectoryTracker``
's ``route_waypoints``) avoid the ``v_perp`` + ``face_travel=True``
``FailUncontrollable`` blowup found there?

Two variants, same v_perp sweep as the original 2-waypoint experiment:

- ``route``: the hairpin route already used elsewhere for TOPP-RA benchmarks
  (``experiment_toppra_reduced_envelope_speed.py``), unmodified --
  ``[[0,0,0],[3,0,0],[3,2.8,0]]``. Segment 0's end tangent (at waypoint 1) is
  now a nonzero Catmull-Rom estimate (see ``HermiteSplineTrajectoryGenerator.
  _estimate_tangents``) instead of the forced-zero end-of-path tangent the
  2-waypoint case had -- so this alone is a different boundary condition,
  not just "more waypoints after the same first segment".
- ``route_near_insert``: same route, but with an extra waypoint inserted very
  close to the start, offset along ``v0``'s direction, to directly test the
  named mitigation ("経路の自由度を増やす（中間点を挟む等）") of spreading
  the v0->target-direction reorientation over a deliberately short first
  segment rather than relying on incidental downstream geometry.

Does NOT modify toppra_trajectory.py (production) or
replanning_trajectory_tracker.py.

Usage: python3 test/experiment_toppra_v0_start_multiwaypoint.py
"""
import numpy as np

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)

from experiment_toppra_v0_start_tangent import (
    SAFETY_MARGIN,
    _build_arclength_path,
    _make_instance,
)

HAIRPIN_ROUTE = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.8, 0.0]]
_INSERT_OFFSET_M = 0.05  # short first segment, along v0's direction


def _with_near_insert(route, v0, offset_m):
    route = np.asarray(route, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v0_norm = np.linalg.norm(v0)
    direction = v0 / v0_norm if v0_norm > 1e-9 else route[1] - route[0]
    direction = direction / np.linalg.norm(direction)
    insert = route[0] + direction * offset_m
    return np.vstack([route[0], insert, route[1:]])


def _sweep(label, waypoints, wrench_envelope):
    print(f"=== {label}: v_perp sweep, face_travel on vs off ===")
    for face_travel in (False, True):
        for v_perp in (0.001, 0.01, 0.05, 0.1, 0.35):
            v0 = np.array([0.05, v_perp, 0.0])
            path, ss = _build_arclength_path(waypoints, v0, face_travel=face_travel)
            instance = _make_instance(path, wrench_envelope)
            jnt = instance.compute_trajectory(sd_start=np.linalg.norm(v0), sd_end=0.0)
            status = f"OK dur={jnt.duration:.2f}s" if jnt is not None else "INFEASIBLE"
            print(f"  face_travel={face_travel!s:5} v_perp={v_perp}: {status}")
    print()


def main():
    alloc = ThrustAllocator()
    wrench_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max, SAFETY_MARGIN)

    _sweep("route (hairpin, no insert)", HAIRPIN_ROUTE, wrench_envelope)

    print(f"=== route_near_insert: extra waypoint {_INSERT_OFFSET_M}m from start, ===")
    print("=== along v0's own direction, per v_perp (first segment direction ===")
    print("=== changes with v0, so the insert is rebuilt for each v_perp) ===")
    for face_travel in (False, True):
        for v_perp in (0.001, 0.01, 0.05, 0.1, 0.35):
            v0 = np.array([0.05, v_perp, 0.0])
            waypoints = _with_near_insert(HAIRPIN_ROUTE, v0, _INSERT_OFFSET_M)
            path, ss = _build_arclength_path(waypoints, v0, face_travel=face_travel)
            instance = _make_instance(path, wrench_envelope)
            jnt = instance.compute_trajectory(sd_start=np.linalg.norm(v0), sd_end=0.0)
            status = f"OK dur={jnt.duration:.2f}s" if jnt is not None else "INFEASIBLE"
            print(f"  face_travel={face_travel!s:5} v_perp={v_perp}: {status}")


if __name__ == "__main__":
    main()
