#!/usr/bin/env python3
"""Investigation: can ``ToppraTrajectory`` express an arbitrary start
velocity ``v0`` (including a component perpendicular to the target
direction, "v_perp") by forcing the Hermite path's start tangent to ``v0``
(the same override ``HermiteSplineTrajectoryGenerator(..., v0=v0)`` already
gives ``ReplanningTrajectoryTracker``'s non-TOPP-RA path) and setting
``toppra``'s scalar ``sd_start`` accordingly? See ``docs/
2026-08-30_toppra_replanning_sd_start_speed_investigation.md`` 追記4/5 for
the full writeup -- summary of what this file demonstrates:

1. The production ``s`` parameter (``ToppraTrajectory`` reuses the Hermite
   spline's local ``tau``, cumulative waypoint-distance) is NOT true arc
   length, so ``sd`` there does not equal real m/s -- ``sd_start=1.0`` is
   "uncontrollable" even for ``v0=0`` (the plain ``static`` case). Fixed
   here by re-parameterizing ``s`` as the true cumulative Euclidean arc
   length of the dense position samples.
2. With that fix, ``v0`` parallel to the path direction works exactly:
   ``sd_start=norm(v0)`` reproduces ``v(0)=v0`` and stays feasible.
3. Any ``v_perp`` (even 0.001 m/s) makes it infeasible once ``face_travel``
   (attitude tracks the position path's tangent) is on, but a much larger
   ``v_perp`` stays feasible with ``face_travel=False`` (translation-only).
4. The real cause: a single 2-waypoint cubic Hermite segment with
   ``m0=v0`` (small, off-target-direction) and ``m1=0`` produces a speed
   profile that is small near the start (before ramping up mid-segment) --
   so the large direction change from "facing v0" to "facing the target"
   gets compressed into a vanishingly small arc-length window, i.e.
   curvature kappa=dtheta/ds blows up near s=0. This is independent of the
   ``_dense_travel_rotvecs`` "sample 0 stays at q0" convention (also tested
   below, ``sample0_from_v0=True`` does not help) -- it is a structural
   property of fitting a low-order, low-DOF path through a velocity
   boundary condition whose direction disagrees with the travel direction.

Does NOT modify toppra_trajectory.py (production).

Usage: python3 test/experiment_toppra_v0_start_tangent.py
"""
import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import _WRENCH_DOF
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    IDENTITY_QUAT,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

SAFETY_MARGIN = 0.7
MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)
FORWARD = (1.0, 0.0, 0.0)
_DEGENERATE_TANGENT_THRESHOLD = 1e-9
_DENSE_SAMPLES = 400  # dense enough for an accurate arc-length reparam


def _dense_travel_rotvecs(v_list, q0, forward_axis, face_travel, sample0_from_v0):
    """Same as toppra_trajectory._dense_travel_rotvecs, plus an experimental
    ``sample0_from_v0`` flag (tested and found NOT to fix the v_perp issue,
    see module docstring point 4)."""
    n = len(v_list)
    q0 = np.asarray(q0, dtype=float)
    rotvecs = np.zeros((n, 3))
    if sample0_from_v0 and face_travel:
        q_prev = compute_q_des(v_list[0], q0, _DEGENERATE_TANGENT_THRESHOLD, forward_axis)
        rotvecs[0] = quat_log(quat_mul(quat_conj(q0), q_prev))
    else:
        q_prev = q0
    for i in range(1, n):
        if face_travel:
            q_prev = compute_q_des(v_list[i], q_prev, _DEGENERATE_TANGENT_THRESHOLD, forward_axis)
        rotvecs[i] = quat_log(quat_mul(quat_conj(q0), q_prev))
    return rotvecs


def _build_arclength_path(waypoints, v0, face_travel=True, sample0_from_v0=False):
    """Same Hermite-fit-then-dense-resample construction as ``ToppraTrajectory.
    __init__``, except ``s`` is re-parameterized as true cumulative Euclidean
    arc length (point 1 above) instead of reusing the Hermite ``tau``."""
    waypoints = np.asarray(waypoints, dtype=float)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.where(distances < 1e-9, 1.0, distances)

    coeffs = HermiteSplineTrajectoryGenerator().generate(
        waypoints, segment_times, v0=v0
    )

    p_list, v_list = [], []
    n_segments = len(segment_times)
    for seg in range(n_segments):
        taus = np.linspace(
            0.0, segment_times[seg], _DENSE_SAMPLES, endpoint=(seg == n_segments - 1)
        )
        for tau in taus:
            p_list.append(evaluate_vector(coeffs[seg], tau, order=0))
            v_list.append(evaluate_vector(coeffs[seg], tau, order=1))
    p_arr = np.array(p_list)
    seglens = np.linalg.norm(np.diff(p_arr, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seglens)])

    rotvecs = _dense_travel_rotvecs(
        v_list, IDENTITY_QUAT.copy(), FORWARD, face_travel, sample0_from_v0
    )
    combined = np.concatenate([p_arr, rotvecs], axis=1)
    return ta.SplineInterpolator(ss, combined), ss


def _make_instance(path, wrench_envelope):
    wrench_F, wrench_g = wrench_envelope
    inertia_matrix = np.diag([MASS, MASS, MASS, INERTIA, INERTIA, INERTIA])

    def inv_dyn(_q, _qd, qdd):
        return inertia_matrix @ qdd

    vel_max = np.array([MAX_VEL] * 3 + [float(MAX_ANGULAR_RATE)] * 3)
    pc_vel = constraint.JointVelocityConstraint(np.vstack([-vel_max, vel_max]).T)
    pc_wrench = constraint.SecondOrderConstraint(
        inv_dyn, lambda _q: wrench_F, lambda _q: wrench_g, dof=_WRENCH_DOF
    )
    return algo.TOPPRA([pc_vel, pc_wrench], path, parametrizer="ParametrizeConstAccel")


def main():
    alloc = ThrustAllocator()
    wrench_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max, SAFETY_MARGIN)
    waypoints = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]

    print("=== check 1: is the production `s` (Hermite tau, not true arc length) ===")
    print("=== usable as a real-speed sd_start=1 trick? (expected: NO) ===")
    from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
        _dense_travel_rotvecs as _prod_rotvecs,
    )
    pw = np.asarray(waypoints, dtype=float)
    d = np.linalg.norm(np.diff(pw, axis=0), axis=1)
    st = np.where(d < 1e-9, 1.0, d)
    coeffs = HermiteSplineTrajectoryGenerator().generate(pw, st, v0=np.zeros(3))
    ss_list, p_list, v_list = [], [], []
    cum = np.concatenate([[0.0], np.cumsum(st)])
    for seg in range(len(st)):
        taus = np.linspace(0.0, st[seg], 20, endpoint=(seg == len(st) - 1))
        for tau in taus:
            ss_list.append(cum[seg] + tau)
            p_list.append(evaluate_vector(coeffs[seg], tau, order=0))
            v_list.append(evaluate_vector(coeffs[seg], tau, order=1))
    rotvecs = _prod_rotvecs(v_list, IDENTITY_QUAT.copy(), FORWARD, True)
    combined = np.concatenate([np.array(p_list), rotvecs], axis=1)
    prod_path = ta.SplineInterpolator(np.array(ss_list), combined)
    instance = _make_instance(prod_path, wrench_envelope)
    jnt = instance.compute_trajectory(sd_start=1.0)
    print(f"  v0=0, sd_start=1.0 (production s): "
          f"{'feasible' if jnt is not None else 'INFEASIBLE (' + str(instance.problem_data.return_code) + ')'}")
    print()

    print("=== check 2: true arc-length s, v0 parallel to target direction ===")
    v0_parallel = np.array([0.3, 0.0, 0.0])
    path, ss = _build_arclength_path(waypoints, v0_parallel)
    instance = _make_instance(path, wrench_envelope)
    jnt = instance.compute_trajectory(sd_start=np.linalg.norm(v0_parallel), sd_end=0.0)
    if jnt is not None:
        v_at_0 = jnt(0.0, 1)[:3]
        print(f"  feasible, duration={jnt.duration:.2f}s, "
              f"v(0)={v_at_0}, matches v0: {np.allclose(v_at_0, v0_parallel)}")
    else:
        print(f"  INFEASIBLE ({instance.problem_data.return_code})")
    print()

    print("=== check 3: true arc-length s, v_perp sweep, face_travel on vs off ===")
    for face_travel in (False, True):
        for v_perp in (0.001, 0.01, 0.05, 0.1, 0.35):
            v0 = np.array([0.05, v_perp, 0.0])
            path, ss = _build_arclength_path(waypoints, v0, face_travel=face_travel)
            instance = _make_instance(path, wrench_envelope)
            jnt = instance.compute_trajectory(sd_start=np.linalg.norm(v0), sd_end=0.0)
            status = f"OK dur={jnt.duration:.2f}s" if jnt is not None else "INFEASIBLE"
            print(f"  face_travel={face_travel!s:5} v_perp={v_perp}: {status}")
    print()

    print("=== check 4: does fixing sample-0's attitude (v0-based, not q0-forced) help? ===")
    print("=== (expected: NO -- curvature blowup is in v_list itself, not the rotvec convention) ===")
    for sample0_from_v0 in (False, True):
        v0 = np.array([0.05, 0.05, 0.0])
        path, ss = _build_arclength_path(
            waypoints, v0, face_travel=True, sample0_from_v0=sample0_from_v0
        )
        instance = _make_instance(path, wrench_envelope)
        jnt = instance.compute_trajectory(sd_start=np.linalg.norm(v0), sd_end=0.0)
        status = f"OK dur={jnt.duration:.2f}s" if jnt is not None else "INFEASIBLE"
        print(f"  sample0_from_v0={sample0_from_v0}: {status}")


if __name__ == "__main__":
    main()
