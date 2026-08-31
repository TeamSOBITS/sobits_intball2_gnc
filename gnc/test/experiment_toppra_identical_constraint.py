#!/usr/bin/env python3
"""Prototype: does wiring the constant wrench-envelope constraint through
toppra's existing ``identical=True`` fast path (see
``docs/2026-08-30_toppra_replanning_sd_start_speed_investigation.md``
"追記2") actually remove the ~0.9s redundant per-gridpoint F/g copy that
profiling found in ``SecondOrderConstraint.compute_constraint_params``?

This does NOT touch ``toppra_trajectory.py`` (production, ``static`` mode
depends on it) -- ``WrenchEnvelopeConstraint`` here is a standalone
``toppra.constraint.LinearConstraint`` subclass, modeled directly on
``toppra.constraint.joint_torque.JointTorqueConstraint`` (the one built-in
constraint that already uses this path), swapped from a box ``[I; -I]``
polytope to our general ``F_ENV @ w <= g_ENV`` half-space polytope.

Usage: python3 test/experiment_toppra_identical_constraint.py
"""
import time
from itertools import product

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from scipy.spatial import ConvexHull
from toppra.constraint import DiscretizationType
from toppra.constraint.linear_constraint import (
    LinearConstraint,
    canlinear_colloc_to_interpolate,
)

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    _WRENCH_DOF,
    _dense_travel_rotvecs,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import IDENTITY_QUAT
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

SAFETY_MARGIN = 0.7
MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)
FORWARD = (1.0, 0.0, 0.0)
REPLAN_BUDGET_S = 0.1
N_REPEATS = 5
_SAMPLES_PER_SEGMENT = 20

WAYPOINTS = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.8, 0.0]]


class WrenchEnvelopeConstraint(LinearConstraint):
    """``F_ENV @ w <= g_ENV`` polytope constraint on ``w = inv_dyn(q, qd,
    qdd)``, constant along the whole path -- same structure as
    ``JointTorqueConstraint.compute_constraint_params`` (box-shaped
    ``F``/``g``), generalized to an arbitrary half-space polytope and wired
    through ``identical=True`` so ``F``/``g`` are built once instead of
    copied per gridpoint (the fix this prototype is testing)."""

    def __init__(self, inv_dyn, F_env, g_env, dof,
                 discretization_scheme=DiscretizationType.Interpolation):
        super().__init__()
        self.inv_dyn = inv_dyn
        self.F_env = np.asarray(F_env, dtype=float)
        self.g_env = np.asarray(g_env, dtype=float)
        self.dof = dof
        self.set_discretization_type(discretization_scheme)
        self.identical = True
        self._format_string = "    Kind: Wrench envelope constraint (identical F/g)\n"

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path "
                "dof ({:d})".format(self.dof, path.dof)
            )
        v_zero = np.zeros(path.dof)
        p_vec = path(gridpoints)
        ps_vec = path(gridpoints, 1)
        pss_vec = path(gridpoints, 2)

        c_vec = np.array([self.inv_dyn(_p, v_zero, v_zero) for _p in p_vec])
        a_vec = np.array(
            [self.inv_dyn(_p, v_zero, _ps) for _p, _ps in zip(p_vec, ps_vec)]
        ) - c_vec
        b_vec = np.array([
            self.inv_dyn(_p, _ps, pss_)
            for _p, _ps, pss_ in zip(p_vec, ps_vec, pss_vec)
        ]) - c_vec

        if self.discretization_type == DiscretizationType.Collocation:
            return a_vec, b_vec, c_vec, self.F_env, self.g_env, None, None
        if self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                a_vec, b_vec, c_vec, self.F_env, self.g_env, None, None,
                gridpoints, identical=True,
            )
        raise NotImplementedError("Other form of discretization not supported!")


def _build_path(position_waypoints, q0):
    """Same dense-resample + rotvec construction as ``ToppraTrajectory.
    __init__`` (toppra_trajectory.py), extracted so this prototype can swap
    only the constraint, not the path."""
    position_waypoints = np.asarray(position_waypoints, dtype=float)
    distances = np.linalg.norm(np.diff(position_waypoints, axis=0), axis=1)
    segment_times = np.where(distances < 1e-9, 1.0, distances)

    pos_coeffs = HermiteSplineTrajectoryGenerator().generate(
        position_waypoints, segment_times
    )

    cum_times = np.concatenate([[0.0], np.cumsum(segment_times)])
    ss_list, p_list, v_list = [], [], []
    n_segments = len(segment_times)
    for seg in range(n_segments):
        taus = np.linspace(
            0.0, segment_times[seg], _SAMPLES_PER_SEGMENT,
            endpoint=(seg == n_segments - 1),
        )
        for tau in taus:
            ss_list.append(cum_times[seg] + tau)
            p_list.append(evaluate_vector(pos_coeffs[seg], tau, order=0))
            v_list.append(evaluate_vector(pos_coeffs[seg], tau, order=1))
    ss = np.array(ss_list)

    rotvecs = _dense_travel_rotvecs(v_list, q0, FORWARD, True)
    combined = np.concatenate([np.array(p_list), rotvecs], axis=1)
    return ta.SplineInterpolator(ss, combined)


def _time_construction(wrench_envelope, use_identical, n_repeats=N_REPEATS):
    wrench_F, wrench_g = wrench_envelope
    inertia_matrix = np.diag([MASS, MASS, MASS, INERTIA, INERTIA, INERTIA])

    def inv_dyn(_q, _qd, qdd):
        return inertia_matrix @ qdd

    times = []
    duration = None
    infeasible = 0
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        path = _build_path(WAYPOINTS, IDENTITY_QUAT.copy())
        vel_max = np.array([MAX_VEL] * 3 + [float(MAX_ANGULAR_RATE)] * 3)
        pc_vel = constraint.JointVelocityConstraint(
            np.vstack([-vel_max, vel_max]).T
        )
        if use_identical:
            pc_wrench = WrenchEnvelopeConstraint(
                inv_dyn, wrench_F, wrench_g, dof=_WRENCH_DOF
            )
        else:
            pc_wrench = constraint.SecondOrderConstraint(
                inv_dyn, lambda _q: wrench_F, lambda _q: wrench_g,
                dof=_WRENCH_DOF,
            )
        instance = algo.TOPPRA([pc_vel, pc_wrench], path,
                                parametrizer="ParametrizeConstAccel")
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            infeasible += 1
            continue
        times.append(time.perf_counter() - t0)
        duration = float(jnt_traj.duration)
    return times, duration, infeasible


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


def _report(label, n_facets, times, duration, infeasible):
    budget_ok = "OK" if times and np.max(times) <= REPLAN_BUDGET_S else "OVER"
    print(f"{label}: {n_facets} facets")
    print(f"  construct time: mean={np.mean(times):.3f}s "
          f"max={np.max(times):.3f}s (n={len(times)}, infeasible={infeasible}) "
          f"[{budget_ok} vs {REPLAN_BUDGET_S}s budget]")
    if duration is not None:
        print(f"  trajectory duration: {duration:.3f}s")


def main():
    alloc = ThrustAllocator()
    full_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max, SAFETY_MARGIN)
    n_facets_full = full_envelope[0].shape[0]

    print("=== baseline (current production path: constraint_F/g lambdas, no identical) ===")
    base_times, base_duration, base_infeasible = _time_construction(
        full_envelope, use_identical=False
    )
    _report("full envelope", n_facets_full, base_times, base_duration, base_infeasible)
    print()

    print("=== prototype (WrenchEnvelopeConstraint, identical=True) ===")
    id_times, id_duration, id_infeasible = _time_construction(
        full_envelope, use_identical=True
    )
    _report("full envelope", n_facets_full, id_times, id_duration, id_infeasible)
    if base_duration is not None and id_duration is not None:
        print(f"  duration matches baseline: "
              f"{'YES' if abs(id_duration - base_duration) < 1e-6 else 'NO'} "
              f"(diff={abs(id_duration - base_duration):.6f}s)")
    print()

    verts = _true_vertices(alloc.A, alloc.fj_max, SAFETY_MARGIN)
    for m in (32, 48, 64):
        F, g = _reduced_envelope(verts, m, seed=0)
        times, duration, infeasible = _time_construction((F, g), use_identical=True)
        _report(f"m={m} (identical=True)", F.shape[0], times, duration, infeasible)


if __name__ == "__main__":
    main()
