#!/usr/bin/env python3
"""Follow-up to ``experiment_toppra_retiming.py``: that experiment found
``ToppraTrajectory.retime(sd_start, ...)`` cannot represent "the vehicle's
live speed at its current position" (``s_current``) at all -- toppra's
``compute_trajectory(sd_start=...)`` is always a boundary condition at the
path's absolute start (``s=0``), which after the very first tick is a
location the vehicle has already left. It also found that any route with
even a gentle corner ahead becomes infeasible above roughly 0.03-0.1 m/s of
``sd_start`` at ``s=0`` -- a separate, corner-curvature-driven limit
(``docs/2026-08-31_toppra_retiming_implementation_plan.md``'s underlying
assumption did not anticipate either finding).

This experiment checks the natural follow-up design: instead of reusing the
same fixed ``SplineInterpolator``/``TOPPRA`` instance and only re-running
``compute_trajectory`` (decided-fixed, cheap, per that doc), rebuild a
**sub-path** every tick -- trim the same fixed geometric curve down to just
"from wherever the vehicle actually is (``s_current``) to the target", so
that ``sd_start`` at the sub-path's own ``s=0`` genuinely means "the
vehicle's real speed, right here, right now". This keeps the path's *shape*
fixed (no route-waypoint regeneration, no MINCO) but drops the "retime is
just ``compute_trajectory``, ~1ms" premise -- rebuilding the path AND the
``WrenchEnvelopeConstraint``/``TOPPRA`` instance is exactly the ~50-200ms-
scale cost ``docs/archive/achieved/
2026-08-30_toppra_replanning_sd_start_speed_investigation.md`` 追記2〜3
measured and optimized (``identical=True``) for the one-shot ``static``
build. The question here is whether that same cost, paid *every tick*
instead of once per goal, still fits the 10Hz (100ms) replanning budget.

Does NOT modify toppra_trajectory.py (production) or
replanning_trajectory_tracker.py.

Usage: python3 test/experiment_toppra_subpath_retiming.py
"""
import time

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    _SAMPLES_PER_SEGMENT,
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
from sobits_intball2_gnc.guidance.utils.wrench_envelope_constraint import (
    WrenchEnvelopeConstraint,
)

MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)
FORWARD = (1.0, 0.0, 0.0)

STRAIGHT_ROUTE = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
HAIRPIN_ROUTE = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.8, 0.0]]

_TICK_BUDGET_S = 0.1  # 10Hz


def _build_full_path(waypoints, q0):
    """Same dense-resample + rotvec construction ToppraTrajectory.__init__
    uses -- returns (ss, combined) for the *whole* fixed geometric path."""
    waypoints = np.asarray(waypoints, dtype=float)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.where(distances < 1e-9, 1.0, distances)
    pos_coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)

    p_list, v_list = [], []
    n_segments = len(segment_times)
    for seg in range(n_segments):
        taus = np.linspace(
            0.0, segment_times[seg], _SAMPLES_PER_SEGMENT,
            endpoint=(seg == n_segments - 1),
        )
        for tau in taus:
            p_list.append(evaluate_vector(pos_coeffs[seg], tau, order=0))
            v_list.append(evaluate_vector(pos_coeffs[seg], tau, order=1))
    p_arr = np.array(p_list)
    seglens = np.linalg.norm(np.diff(p_arr, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seglens)])
    rotvecs = _dense_travel_rotvecs(v_list, q0, FORWARD, True)
    combined = np.concatenate([p_arr, rotvecs], axis=1)
    return ss, combined


def _make_instance(path, wrench_envelope):
    vel_max = np.array([MAX_VEL] * 3 + [float(MAX_ANGULAR_RATE)] * 3)
    pc_vel = constraint.JointVelocityConstraint(np.vstack([-vel_max, vel_max]).T)
    inertia_matrix = np.diag([MASS, MASS, MASS, INERTIA, INERTIA, INERTIA])
    wrench_F, wrench_g = wrench_envelope

    def inv_dyn(_q, _qd, qdd):
        return inertia_matrix @ qdd

    pc_wrench = WrenchEnvelopeConstraint(inv_dyn, wrench_F, wrench_g, dof=_WRENCH_DOF)
    return algo.TOPPRA([pc_vel, pc_wrench], path, parametrizer="ParametrizeConstAccel")


def _build_subpath_and_retime(full_ss, full_combined, s_current, sd_start,
                               wrench_envelope):
    """Trim the fixed full path down to [s_current, s_end], re-anchor its
    arc-length parameter to start at 0, and rebuild a fresh
    SplineInterpolator + TOPPRA instance + compute_trajectory -- the full
    per-tick cost the sub-path design would actually pay."""
    keep = full_ss >= s_current
    sub_ss = full_ss[keep] - s_current
    sub_combined = full_combined[keep]
    if len(sub_ss) < 2 or sub_ss[0] > 1e-9:
        # Splice in the exact interpolated state at s_current itself (rare:
        # only when s_current doesn't land exactly on a dense sample).
        idx = np.searchsorted(full_ss, s_current)
        sub_ss = np.concatenate([[0.0], full_ss[idx:] - s_current])
        sub_combined = np.concatenate(
            [full_combined[idx - 1:idx], full_combined[idx:]], axis=0
        )
    sub_path = ta.SplineInterpolator(sub_ss, sub_combined)
    instance = _make_instance(sub_path, wrench_envelope)
    return instance.compute_trajectory(sd_start=sd_start, sd_end=0.0)


def _sweep(label, waypoints):
    print(f"=== {label}: sub-path rebuild+retime per-tick timing ===")
    alloc = ThrustAllocator()
    wrench_envelope = wrench_envelope_halfspaces(alloc.A, alloc.fj_max)
    q0 = IDENTITY_QUAT.copy()
    full_ss, full_combined = _build_full_path(waypoints, q0)
    path_length = full_ss[-1]

    # s_current sweep: progresses forward along the fixed path, mimicking a
    # vehicle advancing tick to tick (never backward, per the "_next_idx
    # forward-only" convention this mirrors).
    s_points = np.linspace(0.0, path_length * 0.9, 6)
    durations = []
    for i, s_current in enumerate(s_points):
        sd_start = min(0.1, MAX_VEL * 0.3)  # modest live speed, well inside the sd_start=0 feasibility margin found earlier
        t0 = time.perf_counter()
        jnt = _build_subpath_and_retime(
            full_ss, full_combined, s_current, sd_start, wrench_envelope
        )
        elapsed = time.perf_counter() - t0
        durations.append(elapsed)
        status = f"OK dur={jnt.duration:.2f}s" if jnt is not None else "INFEASIBLE"
        budget_flag = "OK" if elapsed < _TICK_BUDGET_S else "OVER"
        print(f"  [{i}] s_current={s_current:.2f}/{path_length:.2f}: "
              f"{status} call={elapsed*1000:.1f}ms ({budget_flag})")

    print(f"  mean={np.mean(durations)*1000:.1f}ms max={np.max(durations)*1000:.1f}ms "
          f"10Hz budget={_TICK_BUDGET_S*1000:.0f}ms")
    print()
    return durations


def main():
    straight = _sweep("straight", STRAIGHT_ROUTE)
    hairpin = _sweep("hairpin", HAIRPIN_ROUTE)
    over_budget = [d for d in straight + hairpin if d >= _TICK_BUDGET_S]
    if over_budget:
        print(f"RESULT: {len(over_budget)} call(s) exceeded the 10Hz budget -- "
              f"sub-path rebuild-per-tick is NOT viable as-is.")
    else:
        print("RESULT: all calls fit the 10Hz budget.")


if __name__ == "__main__":
    main()
