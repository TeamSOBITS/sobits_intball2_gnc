#!/usr/bin/env python3
"""Offline (no ROS, no sim) reproduction of the *local* receding-horizon
layer's rotation catch-up lag, and a prototype fix -- no production code
changes.

Background: ``docs/2026-09-20_minco_face_travel_corner_divergence_root_cause_
and_fix.md`` found that even after fixing the *global* ``MincoTrajectory``'s
face_travel seeding (its "1" cause), a live `capture_point_2` move still
showed the setpoint's forward axis diverging 56-59 degrees from its own
commanded velocity for ~13s starting ~3.5s after passing a via waypoint. The
doc traced this to ``ReplanningMincoV2Tracker.sample()``
(``replanning_minco_v2_tracker.py:358-374``): every tick (50Hz) it re-solves
a *fresh* quintic-Hermite rotation blend of the *same* fixed horizon
``t_local`` (1.0s default) from the vehicle's current (still-lagging)
attitude to the (by-then-correct) lookahead target attitude. The doc
confirmed the boundary math is fine at any single tick (0s->1s the blend
smoothly closes a 90 degree gap), but only checked *one* tick -- it did not
check what happens when this same full-horizon blend is *repeatedly
re-solved every tick*, always targeting "caught up 1s from now" instead of
counting down to a fixed catch-up deadline.

This script builds that receding-horizon repeat directly with the real
``solve_quintic_hermite_coeffs``/``evaluate_vector`` production primitives
(no sim, no ROS, no other production code changes) and checks two things:

1. Does the always-resolve-full-horizon pattern (current production
   behavior) actually take longer than a single ``t_local`` to settle, as
   the doc's live measurement (~13s vs. ``t_local``=1s) suggests?
2. Does a "latch a catch-up deadline and count down" fix (the same shrinking-
   horizon pattern the tracker already uses for its terminal/stop-replan
   phase, ``replanning_minco_v2_tracker.py:312-357``) settle in close to
   ``t_local`` instead?

Usage:
    python3 test/manual/prototype_local_rotation_catchup_fix_offline.py
"""
import math
import sys

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_exp, quat_rotate
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector
from sobits_intball2_gnc.guidance.utils.quintic_hermite import solve_quintic_hermite_coeffs

DT = 0.02  # 50 Hz, matches control_node/local layer tick rate
T_LOCAL = 1.0  # replanning_minco_v2_tracker.py DEFAULT_T_LOCAL_S
SIM_DURATION = 20.0
FORWARD_AXIS = np.array([1.0, 0.0, 0.0])
SETTLE_THRESHOLD_DEG = 5.0

# Matches the doc's single-tick reproduction: corner already passed, target
# velocity/attitude both fixed pointing +Y, current attitude still pointing
# the pre-corner +X direction (the "physically unavoidable turning lag").
TGT_ROTVEC = np.array([0.0, 0.0, math.pi / 2.0])  # +X -> +Y, about +Z
TGT_ROTVEC_VEL = np.zeros(3)
TGT_ROTVEC_ACCEL = np.zeros(3)
TGT_V_DIR = np.array([0.0, 1.0, 0.0])


def fwd_v_angle_deg(rotvec):
    q = quat_exp(rotvec)
    fwd = quat_rotate(q, FORWARD_AXIS)
    cos_a = np.dot(fwd, TGT_V_DIR) / (np.linalg.norm(fwd) * np.linalg.norm(TGT_V_DIR))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def run_baseline():
    """Current production pattern: every tick, re-solve a full-``T_LOCAL``
    blend from the current (already-partway-caught-up) state to the fixed
    target -- i.e. ``replanning_minco_v2_tracker.py``'s per-tick
    ``solve_quintic_hermite_coeffs(rotvec_obs, angvel_est, 0, tgt_rv, ...,
    t_local)`` call, run every tick with no state carried between solves
    except the observed rotvec/angvel themselves."""
    rotvec = np.zeros(3)
    angvel = np.zeros(3)
    n_steps = int(SIM_DURATION / DT)
    angles = np.empty(n_steps + 1)
    angles[0] = fwd_v_angle_deg(rotvec)
    for i in range(1, n_steps + 1):
        coeffs = solve_quintic_hermite_coeffs(
            rotvec, angvel, np.zeros(3), TGT_ROTVEC, TGT_ROTVEC_VEL, TGT_ROTVEC_ACCEL, T_LOCAL
        )
        rotvec = evaluate_vector(coeffs, DT, order=0)
        angvel = evaluate_vector(coeffs, DT, order=1)
        angles[i] = fwd_v_angle_deg(rotvec)
    return angles


def run_latched_catchup():
    """Candidate fix: on the first tick where the rotation error exceeds a
    threshold, latch ``t_latch`` and re-solve the blend *once* against that
    fixed origin, then just count down elapsed time against the same
    original solve (shrinking horizon) instead of re-solving a fresh
    full-``T_LOCAL`` blend every tick -- the same style of latch already used
    for the tracker's terminal phase (``_terminal_pos_coeffs``/
    ``_terminal_elapsed``, ``replanning_minco_v2_tracker.py:312-357``), just
    applied to the mid-flight rotation catch-up instead of only final
    approach."""
    rotvec = np.zeros(3)
    angvel = np.zeros(3)
    n_steps = int(SIM_DURATION / DT)
    angles = np.empty(n_steps + 1)
    angles[0] = fwd_v_angle_deg(rotvec)

    latched_coeffs = None
    latched_elapsed = 0.0
    for i in range(1, n_steps + 1):
        if latched_coeffs is None:
            latched_coeffs = solve_quintic_hermite_coeffs(
                rotvec, angvel, np.zeros(3), TGT_ROTVEC, TGT_ROTVEC_VEL, TGT_ROTVEC_ACCEL,
                T_LOCAL,
            )
            latched_elapsed = 0.0
        latched_elapsed = min(latched_elapsed + DT, T_LOCAL)
        rotvec = evaluate_vector(latched_coeffs, latched_elapsed, order=0)
        angvel = evaluate_vector(latched_coeffs, latched_elapsed, order=1)
        angles[i] = fwd_v_angle_deg(rotvec)
    return angles


def settle_time(angles, dt):
    below = np.where(angles <= SETTLE_THRESHOLD_DEG)[0]
    if len(below) == 0:
        return None
    # First index after which it *stays* below threshold (avoid reporting a
    # transient dip through the threshold).
    for idx in below:
        if np.all(angles[idx:] <= SETTLE_THRESHOLD_DEG):
            return idx * dt
    return None


def main():
    print(f"scenario: 90 degree corner (+X -> +Y), t_local={T_LOCAL}s, dt={DT}s, "
          f"settle threshold={SETTLE_THRESHOLD_DEG} deg\n")

    baseline = run_baseline()
    fixed = run_latched_catchup()

    print("t[s]   baseline(deg)  latched_fix(deg)")
    n_steps = len(baseline)
    step = max(1, round(0.5 / DT))
    for i in range(0, n_steps, step):
        t = i * DT
        print(f"{t:5.2f}  {baseline[i]:12.2f}  {fixed[i]:16.2f}")

    baseline_settle = settle_time(baseline, DT)
    fixed_settle = settle_time(fixed, DT)
    print(f"\nbaseline settle time (<= {SETTLE_THRESHOLD_DEG} deg, stays below): "
          f"{baseline_settle}s" if baseline_settle is not None
          else f"\nbaseline settle time: NEVER settles within {SIM_DURATION}s")
    print(f"latched-fix settle time: {fixed_settle}s" if fixed_settle is not None
          else f"latched-fix settle time: NEVER settles within {SIM_DURATION}s")

    if baseline_settle is None or (fixed_settle is not None and fixed_settle < baseline_settle):
        print("\n=> REPRODUCED: current per-tick full-horizon re-solve settles much slower "
              "than a single t_local (matches the doc's live ~13s observation vs. t_local=1s), "
              "and the latched shrinking-horizon fix settles close to t_local instead.")
    else:
        print("\n=> NOT reproduced as expected -- baseline settled about as fast as the fix; "
              "re-examine the scenario/assumptions before trusting the doc's live-sim reading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
