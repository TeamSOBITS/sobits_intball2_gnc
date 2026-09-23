#!/usr/bin/env python3
"""Isolation experiment (2026-09-20): does the goal-arrival oscillation
observed live (docs/2026-09-18_minco_stretch_fix_sim_verification_facts.md,
facts 9/19) require a real closed feedback loop (the tracker's own
commanded output feeding back as the next tick's "observed" pose, through
some vehicle response lag), or does it appear even with an instantaneous,
lag-free self-referential loop?

This supersedes the conclusion drawn from ``isolate_local_layer_oscillation.
py`` (fact 20/21 in the facts doc): that experiment fed ``pose_fn`` the
tracker's *own single global trajectory's analytic point* -- a value that
never depends on what the tracker itself actually output on any previous
tick. It is therefore not a closed loop at all, and cannot exercise the
KF-estimator's dependence on noisy/lagged position *observations* (as
opposed to the true trajectory state), nor the discontinuity at the
``distance_fallback_m`` latch trip (module docstring of
``replanning_minco_v2_tracker.py``: local-layer aim point jumps from a
moving global-trajectory look-ahead point to the fixed, zero-velocity
``p_target`` the instant remaining distance drops below the threshold).

Re-analysis of the live run's ``guidance_node.log`` (2026-09-20) found that
the last "re-planned trajectory" log line appears at relative t=118.63s,
while the oscillation window (facts 9, t_sim 6201-6223) starts only at
t_sim=6201.22 -- i.e. *after* global replanning had already latched off.
So the previously-blamed "repeated global replan" cannot be the proximate
cause of the oscillation; the latch's own aim-point discontinuity, combined
with closed-loop dynamics, is the next candidate.

This experiment drives ``ReplanningMincoV2Tracker.sample(t)`` for real with
production defaults (``global_replan_period=2.0``, so periodic replanning
and the eventual latch both occur exactly as in the live run), but replaces
``pose_fn`` with a simple first-order-lag "vehicle" that chases the
tracker's own previously-commanded output:

    vehicle_state += (1 - exp(-dt/lag_tau)) * (commanded_state - vehicle_state)

``lag_tau=0`` means the vehicle matches the command instantly (still a
genuine closed loop -- the KF sees exactly what was commanded, one tick
late -- but with no response delay). Larger ``lag_tau`` adds delay,
approximating real thruster/attitude-control response lag.

Offline-only, no production code modified, no live sim/ROS needed.
"""
import sys

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
sys.path.insert(0, "/root/colcon_ws/install/minco_native_py/lib/python3.10/site-packages")
sys.path.insert(0, "/root/colcon_ws/install/minco_native_py/local/lib/python3.10/dist-packages")

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    DEFAULT_DISTANCE_FALLBACK_M,
    DEFAULT_GLOBAL_REPLAN_PERIOD_S,
    DEFAULT_T_LOCAL_S,
    ReplanningMincoV2Tracker,
)

# Same live-captured initial conditions as isolate_local_layer_oscillation.py
# (docs/2026-09-18_minco_stretch_fix_sim_verification_facts.md fact 7).
P0 = np.array([10.996838770858696, -4.296825736180297, 5.005080080098301])
Q0 = np.array([-0.7070906170198997, 0.707122941161243, 7.352282680177928e-05,
                -5.335946126746761e-07])
P_TARGET = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])
ROUTE_WAYPOINTS = np.array([[10.936, -9.0, 5.0]])  # inspection_entry_1

TARGET_SPEED = 0.5
MAX_ACCEL = 0.0996 / 4.5
VIA_HALF_WIDTH = 0.0
WRENCH_SAFETY_MARGIN = 0.7
ATTITUDE_RESAMPLE_SPACING_M = 0.3

DT = 0.02
MAX_T = 260.0  # live run's total was ~161s (fact 15); pad well past that


def quat_lerp_toward(q_from, q_to, alpha):
    """Cheap slerp-ish step for small per-tick alpha; picks the closer
    double-cover sign first so lerp doesn't take the long way around."""
    if np.dot(q_from, q_to) < 0.0:
        q_to = -q_to
    q = q_from + alpha * (q_to - q_from)
    return q / np.linalg.norm(q)


def run_experiment(lag_tau_pos, lag_tau_rot, label):
    vehicle_p = P0.copy()
    vehicle_q = Q0.copy()
    cmd_p = [P0.copy()]
    cmd_q = [Q0.copy()]
    last_pose_t = [0.0]

    def pose_fn():
        nonlocal vehicle_p, vehicle_q
        t = last_pose_t[0]
        return (vehicle_p.copy(), vehicle_q.copy(), t)

    def tf_fresh_fn(_stamp):
        return True

    tracker = ReplanningMincoV2Tracker(
        P0, P_TARGET, pose_fn, tf_fresh_fn, Q0,
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
        route_waypoints=ROUTE_WAYPOINTS,
        global_replan_period=DEFAULT_GLOBAL_REPLAN_PERIOD_S,
        t_local=DEFAULT_T_LOCAL_S,
        via_half_width=VIA_HALF_WIDTH, wrench_safety_margin=WRENCH_SAFETY_MARGIN,
        attitude_resample_spacing_m=ATTITUDE_RESAMPLE_SPACING_M,
    )

    n_steps = int(MAX_T / DT)
    rows = []  # (t, dist_to_goal, replanned, post_latch)
    latch_t = None
    for i in range(1, n_steps + 1):
        t = i * DT
        dt = t - last_pose_t[0]

        if lag_tau_pos <= 0.0:
            vehicle_p = cmd_p[-1].copy()
        else:
            alpha = 1.0 - np.exp(-dt / lag_tau_pos)
            vehicle_p = vehicle_p + alpha * (cmd_p[-1] - vehicle_p)

        if lag_tau_rot <= 0.0:
            vehicle_q = cmd_q[-1].copy()
        else:
            alpha_r = 1.0 - np.exp(-dt / lag_tau_rot)
            vehicle_q = quat_lerp_toward(vehicle_q, cmd_q[-1], alpha_r)

        last_pose_t[0] = t
        p_out, _v, _a, q_out = tracker.sample(t)
        cmd_p.append(np.asarray(p_out))
        cmd_q.append(np.asarray(q_out))

        dist = float(np.linalg.norm(vehicle_p - P_TARGET))
        if dist < DEFAULT_DISTANCE_FALLBACK_M and latch_t is None:
            latch_t = t
        rows.append((t, dist, bool(tracker.last_replan_occurred), latch_t is not None))

        if getattr(tracker, "last_fallback_reason", None) == "tf_stale":
            break

    rows = np.array([(t, d, float(r), float(pl)) for t, d, r, pl in rows])
    n_replans = int(rows[:, 2].sum())

    print(f"\n=== {label} (lag_tau_pos={lag_tau_pos}s, lag_tau_rot={lag_tau_rot}s) ===")
    print(f"n_steps={len(rows)}  n_replans_occurred={n_replans}  "
          f"latch_t={latch_t}")

    post_latch_mask = rows[:, 3] > 0.5
    if not post_latch_mask.any():
        print("  never entered post-latch regime (distance never < "
              f"{DEFAULT_DISTANCE_FALLBACK_M}m) within MAX_T -- inconclusive")
        return

    post = rows[post_latch_mask]
    d = post[:, 1]
    dmin, dmax = float(d.min()), float(d.max())
    # Count local re-growth events: distance increasing by >5mm from a
    # preceding local minimum, mirroring the facts doc's manual awk method.
    regrowth_events = 0
    running_min = d[0]
    for val in d[1:]:
        if val < running_min:
            running_min = val
        elif val - running_min > 0.005:
            regrowth_events += 1
            running_min = val
    print(f"  post-latch window: t=[{post[0,0]:.2f}, {post[-1,0]:.2f}]s "
          f"({len(post)} samples)")
    print(f"  dist min/max in post-latch window: {dmin:.5f}/{dmax:.5f}")
    print(f"  regrowth events (>5mm bounce off a local min): {regrowth_events}")
    if dmax - dmin > 0.02 and regrowth_events > 0:
        print("  -> OSCILLATION reproduced")
    else:
        print("  -> monotonic-ish, no oscillation")


if __name__ == "__main__":
    for lag_tau in (0.0, 0.15, 0.4, 0.8):
        run_experiment(lag_tau, lag_tau, f"self-referential closed loop, lag={lag_tau}s")
