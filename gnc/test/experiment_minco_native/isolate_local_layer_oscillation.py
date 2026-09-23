#!/usr/bin/env python3
"""Isolation experiment (2026-09-18): does the goal-arrival position
oscillation observed live (see docs/
2026-09-18_minco_stretch_fix_sim_verification_facts.md) come from the
`ReplanningMincoV2Tracker`'s GLOBAL layer repeatedly re-solving every
``global_replan_period``, or from its LOCAL layer (a fresh quintic-Hermite
solve every tick, connecting the KF-estimated current state to a
look-ahead point on the global trajectory)?

Offline-only, no production code modified, no live sim/ROS needed. Drives
`ReplanningMincoV2Tracker.sample(t)` directly with `global_replan_period`
set far larger than the trajectory duration (so exactly one global solve
happens, at construction, and the local layer runs unaided for the whole
run). `pose_fn` feeds back the tracker's OWN global trajectory's exact
analytic position/attitude at each `t` -- i.e. a perfect, noise-free
"vehicle" that always sits exactly where the (single, never-replanned)
global MINCO trajectory says it should. This isolates the local layer's
own recursive KF+quintic-Hermite math from both real vehicle-dynamics
tracking error and from repeated global replanning.

If the local-layer-only output still oscillates near the goal the way the
live run did, the local layer (or its interaction with the
distance_fallback_m latch switching the aim point to the fixed goal) is
implicated. If it stays smooth/monotonic, the oscillation must come from
the global layer's repeated re-solving instead.
"""
import sys

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
sys.path.insert(0, "/root/colcon_ws/install/minco_native_py/lib/python3.10/site-packages")
sys.path.insert(0, "/root/colcon_ws/install/minco_native_py/local/lib/python3.10/dist-packages")

import numpy as np

from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    ReplanningMincoV2Tracker,
)

# Real TF values captured live 2026-09-18 (nav_entry start, capture_point_2
# target, via inspection_entry_1) -- see the facts doc's fact 7 section.
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

GLOBAL_REPLAN_PERIOD = 1.0e6  # effectively "never replan again"
T_LOCAL = 1.0

last_t = [0.0]


def pose_fn():
    t = last_t[0]
    traj = tracker.trajectory
    tail_t = min(t, traj.global_total_duration)
    p, _v, _a, q = traj.sample(tail_t)
    return (np.asarray(p), np.asarray(q), t)


def tf_fresh_fn(_stamp):
    return True


tracker = ReplanningMincoV2Tracker(
    P0, P_TARGET, pose_fn, tf_fresh_fn, Q0,
    target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
    route_waypoints=ROUTE_WAYPOINTS,
    global_replan_period=GLOBAL_REPLAN_PERIOD, t_local=T_LOCAL,
    via_half_width=VIA_HALF_WIDTH, wrench_safety_margin=WRENCH_SAFETY_MARGIN,
    attitude_resample_spacing_m=ATTITUDE_RESAMPLE_SPACING_M,
)

duration = tracker.trajectory.global_total_duration
print("global trajectory duration (single solve):", duration)

dt = 0.02
n_steps = int(duration / dt) + 400  # run a bit past nominal duration too
rows = []
for i in range(1, n_steps + 1):
    t = i * dt
    last_t[0] = t
    p_out, v_out, a_out, q_out = tracker.sample(t)
    dist = float(np.linalg.norm(p_out - P_TARGET))
    rows.append((t, dist, bool(tracker.last_replan_occurred),
                 bool(tracker.last_local_fallback)))
    if dist < 0.001 and t > duration * 0.5:
        break

n_replans = sum(1 for _, _, r, _ in rows if r)
n_local_fallback = sum(1 for _, _, _, f in rows if f)
rows = np.array([(t, d) for t, d, _, _ in rows])

print("n samples:", len(rows), " n_replans_occurred:", n_replans,
      " n_local_fallback:", n_local_fallback)
print("\nlast 5 seconds of local-layer-only distance-to-goal (perfect feedback):")
mask = rows[:, 0] >= rows[-1, 0] - 40
for t, d in rows[mask][::10]:
    print(f"t={t:8.3f}  dist_to_goal={d:.5f}")

dmin = rows[mask][:, 1].min()
dmax = rows[mask][:, 1].max()
print(f"\nmin/max distance-to-goal in that window: {dmin:.5f} / {dmax:.5f}")
