#!/usr/bin/env python3
"""Re-check of the "Zeno stall" investigation (docs/archive/
2026-09-01_replanning_minco_zeno_stall_investigation.md) against
``plan_minco_heuristic_time`` instead of the old free-time ``plan_minco``.

That investigation found that re-solving MINCO's free-time joint
(space+time) optimization every replan tick, from the actual measured
(near-zero) v0, causes a "Zeno stall" (repeatedly re-solving the same
~47s trajectory's near-static ramp-up phase, so the vehicle barely moves)
at high replan rates, and unbounded overshoot/divergence at low replan
rates ("案A": lowering replan_rate_hz alone does not fix it, the failure
mode just changes shape). This was concluded to require a 2-layer
global+local split (案E) -- which is what Phase 4's ReplanningMincoV2Tracker
implements, and whose local layer we've since found has no wrench-envelope
feasibility awareness (docs/archive/achieved/
2026-09-17_replanning_minco_v2_local_layer_duty_saturation_incident.md --
note that doc's incident diagnosis was later found wrong, see its
correction note and docs/archive/achieved/2026-09-18_replanning_minco_v2_zeno_route_rediagnosis_toppra_ready_bug.md;
the "no wrench-envelope feasibility awareness" design fact itself still
holds).

BUT: that Zeno-stall investigation predates Phase 1
(docs/2026-09-01_replanning_minco_v4_production_port_plan.md), which
replaced MINCO's free-time joint optimization with a Fast-Planner-faithful
heuristic-time + analytic-stretch scheme (plan_minco_heuristic_time) for
exactly the reason Fast-Planner/EGO-Planner never hit this instability in
the first place: they never do free-time joint optimization at replan time.
``ReplanningTrajectoryTracker`` already supports switching to this (the
``use_minco_heuristic_time`` flag, default False, docstring notes "not yet
sim-validated") but production (``replanning_minco`` mode in
guidance_executor.py) does not enable it -- nobody has checked whether a
SINGLE-layer heuristic-time replan loop (no separate local Hermite layer
at all -- just periodically re-solve and directly sample the resulting,
already wrench-feasible-by-construction MincoTrajectory) avoids the
Zeno stall / divergence that motivated inventing the 2-layer split.

If it does, the entire local-layer feasibility-governor design explored in
gnc/test/experiment_local_layer_feasibility_governor.py may be unnecessary.

Uses the real, unmodified ``MincoTrajectory`` (which calls the real
``minco_native_py.plan_minco_heuristic_time``) -- no production code
touched. "完全追従" methodology (true state advances by exactly sampling
the just-solved trajectory dt seconds forward), matching the original
investigation's own methodology for a like-for-like comparison against its
documented dt=2.0s numbers.

Not a pytest test (no test_ prefix) -- standalone experiment:
    python3 test/experiment_heuristic_time_single_layer_zeno_check.py
"""
import numpy as np

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

TARGET_SPEED = 0.5
MAX_ACCEL = 0.0996 / 4.5  # trajectory_controller.max_force min axis / mass
Q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity, w,x,y,z -- face_travel derives rotvecs

# Same incident route as the original investigation and this session's
# Phase 5 sim run: current -> nav_entry -> inspection_entry_1 -> capture_point_2
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
INSPECTION_ENTRY_1 = np.array([10.936, -9.0, 5.0])  # z not in original grep, matches nav_entry's plane
CAPTURE_POINT_2 = np.array([10.269, -9.435, 5.236])
ROUTE = [NAV_ENTRY, INSPECTION_ENTRY_1, CAPTURE_POINT_2]

START_STATIC = np.array([10.998, -4.297, 5.006])  # matches the original doc's cold-start head position


def run_scenario(label, start_pos, start_vel, dt, duration_s):
    p_now = np.array(start_pos, dtype=float)
    v0 = np.array(start_vel, dtype=float)
    w0 = np.zeros(3)
    t = 0.0
    solve_times = []
    trace = []
    n_infeasible = 0
    while t < duration_s:
        waypoints = np.vstack([p_now] + ROUTE)
        try:
            traj = MincoTrajectory(
                waypoints, Q0, v0=v0, w0=w0,
                target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
            )
        except Exception as exc:  # MincoInfeasibleError
            n_infeasible += 1
            t += dt
            continue
        solve_times.append(traj.solve_wall_seconds)
        p_next, v_next, _a, _q = traj.sample(dt)
        p_now, v0 = p_next, v_next
        t += dt
        if int(t / dt) % max(1, int(round(10.0 / dt))) == 0 or t >= duration_s:
            trace.append((t, p_now.copy(), np.linalg.norm(v0), traj.global_total_duration))

    final_err = np.linalg.norm(p_now - CAPTURE_POINT_2)
    print(f"\n--- {label} (dt={dt}s, {duration_s}s total) ---")
    for tt, pp, vnorm, dur in trace:
        print(f"  t={tt:6.1f}s  head=({pp[0]:7.3f},{pp[1]:7.3f},{pp[2]:7.3f})  "
              f"|v|={vnorm:.4f}  planned_duration={dur:6.2f}s")
    print(f"  final target error = {final_err:.3f}m  "
          f"(n_infeasible_replans={n_infeasible}/{int(duration_s/dt)}, "
          f"solve_time mean={1000*np.mean(solve_times):.1f}ms max={1000*np.max(solve_times):.1f}ms)")
    return final_err


print("=== single-layer plan_minco_heuristic_time, direct sampling, no local Hermite layer ===")
print("(comparing against the documented free-time plan_minco failure modes:")
print(" dt=0.1s -> stall (barely moves); dt=2.0s -> ~0.8m overshoot then drifts)")

for dt in (0.1, 1.0, 2.0):
    run_scenario(f"static start, {dt}", START_STATIC, [0.0, 0.0, 0.0], dt, 120.0)

for dt in (0.1, 1.0, 2.0):
    run_scenario(f"cruise start (init_vy=0.3), {dt}", START_STATIC, [0.0, 0.3, 0.0], dt, 40.0)
