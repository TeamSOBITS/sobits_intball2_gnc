#!/usr/bin/env python3
"""Capture the REAL production-densified waypoints_flat that
minco_native_py.plan_minco_heuristic_time actually receives for the
2026-09-18 Zeno-stall-route sim repro (p0/q0 from real TF, via
[nav_entry, inspection_entry_1], target capture_point_2).

Monkeypatches minco_native_py.plan_minco_heuristic_time to capture its
arguments instead of running the real solve, then calls the real,
unmodified MincoTrajectory (gnc/sobits_intball2_gnc/guidance/trajectory/
minco_trajectory.py) so the real attitude_resample_spacing_m densify
logic runs. No production code is modified.

Output: real_waypoints_flat_zeno_route_24pt.txt (checked in alongside
this script) -- 24 waypoints, matches the real guidance_node log line
"plan_minco_heuristic_time failed (error_code=1) for 24 waypoints"
exactly (see docs/2026-09-18_plan_minco_heuristic_time_rotation_aware_floor_offline_verification.md).

Feed that file to bench_v17_attitude_aware_heuristic_time's optional
2nd argv to test a candidate fix against the REAL production input
instead of a hand-built synthetic scenario (the synthetic ones turned
out to be misleadingly easier -- same doc, "座標の矛盾発覚" section).

Run: python3 capture_real_waypoints_zeno_route.py
(needs minco_native_py importable, i.e. run inside a shell with the
workspace sourced)
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import os
import numpy as np
from unittest.mock import patch
import minco_native_py
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory, MincoInfeasibleError

captured = {}

def fake_plan_minco_heuristic_time(waypoints_flat, v0, w0, target_speed, max_accel,
                                    via_half_width, wrench_safety_margin):
    captured["waypoints_flat"] = list(waypoints_flat)
    captured["v0"] = list(v0)
    captured["w0"] = list(w0)
    captured["target_speed"] = target_speed
    captured["max_accel"] = max_accel
    captured["via_half_width"] = via_half_width
    captured["wrench_safety_margin"] = wrench_safety_margin
    # real failure signature reproduced by the actual native call today
    return (False, 1, [], [], 0.0)

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])  # x,y,z,w (ROS order) -- MincoTrajectory expects?
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

waypoints = [p0, nav_entry, inspection_entry_1, capture_point_2]

with patch("minco_native_py.plan_minco_heuristic_time", side_effect=fake_plan_minco_heuristic_time):
    try:
        MincoTrajectory(
            waypoints, q0, v0=np.zeros(3), w0=np.zeros(3),
            via_half_width=0.0,
            attitude_resample_spacing_m=0.3,
            wrench_safety_margin=0.7,
            target_speed=0.5, max_accel=0.0996 / 4.5,
        )
    except MincoInfeasibleError:
        pass

n = len(captured["waypoints_flat"]) // 6
print("num_waypoints(densified) =", n)
print("v0 =", captured["v0"], "w0 =", captured["w0"])
print("target_speed =", captured["target_speed"], "max_accel =", captured["max_accel"])
print("via_half_width =", captured["via_half_width"], "wrench_safety_margin =", captured["wrench_safety_margin"])

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "real_waypoints_flat_zeno_route_24pt.txt")
with open(out_path, "w") as f:
    f.write("%d\n" % n)
    wf = captured["waypoints_flat"]
    for i in range(n):
        row = wf[6*i:6*i+6]
        f.write(" ".join("%.10f" % x for x in row) + "\n")
print("wrote", out_path)
