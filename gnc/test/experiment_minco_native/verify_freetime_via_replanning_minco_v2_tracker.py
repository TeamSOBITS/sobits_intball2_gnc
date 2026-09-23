#!/usr/bin/env python3
"""A案（free-time経路、target_speed=None/max_accel=None）を、
ReplanningMincoV2Tracker（本番でsim実行に使われるクラス、今回
target_speed/max_accelのNone許可分岐を追加）を直接使ってオフライン検証する。
ROS/シムなし、pose_fnをトラッカー自身の直前の出力を完全に追従する
理想追従（perfect tracking oracle）にして、global/local 2層・MODEL_KF・
via退役ロジックを含む本物のトラッカーの状態遷移を通しで確認する。

シナリオはcapture_real_waypoints_zeno_route.py/
docs/2026-09-18_minco_stretch_fix_sim_verification_facts.mdのシム検証と
同一（p0/q0は実TF値、via=[nav_entry, inspection_entry_1]、
target=capture_point_2）。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    ReplanningMincoV2Tracker,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoInfeasibleError

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

state = {"pos": p0.copy(), "quat": q0.copy(), "stamp": 0.0}


def pose_fn():
    return state["pos"].copy(), state["quat"].copy(), state["stamp"]


def tf_fresh_fn(stamp):
    return True


print("=== ReplanningMincoV2Tracker(target_speed=None, max_accel=None) 構築 ===")
try:
    tracker = ReplanningMincoV2Tracker(
        p0, capture_point_2, pose_fn, tf_fresh_fn, q0,
        target_speed=None, max_accel=None,
        route_waypoints=np.vstack([nav_entry, inspection_entry_1]),
        via_half_width=0.0, wrench_safety_margin=0.7,
        attitude_resample_spacing_m=0.3,
    )
except MincoInfeasibleError as e:
    print(f"MincoInfeasibleError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

print(f"initial global trajectory duration={tracker.trajectory.global_total_duration:.3f}s")
print(f"initial total_duration estimate={tracker.total_duration:.3f}s")

dt = 0.05
t = 0.0
replan_count = 0
max_ticks = 3000
for i in range(max_ticks):
    t += dt
    p_out, v_out, a_out, q_out = tracker.sample(t)
    state["pos"] = p_out.copy()
    state["quat"] = q_out.copy()
    state["stamp"] = t
    if tracker.last_replan_occurred:
        replan_count += 1
    if tracker.last_fallback_reason is not None:
        print(f"t={t:.2f}s UNEXPECTED fallback: {tracker.last_fallback_reason}")
    if tracker.last_local_fallback:
        print(f"t={t:.2f}s local fallback triggered")

    remaining = np.linalg.norm(capture_point_2 - p_out)
    if remaining < 0.005 and i > 10:
        print(f"converged at t={t:.2f}s, remaining={remaining*1000:.2f}mm")
        break
else:
    print(f"did NOT converge within {max_ticks} ticks (t={t:.2f}s)")

print(f"\nfinal p_out={p_out}")
print(f"final error vs capture_point_2 = {np.linalg.norm(p_out - capture_point_2)*1000:.3f}mm")
print(f"global replans occurred: {replan_count}")
print(f"final total_duration estimate={tracker.total_duration:.3f}s")
