#!/usr/bin/env python3
"""A案(free-time)の周期的global replanで速度が時間とともに劣化する
根本原因調査。ReplanningMincoV2Trackerを理想追従オラクルで動かしながら、
_rebuild_global直後のsegment_times・route長・v0(境界条件)・
_freetime_avg_speedを毎回ダンプし、なぜ速い区間が実現されないかを見る。
本番コードは読むだけ、一切変更しない。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    ReplanningMincoV2Tracker,
)

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

tracker = ReplanningMincoV2Tracker(
    p0, capture_point_2, pose_fn, tf_fresh_fn, q0,
    target_speed=None, max_accel=None,
    route_waypoints=np.vstack([nav_entry, inspection_entry_1]),
    via_half_width=0.0, wrench_safety_margin=0.7,
    attitude_resample_spacing_m=0.3,
)

orig_rebuild = tracker._rebuild_global
replan_log = []

def traced_rebuild(p_now, v0, w0):
    orig_rebuild(p_now, v0, w0)
    traj = tracker._global_trajectory
    remaining = np.linalg.norm(capture_point_2 - p_now)
    replan_log.append({
        "t": tracker._prev_t,
        "p_now": p_now.copy(),
        "v0": v0.copy(),
        "|v0|": np.linalg.norm(v0),
        "remaining": remaining,
        "K": len(traj._segment_times),
        "duration": traj.global_total_duration,
        "avg_speed": tracker._freetime_avg_speed,
        "segment_times_head": list(traj._segment_times[:3]),
        "segment_times_tail": list(traj._segment_times[-3:]),
    })

tracker._rebuild_global = traced_rebuild

dt = 0.05
t = 0.0
for i in range(2000):
    t += dt
    p_out, v_out, a_out, q_out = tracker.sample(t)
    state["pos"] = p_out.copy(); state["quat"] = q_out.copy(); state["stamp"] = t
    if np.linalg.norm(capture_point_2 - p_out) < 0.005 and i > 10:
        break

for e in replan_log:
    print(f"t={e['t']:6.2f} remaining={e['remaining']:.4f}m |v0|={e['|v0|']:.4f} "
          f"K={e['K']:3d} duration={e['duration']:7.3f}s avg_speed={e['avg_speed']:.4f} "
          f"seg_head={[round(x,3) for x in e['segment_times_head']]} "
          f"seg_tail={[round(x,3) for x in e['segment_times_tail']]}")
