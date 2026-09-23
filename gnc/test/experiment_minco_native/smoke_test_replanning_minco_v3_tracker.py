"""ReplanningMincoV3Tracker(本番実装、gnc/sobits_intball2_gnc/guidance/
trajectory_tracking/replanning_minco_v3_tracker.py)の実装直後スモークテスト。

理想追従オラクル（pose_fnがtracker自身の直前出力にそのまま追従する、
prototype_ego_v2_style_local_replan.pyと同じ手法）で、実際にコンストラクト
してsample()ループを回し、収束するか・例外が出ないかを確認する。
`docs/2026-09-20_ego_v2_style_replan_migration_plan.md`のシムテストの前段、
まずクラス単体でのオフライン疎通確認。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np

from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import (
    ReplanningMincoV3Tracker,
)

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

DT = 0.05
MAX_TICKS = 6000
CONVERGE_TOL = 0.005

state = {"pos": p0.copy(), "quat": q0.copy(), "t": 0.0}


def fake_pose_fn():
    return state["pos"].copy(), state["quat"].copy(), state["t"]


def fake_tf_fresh_fn(_stamp):
    return True


tracker = ReplanningMincoV3Tracker(
    p0, capture_point_2, fake_pose_fn, fake_tf_fresh_fn, q0,
    target_speed=0.5, max_accel=0.0996 / 4.5,
    route_waypoints=[nav_entry, inspection_entry_1],
    wrench_safety_margin=0.7, attitude_resample_spacing_m=0.3,
)
print(f"initial global trajectory duration={tracker.trajectory.global_total_duration:.3f}s "
      f"num_waypoints={tracker.trajectory.num_waypoints}")

t = 0.0
converged_at = None
for i in range(MAX_TICKS):
    t += DT
    state["t"] = t
    p, v, a, q = tracker.sample(t)
    state["pos"] = p.copy()
    state["quat"] = q.copy()

    if tracker.last_local_fallback:
        print(f"t={t:.2f}s local fallback triggered (reason={tracker.last_fallback_reason})")
    if tracker.last_fallback_reason == "tf_stale":
        print(f"t={t:.2f}s tf_stale fallback -- aborting")
        break

    remaining = np.linalg.norm(capture_point_2 - p)
    if remaining < CONVERGE_TOL and i > 10:
        converged_at = t
        break
else:
    print(f"did NOT converge within {MAX_TICKS} ticks (t={t:.2f}s)")

if converged_at is not None:
    print(f"converged at t={converged_at:.2f}s")
    print(f"final position error = {np.linalg.norm(state['pos'] - capture_point_2)*1000:.3f}mm")

print(f"total_duration (ETA) at final tick = {tracker.total_duration:.3f}s")
