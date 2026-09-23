#!/usr/bin/env python3
"""A案（globalレイヤーのMINCO solveをfree-timeへ切り替え）を、生の
minco_native_py.plan_mincoではなく、本番の未変更クラス
MincoTrajectory（gnc/sobits_intball2_gnc/guidance/trajectory/
minco_trajectory.py）経由で検証する。densify（attitude_resample_spacing_m）・
via_half_width・wrench_safety_marginの受け渡しなど、本番の抽象化層を
実際に通した上で、free-time経路(target_speed=None, max_accel=None)が
正しく動くか・想定duration(~61.68s)と一致するか・sample()で軌道を
連続的に評価できるかを確認する。

ReplanningMincoV2Tracker（実際にsimで使われているクラス）は
target_speed/max_accel を float() で必須化しておりfree-time経路を
選択できないため、本スクリプトはその1段下のMincoTrajectory単体を
対象にする（本番コードは一切変更しない）。

シナリオはcapture_real_waypoints_zeno_route.pyと同一
（2026-09-18 zeno-stallルートのシム再現に使った実TF値）。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory, MincoInfeasibleError

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])
waypoints = [p0, nav_entry, inspection_entry_1, capture_point_2]

COMMON_KWARGS = dict(
    v0=np.zeros(3), w0=np.zeros(3),
    via_half_width=0.0,
    attitude_resample_spacing_m=0.3,
    wrench_safety_margin=0.7,
)

print("=== 本番と同じ設定 (target_speed/max_accel指定、plan_minco_heuristic_time経路) ===")
try:
    traj_heuristic = MincoTrajectory(
        waypoints, q0, target_speed=0.5, max_accel=0.0996 / 4.5, **COMMON_KWARGS
    )
    print(f"num_waypoints(densified)={traj_heuristic.num_waypoints} "
          f"duration={traj_heuristic.global_total_duration:.4f}s")
except MincoInfeasibleError as e:
    print(f"MincoInfeasibleError: {e}")

print("\n=== A案: target_speed=None, max_accel=None (plan_minco free-time経路) ===")
try:
    traj_freetime = MincoTrajectory(waypoints, q0, target_speed=None, max_accel=None, **COMMON_KWARGS)
    print(f"num_waypoints(densified)={traj_freetime.num_waypoints} "
          f"duration={traj_freetime.global_total_duration:.4f}s")
except MincoInfeasibleError as e:
    print(f"MincoInfeasibleError: {e}")
    sys.exit(1)

print("\n--- sample()で軌道を連続評価できるか確認 (10点) ---")
dur = traj_freetime.global_total_duration
for frac in np.linspace(0.0, 1.0, 10):
    t = frac * dur
    pos, vel, acc, quat = traj_freetime.sample(t)
    print(f"t={t:7.3f}s pos={np.round(pos, 4)} |vel|={np.linalg.norm(vel):.4f}")

print("\n--- 始点・終点がwaypointと一致するか確認 ---")
pos0, _, _, _ = traj_freetime.sample(0.0)
posN, _, _, _ = traj_freetime.sample(dur)
print(f"sample(0)     = {pos0} (expected ~{p0})")
print(f"sample(dur)   = {posN} (expected ~{capture_point_2})")
print(f"start error = {np.linalg.norm(pos0 - p0)*1000:.3f}mm")
print(f"end error   = {np.linalg.norm(posN - capture_point_2)*1000:.3f}mm")
