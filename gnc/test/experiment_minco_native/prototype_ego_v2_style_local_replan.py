#!/usr/bin/env python3
"""EGO-Planner v2の実際の構造（planner_manager.cppを直接確認）に合わせた
単層receding-horizonプロトタイプ、本番コード非依存(``ReplanningMincoV2Tracker``
は使わない、新規クラスもここだけに閉じる)。

うちの既存``ReplanningMincoV2Tracker``（global=残りルート全体をLBFGS/stretch
で毎``global_replan_period``秒フル再最適化＋local=毎tickのclosed-form
quintic Hermite平滑化）は、``docs/
2026-09-20_minco_global_replan_freetime_switch_offline_investigation.md``の
実シム検証で「終盤ほぼ完全停止」という理想追従オラクルには現れない劣化を
示し、A案（globalをfree-timeに変えるだけ）では解決しないと分かった。

EGO-Planner-v2の実際のソース（``/home/space_project/reference_repos/
EGO-Planner-v2-main/swarm-playground/main_ws/src/planner/plan_manage/src/
planner_manager.cpp``）を読むと、うちとは「重い/軽い」の役割が逆:

- global（``planGlobalTrajWaypoints``）: 一度きり、LBFGS最適化なしの
  安い閉形式ガイド経路（うちで言えば``plan_minco_heuristic_time``相当だが
  二度と再solveしない）。
- local（``reboundReplan``）: **これが実際に追従される軌道そのもの**。
  ``getLocalTarget``でglobal経路上のplanning_horizen先の点を目標に、
  そこまでの短いMINCO軌道を頻繁に（固定周期ではなくイベント駆動、
  デフォルト``thresh_replan_time=1.0s``）free-time LBFGSで解き直す。
  closed-form平滑化コネクタ層は存在しない。

このスクリプトはこの構造を最小限で再現する:
- global: ``MincoTrajectory(..., target_speed=0.5, max_accel=_MAX_ACCEL)``
  （heuristic_time経路）で一度だけ構築、以後再solveしない。
- local: 1.0秒ごと（または現在のlocal軌道が終わりに近づいたら）、global
  軌道上の``planning_horizen``先の点を目標に、現在位置/速度からの単一
  segment free-time MINCO(``target_speed=None, max_accel=None``)を解き直し、
  そのlocal軌道を直接追従する（Hermite層なし、hard cutover）。

姿勢はスコープ外（``face_travel=False``でq0固定、ユーザー指示）。
via-waypoint退役ロジックもスコープ外（今回はglobalは一度きりなので不要）。

検証項目（``docs/2026-09-20_..._offline_investigation.md``の反省を踏まえ、
「fallbackが起きないか」だけでなく実際の**速度の時系列**を見る):
- 理想追従オラクル（pose_fnがtracker自身の直前出力に完全追従）で、
  10秒ビンごとの平均速度が終盤まで大きく落ち込まないか
- 収束するか、feasibility例外が出ないか

Run: python3 prototype_ego_v2_style_local_replan.py
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np

from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)

_TARGET_SPEED = 0.5
_MAX_ACCEL = 0.0996 / 4.5  # test_minco_native_regression.pyと同一値

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

WAYPOINTS = [p0, nav_entry, inspection_entry_1, capture_point_2]
PLANNING_HORIZEN = 2.0  # EGO-v2 default 5.0m; ルート全長約6.6mに対する比率を保つため縮小
REPLAN_PERIOD = 1.0  # EGO-v2 thresh_replan_time default
DT = 0.05
MAX_TICKS = 6000
CONVERGE_TOL = 0.005


def get_local_target(global_traj, start_pt, global_end_pt, cur_glb_t, planning_horizen):
    total_dur = global_traj.global_total_duration
    t_step = max(planning_horizen / 20.0 / max(_TARGET_SPEED, 1e-6), DT)
    t = cur_glb_t
    while t < total_dur:
        pos_t, vel_t, _, _ = global_traj.sample(t)
        if np.linalg.norm(pos_t - start_pt) >= planning_horizen:
            return pos_t, vel_t, t, False
        t += t_step
    pos_t, vel_t, _, _ = global_traj.sample(total_dur)
    return global_end_pt.copy(), np.zeros(3), total_dur, True


def solve_local(p_now, v_now, local_target_pos, local_target_vel):
    return MincoTrajectory(
        [p_now, local_target_pos], q0, v0=v_now, w0=np.zeros(3),
        face_travel=False, target_speed=None, max_accel=None,
    )


print("=== global軌道構築（一度きり、heuristic_time、再solveしない）===")
global_traj = MincoTrajectory(
    WAYPOINTS, q0, v0=np.zeros(3), w0=np.zeros(3),
    face_travel=False, via_half_width=0.0, wrench_safety_margin=0.7,
    attitude_resample_spacing_m=0.3,
    target_speed=_TARGET_SPEED, max_accel=_MAX_ACCEL,
)
print(f"global duration={global_traj.global_total_duration:.3f}s")

state_pos = p0.copy()
state_vel = np.zeros(3)
glb_t_of_lc_tgt = 0.0
local_traj = None
local_start_wall_t = 0.0
touch_goal = False
replan_count = 0

history_t = []
history_pos = []

t = 0.0
for i in range(MAX_TICKS):
    need_replan = (
        local_traj is None
        or (t - local_start_wall_t) >= min(REPLAN_PERIOD, local_traj.global_total_duration)
    )
    if need_replan and not touch_goal:
        local_target_pos, local_target_vel, glb_t_of_lc_tgt, touch_goal = get_local_target(
            global_traj, state_pos, capture_point_2, glb_t_of_lc_tgt, PLANNING_HORIZEN
        )
        try:
            local_traj = solve_local(state_pos, state_vel, local_target_pos, local_target_vel)
        except MincoInfeasibleError as e:
            print(f"t={t:.2f}s local replan MincoInfeasibleError: {e}")
            break
        local_start_wall_t = t
        replan_count += 1

    tau = t - local_start_wall_t
    p_out, v_out, a_out, q_out = local_traj.sample(tau)
    state_pos = p_out.copy()
    state_vel = v_out.copy()

    history_t.append(t)
    history_pos.append(state_pos.copy())

    remaining = np.linalg.norm(capture_point_2 - state_pos)
    t += DT
    if remaining < CONVERGE_TOL and i > 10:
        print(f"converged at t={t:.2f}s, remaining={remaining*1000:.2f}mm")
        break
else:
    print(f"did NOT converge within {MAX_TICKS} ticks (t={t:.2f}s)")

print(f"final p_out={state_pos}")
print(f"final error vs capture_point_2 = {np.linalg.norm(state_pos - capture_point_2)*1000:.3f}mm")
print(f"local replans occurred: {replan_count}")

history_t = np.array(history_t)
history_pos = np.array(history_pos)

print("\n=== 10秒ビンごとの平均/最大速度 ===")
bin_size = 10.0
t_max = history_t[-1]
bin_start = 0.0
while bin_start < t_max:
    mask = (history_t >= bin_start) & (history_t < bin_start + bin_size)
    if mask.sum() >= 2:
        pos_bin = history_pos[mask]
        t_bin = history_t[mask]
        diffs = np.linalg.norm(np.diff(pos_bin, axis=0), axis=1)
        dts = np.diff(t_bin)
        speeds = diffs / dts
        print(f"t={bin_start:5.0f}-{bin_start + bin_size:5.0f}s: "
              f"mean={speeds.mean():.4f} max={speeds.max():.4f} m/s")
    bin_start += bin_size
