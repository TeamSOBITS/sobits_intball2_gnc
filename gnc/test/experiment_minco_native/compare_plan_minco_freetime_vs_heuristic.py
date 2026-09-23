#!/usr/bin/env python3
"""EGO-Planner v2の「per-piece連続時間最適化」(reallocateTime/lengthenTime
不要、時間を自由変数としてL-BFGSに解かせる方式)は既にminco_native_py.plan_minco
としてバインディングされている(minco_solver.cpp:619 planMinco、backwardT/forwardT
によるvirtual time変換 + weight scheduleでwaypointsと同時に最適化)。

本番で使われているplan_minco_heuristic_time(ヒューリスティック時間 + uniform
stretchループ)と、この既存のplan_minco(自由変数時間)を同一の実route入力
(real_waypoints_flat_zeno_route_24pt.txt、docs/2026-09-18_minco_stretch_fix_
sim_verification_facts.mdのシム検証で使ったのと同じ入力)に対して比較し、
- 総所要時間(duration)
- feasibility(success/error_code)
- leg2(回転が最初のsub-segmentに集中する区間)内でのsegment_times分布が
  uniform stretchのような「全サブセグメント同一T」になっていないか
を軽く確認する。C++は一切変更していない、既存バインディングの呼び出しのみ。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import os
import minco_native_py as m

HERE = os.path.dirname(os.path.abspath(__file__))
WPTS_PATH = os.path.join(HERE, "real_waypoints_flat_zeno_route_24pt.txt")

with open(WPTS_PATH) as f:
    n = int(f.readline())
    waypoints_flat = []
    for _ in range(n):
        waypoints_flat.extend(float(x) for x in f.readline().split())

v0 = [0.0, 0.0, 0.0]
w0 = [0.0, 0.0, 0.0]
VIA_HALF_WIDTH = 0.0
WRENCH_SAFETY_MARGIN = 0.7
TARGET_SPEED = 0.5
MAX_ACCEL = 0.0996 / 4.5

print(f"num_waypoints = {n}")

print("\n=== plan_minco_heuristic_time (production, uniform stretch) ===")
ok_h, err_h, T_h, _, dur_h = m.plan_minco_heuristic_time(
    waypoints_flat, v0, w0, TARGET_SPEED, MAX_ACCEL, VIA_HALF_WIDTH, WRENCH_SAFETY_MARGIN
)
print(f"success={ok_h} error_code={err_h} duration={dur_h:.4f}s K={len(T_h)}")

print("\n=== plan_minco (free-time, EGO-v2-style per-piece L-BFGS) ===")
ok_f, err_f, T_f, _, dur_f = m.plan_minco(
    waypoints_flat, v0, w0, VIA_HALF_WIDTH, WRENCH_SAFETY_MARGIN
)
print(f"success={ok_f} error_code={err_f} duration={dur_f:.4f}s K={len(T_f)}")

if ok_h and ok_f:
    ratio = dur_f / dur_h
    print(f"\nduration ratio (free-time / heuristic+stretch) = {ratio:.4f}")

print("\n--- per-segment T comparison (heuristic+stretch vs free-time) ---")
K = min(len(T_h), len(T_f))
for i in range(K):
    print(f"seg{i:2d}: heuristic T={T_h[i]:.4f}  free-time T={T_f[i]:.4f}")
