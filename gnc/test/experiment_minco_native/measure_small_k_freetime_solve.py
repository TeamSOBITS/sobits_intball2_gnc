#!/usr/bin/env python3
"""EGO-Planner v2はlocal replanをplanning_horizen=5.0m、polyTraj_piece_length=1.5m
(piece_nums=ceil(dist/1.5)、5mなら3〜4piece)というごく小さい問題規模で毎回
free-time L-BFGSを解いている(warm-startも併用)。うちのplan_minco(free-time)が
同程度の小さいKでどれだけ速いか(warm-startなしのcold-start)を、実ルート
(real_waypoints_flat_zeno_route_24pt.txt、0.3m間隔で密なので生データのまま
だとpiece_length相当が0.3mになってしまう)から1.5m間隔相当になるよう間引いて
計測する。C++は一切変更していない、既存バインディングの呼び出しのみ。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import os
import timeit
import numpy as np
import minco_native_py as m

HERE = os.path.dirname(os.path.abspath(__file__))
WPTS_PATH = os.path.join(HERE, "real_waypoints_flat_zeno_route_24pt.txt")

with open(WPTS_PATH) as f:
    n = int(f.readline())
    rows = [list(map(float, f.readline().split())) for _ in range(n)]
rows = np.array(rows)

v0 = [0.0, 0.0, 0.0]
w0 = [0.0, 0.0, 0.0]
VIA_HALF_WIDTH = 0.0
WRENCH_SAFETY_MARGIN = 0.7
N_RUNS = 20

# 元データは0.3m間隔で密（attitude_resample_spacing_m=0.3）なので、
# EGO v2のpolyTraj_piece_length=1.5m相当にするには5点おきに間引く
STRIDE = 5

for stride, label in [(1, "0.3m間隔(元データそのまま)"), (STRIDE, "1.5m間隔相当(EGO v2 piece_length同等)")]:
    thinned = rows[::stride]
    if thinned[-1, 0] != rows[-1, 0]:
        thinned = np.vstack([thinned, rows[-1]])
    for k_take in [2, 3, 4, 5]:
        if k_take + 1 > len(thinned):
            continue
        sub = thinned[:k_take + 1]
        wf = sub.flatten().tolist()
        dist = np.linalg.norm(np.diff(sub[:, :3], axis=0), axis=1).sum()
        try:
            t = timeit.timeit(lambda: m.plan_minco(wf, v0, w0, VIA_HALF_WIDTH, WRENCH_SAFETY_MARGIN), number=N_RUNS)
        except Exception as e:
            print(f"[{label}] K={k_take} dist={dist:.2f}m -> EXCEPTION: {e}")
            continue
        ok, err, T, _, dur = m.plan_minco(wf, v0, w0, VIA_HALF_WIDTH, WRENCH_SAFETY_MARGIN)
        print(f"[{label}] K={k_take} dist={dist:.2f}m -> solve={t/N_RUNS*1000:.2f}ms "
              f"success={ok} error_code={err} traj_duration={dur:.3f}s")
