"""K>1化（事実51）とwarm start（事実53）を組み合わせた効果をオフライン検証する。

docs/2026-09-20_ego_planner_v2_aligned_local_replan_design_and_offline_
verification.md 事実53の追記用。事実51の「密なglobal(zenoルート)・localの
via点数K sweep」と同じジオメトリ・パラメータを使い、各K（1, 3）についてwarm
start無し/有りを比較する。`prototype_ego_v2_style_local_replan.py`の
receding-horizon構造を踏襲するが、local側は`MincoTrajectory`ラッパーを
経由せず`minco_native_py.plan_minco`を直接呼ぶ（ラッパーがwarm_start_qvia/
warm_start_Tを素通しする経路をまだ持たないため、本番コード
`minco_trajectory.py`は一切変更しない）。

本番コード非依存・実験用スタンドアロン。
"""
import sys
import time

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np

import minco_native_py
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

_TARGET_SPEED = 0.5
_MAX_ACCEL = 0.0996 / 4.5

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

WAYPOINTS = [p0, nav_entry, inspection_entry_1, capture_point_2]
PLANNING_HORIZEN = 2.0
REPLAN_PERIOD = 1.0
DT = 0.05
MAX_TICKS = 6000
CONVERGE_TOL = 0.005
VIA_HALF_WIDTH = 0.3
WRENCH_SAFETY_MARGIN = 0.7


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


def build_waypoints_flat(p_now, target_pos, num_via):
    # ゴール近傍でlocal区間長がplanning_horizenより大幅に短くなると、
    # via点分割が密集しすぎてsegment時間が過小になりinfeasibleになりうる
    # ため、区間長がnum_via+1個のセグメントを賄うのに十分（>=0.3m/セグメント
    # 目安）でなければ分割数を落とす。
    seg_len = np.linalg.norm(target_pos - p_now) / (num_via + 1)
    effective_num_via = num_via
    while effective_num_via > 0 and seg_len < 0.3:
        effective_num_via -= 1
        seg_len = np.linalg.norm(target_pos - p_now) / (effective_num_via + 1)

    pts = [p_now]
    for i in range(1, effective_num_via + 1):
        frac = i / (effective_num_via + 1)
        pts.append(p_now + frac * (target_pos - p_now))
    pts.append(target_pos)
    flat = []
    for p in pts:
        flat.extend(p.tolist())
        flat.extend((0.0, 0.0, 0.0))
    return flat, effective_num_via


def extract_qvia(coeffs_flat, num_via):
    qvia = []
    for i in range(num_via):
        seg = i + 1
        for d in range(3):
            qvia.append(coeffs_flat[seg * 36 + d * 6 + 0])
    return qvia


def sample_local(segment_times, coeffs_flat, tau):
    cum = 0.0
    seg = 0
    local_t = tau
    for i, T in enumerate(segment_times):
        if tau < cum + T or i == len(segment_times) - 1:
            seg = i
            local_t = max(0.0, min(tau - cum, T))
            break
        cum += T
    pos = np.zeros(3)
    vel = np.zeros(3)
    for d in range(3):
        c = coeffs_flat[seg * 36 + d * 6: seg * 36 + d * 6 + 6]
        p = 0.0
        tp = 1.0
        for k in range(6):
            p += c[k] * tp
            tp *= local_t
        vel_coeffs = [c[1], 2 * c[2], 3 * c[3], 4 * c[4], 5 * c[5]]
        v = 0.0
        tp = 1.0
        for coef in vel_coeffs:
            v += coef * tp
            tp *= local_t
        pos[d] = p
        vel[d] = v
    return pos, vel


def solve_local(p_now, v_now, target_pos, num_via, warm_qvia, warm_T):
    waypoints_flat, effective_num_via = build_waypoints_flat(p_now, target_pos, num_via)
    kwargs = {}
    if warm_qvia is not None and warm_T is not None:
        kwargs["warm_start_qvia"] = warm_qvia
        kwargs["warm_start_T"] = warm_T
    t0 = time.perf_counter()
    success, error_code, segment_times, coeffs_flat, duration = minco_native_py.plan_minco(
        waypoints_flat, v_now.tolist(), [0.0, 0.0, 0.0],
        VIA_HALF_WIDTH, WRENCH_SAFETY_MARGIN, **kwargs
    )
    elapsed = time.perf_counter() - t0
    if not success:
        raise RuntimeError(f"local replan infeasible (error_code={error_code})")
    return segment_times, coeffs_flat, duration, elapsed, effective_num_via


def run(num_via, use_warm_start, global_traj):
    state_pos = p0.copy()
    state_vel = np.zeros(3)
    glb_t_of_lc_tgt = 0.0
    local_segment_times = None
    local_coeffs = None
    local_start_wall_t = 0.0
    touch_goal = False
    replan_count = 0
    total_solve_time = 0.0
    prev_qvia = None
    prev_T = None

    t = 0.0
    for i in range(MAX_TICKS):
        local_dur = sum(local_segment_times) if local_segment_times else 0.0
        need_replan = (
            local_segment_times is None
            or (t - local_start_wall_t) >= min(REPLAN_PERIOD, local_dur)
        )
        if need_replan and not touch_goal:
            local_target_pos, _local_target_vel, glb_t_of_lc_tgt, touch_goal = get_local_target(
                global_traj, state_pos, capture_point_2, glb_t_of_lc_tgt, PLANNING_HORIZEN
            )
            wqv = prev_qvia if use_warm_start else None
            wT = prev_T if use_warm_start else None
            try:
                local_segment_times, local_coeffs, _duration, elapsed, actual_num_via = solve_local(
                    state_pos, state_vel, local_target_pos, num_via, wqv, wT
                )
            except RuntimeError as e:
                print(f"  t={t:.2f}s local replan failed: {e}")
                break
            total_solve_time += elapsed
            replan_count += 1
            prev_qvia = extract_qvia(local_coeffs, actual_num_via)
            prev_T = list(local_segment_times)
            local_start_wall_t = t

        tau = t - local_start_wall_t
        state_pos, state_vel = sample_local(local_segment_times, local_coeffs, tau)

        remaining = np.linalg.norm(capture_point_2 - state_pos)
        t += DT
        if remaining < CONVERGE_TOL and i > 10:
            break
    else:
        t = float("nan")

    return {
        "converge_t": t,
        "replan_count": replan_count,
        "total_solve_time_ms": total_solve_time * 1e3,
        "mean_solve_time_ms": (total_solve_time / replan_count * 1e3) if replan_count else float("nan"),
        "final_error_mm": np.linalg.norm(state_pos - capture_point_2) * 1e3,
    }


def main():
    print("=== global軌道構築（一度きり、heuristic_time）===")
    # face_travel=True必須: minco_trajectory.py:133のdensify分岐は
    # face_travelがTrueの場合のみattitude_resample_spacing_mを適用する
    # （face_travel=Falseだと密化されず粗いwaypointのままspurious overshoot
    # を招く、事実50/事実54訂正・migration_plan.md「やるべき内容1」参照）。
    # 姿勢自体は今回のlocal replan検証では使わないが、globalの空間的な
    # 滑らかさのためにdensifyだけは必要。
    global_traj = MincoTrajectory(
        WAYPOINTS, q0, v0=np.zeros(3), w0=np.zeros(3),
        face_travel=True, via_half_width=0.0, wrench_safety_margin=WRENCH_SAFETY_MARGIN,
        attitude_resample_spacing_m=0.3,
        target_speed=_TARGET_SPEED, max_accel=_MAX_ACCEL,
    )
    print(f"global duration={global_traj.global_total_duration:.3f}s\n")

    for num_via in (0, 2):
        k = num_via + 1
        print(f"--- K={k} (via{num_via}点) ---")
        for use_warm in (False, True):
            label = "warm" if use_warm else "cold"
            result = run(num_via, use_warm, global_traj)
            print(f"  {label}: converge_t={result['converge_t']:.2f}s  "
                  f"replans={result['replan_count']}  "
                  f"total_solve={result['total_solve_time_ms']:.2f}ms  "
                  f"mean_solve={result['mean_solve_time_ms']:.3f}ms  "
                  f"final_err={result['final_error_mm']:.3f}mm")
        print()


if __name__ == "__main__":
    main()
