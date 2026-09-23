#!/usr/bin/env python3
"""EGO-v2構造プロトタイプ（`prototype_ego_v2_style_local_replan.py`の
訂正版ロジック、global軌道は必ず``face_travel=True``でdensifyする
——``docs/2026-09-20_ego_planner_v2_aligned_local_replan_design_and_
offline_verification.md``追記2の教訓、face_travel=Falseだとdensifyが
無効化されspurious spatial overshootが起きる）を、複数ジオメトリ・
複数tauへ汎化確認する。本番コード非依存・未変更、ROS/シムなし。

ジオメトリ:
- straight: waypointなし、直線5m
- curve90: 90度ターン(via1点)
- sharp170: 170度に近い急カーブ(via1点、ほぼ逆走に近い迂回)
- zeno4pt: 実TF値の4点経由ルート(過去の全検証と同一、既存の回帰参照用)

tau: 0.0(理想オラクル、遅延なし)・0.3・0.8
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj, quat_exp, quat_log, quat_mul, unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError, MincoTrajectory,
)

Q0 = np.array([0.0, 0.0, 0.0, 1.0])
TARGET_SPEED = 0.5
MAX_ACCEL = 0.0996 / 4.5
PLANNING_HORIZEN = 2.0
REPLAN_PERIOD = 1.0
DT = 0.05
MAX_TICKS = 6000
CONVERGE_TOL = 0.005


def build_geometries():
    geoms = {}

    p0 = np.array([0.0, 0.0, 0.0])
    target = np.array([5.0, 0.0, 0.0])
    geoms["straight_5m"] = (p0, [], target)

    p0 = np.array([0.0, 0.0, 0.0])
    via = np.array([2.5, 0.0, 0.0])
    target = np.array([2.5, 2.5, 0.0])
    geoms["curve90_2.5m_legs"] = (p0, [via], target)

    p0 = np.array([0.0, 0.0, 0.0])
    via = np.array([2.5, 0.0, 0.0])
    target = np.array([2.5 - 2.5 * np.cos(np.radians(10)), 2.5 * np.sin(np.radians(10)), 0.0])
    geoms["sharp170_2.5m_legs"] = (p0, [via], target)

    zeno_p0 = np.array([10.2731, -4.1918, 5.7320])
    zeno_nav_entry = np.array([11.0, -4.3, 5.0])
    zeno_inspection_entry_1 = np.array([10.936, -9.0, 5.0])
    zeno_target = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])
    geoms["zeno4pt_real_tf"] = (zeno_p0, [zeno_nav_entry, zeno_inspection_entry_1], zeno_target)

    return geoms


class LagSimulator:
    def __init__(self, p0, tau):
        self.pos = np.asarray(p0, dtype=float).copy()
        self.tau = float(tau)

    def step(self, cmd_pos, dt):
        if self.tau <= 0.0:
            self.pos = np.asarray(cmd_pos, dtype=float).copy()
        else:
            alpha = dt / self.tau
            self.pos = self.pos + alpha * (np.asarray(cmd_pos, dtype=float) - self.pos)
        return self.pos.copy()


def get_local_target(global_traj, start_pt, global_end_pt, cur_glb_t, planning_horizen):
    total_dur = global_traj.global_total_duration
    t_step = max(planning_horizen / 20.0 / TARGET_SPEED, DT)
    t = cur_glb_t
    while t < total_dur:
        pos_t, vel_t, _, _ = global_traj.sample(t)
        if np.linalg.norm(pos_t - start_pt) >= planning_horizen:
            return pos_t, vel_t, t, False
        t += t_step
    pos_t, vel_t, _, _ = global_traj.sample(total_dur)
    return global_end_pt.copy(), np.zeros(3), total_dur, True


def run(p0, via_points, target, tau):
    waypoints = [p0] + list(via_points) + [target]
    global_traj = MincoTrajectory(
        waypoints, Q0, v0=np.zeros(3), w0=np.zeros(3),
        face_travel=True, via_half_width=0.0, wrench_safety_margin=0.7,
        attitude_resample_spacing_m=0.3,
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
    )

    lag = LagSimulator(p0, tau)
    state_vel = np.zeros(3)
    glb_t_of_lc_tgt = 0.0
    local_traj = None
    local_start_wall_t = 0.0
    touch_goal = False
    fallback_events = []

    history_t, history_pos = [], []
    prev_pos = lag.pos.copy()
    t = 0.0
    for i in range(MAX_TICKS):
        need_replan = (
            local_traj is None
            or (t - local_start_wall_t) >= min(REPLAN_PERIOD, local_traj.global_total_duration)
        )
        if need_replan and not touch_goal:
            local_target_pos, local_target_vel, glb_t_of_lc_tgt, touch_goal = get_local_target(
                global_traj, lag.pos, target, glb_t_of_lc_tgt, PLANNING_HORIZEN
            )
            try:
                local_traj = MincoTrajectory(
                    [lag.pos.copy(), local_target_pos], Q0, v0=state_vel, w0=np.zeros(3),
                    face_travel=False, target_speed=None, max_accel=None,
                )
            except MincoInfeasibleError as e:
                fallback_events.append((t, f"local_infeasible: {e}"))
                break
            local_start_wall_t = t

        tau_local = t - local_start_wall_t
        p_out, v_out, a_out, q_out = local_traj.sample(tau_local)
        true_pos = lag.step(p_out, DT)
        state_vel = (true_pos - prev_pos) / DT
        prev_pos = true_pos.copy()

        history_t.append(t)
        history_pos.append(true_pos.copy())

        remaining = np.linalg.norm(target - true_pos)
        t += DT
        if remaining < CONVERGE_TOL and i > 10:
            return {
                "converged_t": t, "remaining_mm": remaining * 1000,
                "global_duration": global_traj.global_total_duration,
                "fallback_events": fallback_events,
                "history_t": history_t, "history_pos": history_pos,
            }
    return {
        "converged_t": None, "remaining_mm": remaining * 1000,
        "global_duration": global_traj.global_total_duration,
        "fallback_events": fallback_events,
        "history_t": history_t, "history_pos": history_pos,
    }


def min_progress_rate(history_t, history_pos, target, bin_size=10.0):
    ts = np.array(history_t)
    ps = np.array(history_pos)
    rs = np.linalg.norm(target - ps, axis=1)
    worst = 0.0
    b = 0.0
    while b < ts[-1]:
        mask = (ts >= b) & (ts < b + bin_size)
        if mask.sum() >= 2:
            r_bin = rs[mask]
            rate = (r_bin[0] - r_bin[-1]) / bin_size
            worst = min(worst, rate)
        b += bin_size
    return worst


geoms = build_geometries()
for geom_name, (p0, via_points, target) in geoms.items():
    print(f"\n{'=' * 20} {geom_name} {'=' * 20}")
    for tau in [0.0, 0.3, 0.8]:
        result = run(p0, via_points, target, tau)
        worst_rate = min_progress_rate(result["history_t"], result["history_pos"], target) \
            if result["history_t"] else float("nan")
        status = (
            f"converged t={result['converged_t']:.2f}s" if result["converged_t"] is not None
            else f"NOT converged (remaining={result['remaining_mm']:.1f}mm)"
        )
        print(f"tau={tau:.1f}s  global_dur={result['global_duration']:.2f}s  {status}  "
              f"worst_10s_progress_rate={worst_rate:+.4f}m/s  "
              f"fallback={result['fallback_events'][:2]}")
