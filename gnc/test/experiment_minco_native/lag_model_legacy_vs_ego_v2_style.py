#!/usr/bin/env python3
"""追従遅れ・誤差を模した簡易一次遅れモデル付きオフラインテスト。

``docs/2026-09-20_minco_global_replan_freetime_switch_offline_
investigation.md``の「次にやるべきこと」項目1（理想追従オラクル(事実33-34)
では再現しない、実シムのみで起きる「終盤ほぼ完全停止」(事実27-31)の根本
原因を、追従遅れを模した簡易遅延モデルで再現するか切り分ける）と、
``docs/2026-09-20_ego_planner_v2_aligned_local_replan_design_and_
offline_verification.md``の「次にやるべきこと」項目1（新構造=EGO-v2
アーキテクチャでも同じ劣化が起きないか）を、同じ簡易遅延モデルで両方
検証する。本番コード非依存・未変更、ROS/シムなし。

簡易遅延モデル（``LagSimulator``）: 「真の」機体位置・姿勢が、tracker/
プロトタイプが出力した指令(setpoint)に対して一次遅れ ``d(true)/dt =
(cmd - true) / tau`` で追従する、というだけの単純なモデル
（``trajectory_controller``の実際のPD制御・duty配分・fan物理を一切
再現しない、「遅れがあるとどうなるか」の傾向だけを見るための最小限の
モデル。依頼者の言う「簡易」の通り）。各tickでtrue位置/姿勢を更新し、
次のreplanの境界条件・pose_fnの戻り値には**この遅れた真値**を使う
（理想オラクルは指令をそのまま真値として使っていたのに対する変更点は
ここだけ）。

2つの構造を同じ遅延モデル・同じシナリオ（zenoルート、実TF値、
``docs/2026-09-18_minco_stretch_fix_sim_verification_facts.md``と同一）
で比較する:

- legacy: ``ReplanningMincoV2Tracker``（本番と同一クラス、変更なし）を
  ``target_speed=None, max_accel=None``で構築 -- 実シムで劣化が観測された
  のと同じ設定（fact 27, minco_freetime=true相当）。
- ego_v2_style: ``prototype_ego_v2_style_local_replan.py``のロジックを
  関数化したもの（global一度きり+local頻繁free-time LBFGS、Hermite層なし）。

tau（遅れの時定数）は複数値でスイープし、単一値への依存を避ける。
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
    MincoTrajectory,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    ReplanningMincoV2Tracker,
)

p0 = np.array([10.2731, -4.1918, 5.7320])
q0 = np.array([-0.3646, 0.4827, 0.7962, 0.0069])
nav_entry = np.array([11.0, -4.3, 5.0])
inspection_entry_1 = np.array([10.936, -9.0, 5.0])
capture_point_2 = np.array([10.2692137671802, -9.435036380998612, 5.235784547749638])

WAYPOINTS = [p0, nav_entry, inspection_entry_1, capture_point_2]
DT = 0.05
MAX_TICKS = 6000
CONVERGE_TOL = 0.005
PLANNING_HORIZEN = 2.0
REPLAN_PERIOD = 1.0
TAUS = [0.3, 0.8]


class LagSimulator:
    """真の位置・姿勢が指令に一次遅れで追従する最小限モデル(module docstring)。"""

    def __init__(self, p0, q0, tau):
        self.pos = np.asarray(p0, dtype=float).copy()
        self.rotvec = np.zeros(3)
        self.q0 = np.asarray(q0, dtype=float)
        self.tau = float(tau)
        self._prev_rotvec_obs = np.zeros(3)

    def step(self, cmd_pos, cmd_q, dt):
        alpha = dt / self.tau
        cmd_rotvec_raw = quat_log(quat_mul(quat_conj(self.q0), np.asarray(cmd_q, dtype=float)))
        cmd_rotvec = unwrap_rotvec(cmd_rotvec_raw, self._prev_rotvec_obs)
        self._prev_rotvec_obs = cmd_rotvec
        self.pos = self.pos + alpha * (np.asarray(cmd_pos, dtype=float) - self.pos)
        self.rotvec = self.rotvec + alpha * (cmd_rotvec - self.rotvec)
        return self.pos.copy(), self.true_quat()

    def true_quat(self):
        return quat_mul(self.q0, quat_exp(self.rotvec))


def speed_bins(history_t, history_pos, bin_size=10.0):
    history_t = np.array(history_t)
    history_pos = np.array(history_pos)
    t_max = history_t[-1]
    rows = []
    bin_start = 0.0
    while bin_start < t_max:
        mask = (history_t >= bin_start) & (history_t < bin_start + bin_size)
        if mask.sum() >= 2:
            pos_bin = history_pos[mask]
            t_bin = history_t[mask]
            diffs = np.linalg.norm(np.diff(pos_bin, axis=0), axis=1)
            dts = np.diff(t_bin)
            speeds = diffs / dts
            rows.append((bin_start, bin_start + bin_size, speeds.mean(), speeds.max()))
        bin_start += bin_size
    return rows


def run_legacy(tau):
    lag = LagSimulator(p0, q0, tau)

    def pose_fn():
        return lag.pos.copy(), lag.true_quat(), pose_fn.t

    pose_fn.t = 0.0

    def tf_fresh_fn(stamp):
        return True

    try:
        tracker = ReplanningMincoV2Tracker(
            p0, capture_point_2, pose_fn, tf_fresh_fn, q0,
            target_speed=None, max_accel=None,
            route_waypoints=np.vstack([nav_entry, inspection_entry_1]),
            via_half_width=0.0, wrench_safety_margin=0.7,
            attitude_resample_spacing_m=0.3,
        )
    except MincoInfeasibleError as e:
        return {"error": f"initial construct MincoInfeasibleError: {e}"}

    history_t, history_pos = [], []
    t = 0.0
    fallback_events = []
    for i in range(MAX_TICKS):
        t += DT
        pose_fn.t = t
        p_out, v_out, a_out, q_out = tracker.sample(t)
        lag.step(p_out, q_out, DT)
        history_t.append(t)
        history_pos.append(lag.pos.copy())
        if tracker.last_fallback_reason is not None:
            fallback_events.append((t, tracker.last_fallback_reason))
        remaining = np.linalg.norm(capture_point_2 - lag.pos)
        if remaining < CONVERGE_TOL and i > 10:
            return {
                "converged_t": t, "remaining_mm": remaining * 1000,
                "history_t": history_t, "history_pos": history_pos,
                "fallback_events": fallback_events,
            }
    return {
        "converged_t": None, "remaining_mm": remaining * 1000,
        "history_t": history_t, "history_pos": history_pos,
        "fallback_events": fallback_events,
    }


def get_local_target(global_traj, start_pt, global_end_pt, cur_glb_t, planning_horizen, target_speed):
    total_dur = global_traj.global_total_duration
    t_step = max(planning_horizen / 20.0 / max(target_speed, 1e-6), DT)
    t = cur_glb_t
    while t < total_dur:
        pos_t, vel_t, _, _ = global_traj.sample(t)
        if np.linalg.norm(pos_t - start_pt) >= planning_horizen:
            return pos_t, vel_t, t, False
        t += t_step
    pos_t, vel_t, _, _ = global_traj.sample(total_dur)
    return global_end_pt.copy(), np.zeros(3), total_dur, True


def run_ego_v2_style(tau):
    target_speed = 0.5
    max_accel = 0.0996 / 4.5

    global_traj = MincoTrajectory(
        WAYPOINTS, q0, v0=np.zeros(3), w0=np.zeros(3),
        face_travel=False, via_half_width=0.0, wrench_safety_margin=0.7,
        attitude_resample_spacing_m=0.3,
        target_speed=target_speed, max_accel=max_accel,
    )

    lag = LagSimulator(p0, q0, tau)
    state_vel = np.zeros(3)
    glb_t_of_lc_tgt = 0.0
    local_traj = None
    local_start_wall_t = 0.0
    touch_goal = False
    fallback_events = []

    history_t, history_pos = [], []
    t = 0.0
    for i in range(MAX_TICKS):
        need_replan = (
            local_traj is None
            or (t - local_start_wall_t) >= min(REPLAN_PERIOD, local_traj.global_total_duration)
        )
        if need_replan and not touch_goal:
            local_target_pos, local_target_vel, glb_t_of_lc_tgt, touch_goal = get_local_target(
                global_traj, lag.pos, capture_point_2, glb_t_of_lc_tgt, PLANNING_HORIZEN, target_speed
            )
            try:
                local_traj = MincoTrajectory(
                    [lag.pos.copy(), local_target_pos], q0, v0=state_vel, w0=np.zeros(3),
                    face_travel=False, target_speed=None, max_accel=None,
                )
            except MincoInfeasibleError as e:
                fallback_events.append((t, f"local_infeasible: {e}"))
                break
            local_start_wall_t = t

        tau_local = t - local_start_wall_t
        p_out, v_out, a_out, q_out = local_traj.sample(tau_local)
        true_pos, true_q = lag.step(p_out, q_out, DT)
        # 遅れた真の速度は有限差分で近似（実際のtrajectory_controllerが
        # TF差分で速度推定しているのと同じ考え方、メモリ[[navigation_
        # independence_requirement]]周辺の設計を踏襲）。
        state_vel = (true_pos - history_pos[-1]) / DT if history_pos else state_vel

        history_t.append(t)
        history_pos.append(true_pos.copy())

        remaining = np.linalg.norm(capture_point_2 - true_pos)
        t += DT
        if remaining < CONVERGE_TOL and i > 10:
            return {
                "converged_t": t, "remaining_mm": remaining * 1000,
                "history_t": history_t, "history_pos": history_pos,
                "fallback_events": fallback_events,
            }
    return {
        "converged_t": None, "remaining_mm": remaining * 1000,
        "history_t": history_t, "history_pos": history_pos,
        "fallback_events": fallback_events,
    }


for tau in TAUS:
    print(f"\n{'=' * 20} tau={tau}s {'=' * 20}")
    for label, run_fn in [("legacy(free-time global + Hermite local)", run_legacy),
                          ("ego_v2_style(local receding-horizon LBFGS)", run_ego_v2_style)]:
        print(f"\n--- {label} ---")
        result = run_fn(tau)
        if "error" in result:
            print(f"FAILED: {result['error']}")
            continue
        if result["converged_t"] is not None:
            print(f"converged at t={result['converged_t']:.2f}s, remaining={result['remaining_mm']:.2f}mm")
        else:
            print(f"did NOT converge within {MAX_TICKS} ticks, remaining={result['remaining_mm']:.2f}mm")
        if result["fallback_events"]:
            print(f"fallback/infeasible events: {result['fallback_events'][:5]}"
                  f"{' ...' if len(result['fallback_events']) > 5 else ''}")
        else:
            print("no fallback/infeasible events")
        for b0, b1, mean_v, max_v in speed_bins(result["history_t"], result["history_pos"]):
            print(f"  t={b0:5.0f}-{b1:5.0f}s: mean={mean_v:.4f} max={max_v:.4f} m/s")
