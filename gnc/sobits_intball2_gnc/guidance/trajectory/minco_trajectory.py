#!/usr/bin/env python3
"""MINCO-based combined position+attitude trajectory (ROS-agnostic).

Phase 1（``docs/archive/achieved/2026-08-30_minco_attitude_torque_status_and_next_steps.md``）
の呼び出し口。``minco_native_py``（pybind11拡張、``minco_solver.cpp``が
実際の最適化を行う）の ``plan_minco(waypoints_flat, v0, w0)`` を1箇所だけから
叩く（Phase 2でIPC呼び出しに差し替える際、この呼び出し部分だけを局所変更
すれば済むようにするための設計、同docの「開発の2段階」節参照）。

:class:`~sobits_intball2_gnc.guidance.trajectory.toppra_trajectory.
ToppraTrajectory` と同じ ``sample(t) -> (p, v, a, q)`` 契約で返す
（同docの「MINCOも同じ型を踏襲する」節）。``ReplanningTrajectoryTracker``
が今使っている位置のみの``Trajectory``（face-travelで毎tick姿勢を後付け
計算する）とは異なり、姿勢もMINCOの最適化対象に含まれる6-DOF軌道を扱う。

姿勢waypoint（``plan_minco``に渡す各waypointの回転ベクトル）は、呼び出し側
（``ReplanningTrajectoryTracker._replan``）が位置waypointしか持たないため、
:class:`ToppraTrajectory` の ``_dense_travel_rotvecs`` と同じ face-travel
ヒューリスティック（waypoint間の位置差分方向を向く）で、この中で自前に
導出する。
"""
import time

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

_DEGENERATE_TANGENT_THRESHOLD = 1e-9
_N_DIMS = 6
_N_COEFFS = 6


class MincoInfeasibleError(ValueError):
    """``plan_minco`` reported ``success=False`` (最適化がwrench envelope制約を
    満たす解に収束しなかった、またはC++側で例外が発生した).

    ``SegmentTimeInfeasibleError``（``base_segment_time_allocator.py``）と
    同じ位置付け: 呼び出し側は捕捉してフォールバックすべき、"genuine
    kinematic dead end" 相当のエラー。
    """


class MincoTrajectory:
    """MINCO姿勢/トルク統合軌道。コンストラクタで一度だけ最適化を実行する
    （``ReplanningTrajectoryTracker``で使う場合、re-planごとに新しい
    インスタンスを作る想定 -- ``Trajectory.replace_coeffs``のような
    in-place更新はしない）。

    Args:
        position_waypoints: 位置waypoint列、shape ``(N, 3)``、``N>=2``。
        q0: 姿勢の基準クォータニオン ``[x,y,z,w]``（``ToppraTrajectory``と
            同じ規約、姿勢は``q0``相対の回転ベクトルとしてMINCOに渡す）。
        v0: head（waypoints[0]）の初速度、shape ``(3,)``。``None``は零。
        w0: head の初角速度、shape ``(3,)``。``None``は零。
        forward_axis: face-travel姿勢waypoint導出で機首を向ける機体軸。
        face_travel: ``False``の場合、姿勢waypointは全て``q0``に固定
            （回転ベクトル0）。
        via_half_width: 位置via点の自由変数box半幅[m]（``minco_solver.cpp``の
            ``VIA_HALF_WIDTH``だったハードコード定数を呼び出し側から渡せる
            ようにしたもの。``0.0``なら経由点を厳密に固定＝TOPPRAの通過点
            拘束と同等になる。``docs/2026-08-30_static_minco_face_travel_gap.md``
            追記3参照）。
        attitude_resample_spacing_m: ``None``（既定）なら従来通り与えた
            waypointの数だけしかface-travel姿勢をseedしない。正の値を渡すと、
            各waypoint間をこの間隔以下になるよう直線分割し、その分割点でも
            face-travel姿勢を計算してから``plan_minco``に渡す（``via_half_width``
            も同じ値が全分割点に適用されるので、``0.0``と組み合わせれば
            位置経路の形は変えずに姿勢のみ密にできる）。区間が長いほど飛行中に
            姿勢が進行方向から外れていく問題（``docs/
            2026-08-30_static_minco_face_travel_gap.md``追記4参照）への対処。
            分割点数だけ``K``（区間数）が増えるため、solve時間は増える。

    Raises:
        MincoInfeasibleError: ``plan_minco``が``success=False``を返した場合。
    """

    def __init__(self, position_waypoints, q0, v0=None, w0=None,
                 forward_axis=(1.0, 0.0, 0.0), face_travel=True,
                 via_half_width=0.3, attitude_resample_spacing_m=None):
        position_waypoints = np.asarray(position_waypoints, dtype=float)
        if position_waypoints.ndim != 2 or position_waypoints.shape[1] != 3 \
                or position_waypoints.shape[0] < 2:
            raise ValueError("position_waypoints must have shape (N, 3), N>=2")

        self._q0 = np.asarray(q0, dtype=float)
        v0 = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float)
        w0 = np.zeros(3) if w0 is None else np.asarray(w0, dtype=float)

        if attitude_resample_spacing_m is not None and face_travel:
            position_waypoints = self._densify(
                position_waypoints, float(attitude_resample_spacing_m)
            )

        rotvecs = self._waypoint_rotvecs(
            position_waypoints, self._q0, forward_axis, face_travel
        )

        waypoints_flat = []
        for pos, rv in zip(position_waypoints, rotvecs):
            waypoints_flat.extend(float(c) for c in pos)
            waypoints_flat.extend(float(c) for c in rv)

        # wall-clock, not sim time: this measures actual solve compute cost
        # (the sim clock doesn't advance while this synchronous call blocks
        # the node's spin loop anyway, so it couldn't measure this even in
        # principle) -- a passive diagnostic, not used for any control/
        # timing decision, so CLAUDE.mdのreal-time禁止の対象外。
        solve_t0 = time.perf_counter()
        success, error_code, segment_times, coeffs_flat, duration = self._call_minco(
            waypoints_flat, v0.tolist(), w0.tolist(), via_half_width
        )
        self.solve_wall_seconds = time.perf_counter() - solve_t0
        self.num_waypoints = len(position_waypoints)
        if not success:
            raise MincoInfeasibleError(
                "plan_minco failed (error_code=%d) for %d waypoints"
                % (error_code, len(position_waypoints))
            )

        self._segment_times = np.asarray(segment_times, dtype=float)
        self._cum_times = np.concatenate([[0.0], np.cumsum(self._segment_times)])
        n_segments = len(self._segment_times)
        coeffs = np.asarray(coeffs_flat, dtype=float).reshape(
            n_segments, _N_DIMS, _N_COEFFS
        )
        self._pos_coeffs = coeffs[:, 0:3, :]
        self._rot_coeffs = coeffs[:, 3:6, :]
        self._duration = float(duration)

    @staticmethod
    def _call_minco(waypoints_flat, v0, w0, via_half_width):
        """``minco_native_py``への単一の呼び出し口（モジュール docstring参照）。
        Phase 2ではこの関数の中身だけをIPC呼び出しに差し替える。"""
        import minco_native_py  # 遅延import: 拡張未ビルド環境でもこのモジュール自体はimportできるように
        return minco_native_py.plan_minco(waypoints_flat, v0, w0, via_half_width)

    @staticmethod
    def _densify(position_waypoints, spacing_m):
        """各区間を``spacing_m``以下の間隔になるよう等分割し、元のwaypointは
        分割境界としてそのまま残す（経路の直線形状は変えない、姿勢のseed点
        だけを増やすための前処理）。"""
        if spacing_m <= 0.0:
            raise ValueError("attitude_resample_spacing_m must be > 0")
        dense = [position_waypoints[0]]
        for i in range(1, len(position_waypoints)):
            p_prev = position_waypoints[i - 1]
            p_next = position_waypoints[i]
            seg_len = np.linalg.norm(p_next - p_prev)
            n_sub = max(1, int(np.ceil(seg_len / spacing_m)))
            for k in range(1, n_sub + 1):
                dense.append(p_prev + (p_next - p_prev) * (k / n_sub))
        return np.array(dense)

    @staticmethod
    def _waypoint_rotvecs(position_waypoints, q0, forward_axis, face_travel):
        """各waypointの``q0``相対回転ベクトルをface-travelヒューリスティックで
        導出する（``ToppraTrajectory._dense_travel_rotvecs``と同じ考え方、
        密サンプルではなくwaypointごとに1回だけ計算する版）。

        ``compute_q_des``自体は絶対姿勢（reference frame）を返すため、
        ``sample()``が期待する``q0``相対のrotvecにするには``quat_conj(q0)``を
        掛けてから``quat_log``する必要がある（``_dense_travel_rotvecs``と同じ）。
        これを忘れると``q0``が単位姿勢から離れているほど姿勢が大きく破綻する
        （``docs/2026-08-30_static_minco_face_travel_gap.md``参照）。
        """
        n = len(position_waypoints)
        rotvecs = np.zeros((n, 3))
        if not face_travel:
            return rotvecs
        q0 = np.asarray(q0, dtype=float)
        q_prev = q0.copy()
        for i in range(1, n):
            direction = position_waypoints[i] - position_waypoints[i - 1]
            q_prev = compute_q_des(
                direction, q_prev, _DEGENERATE_TANGENT_THRESHOLD, forward_axis
            )
            rotvecs[i] = quat_log(quat_mul(quat_conj(q0), q_prev))
        return rotvecs

    @property
    def global_total_duration(self):
        return self._duration

    def sample(self, t):
        """Return ``(p, v, a, q)`` at time ``t``（``duration``でクランプし、
        以降は終端状態を保持）。``ToppraTrajectory.sample()``と同じ契約。"""
        t = min(max(float(t), 0.0), self._duration)
        seg_idx = self._segment_index(t)
        tau = t - self._cum_times[seg_idx]

        p = evaluate_vector(self._pos_coeffs[seg_idx], tau, order=0)
        v = evaluate_vector(self._pos_coeffs[seg_idx], tau, order=1)
        a = evaluate_vector(self._pos_coeffs[seg_idx], tau, order=2)
        rv = evaluate_vector(self._rot_coeffs[seg_idx], tau, order=0)
        q = quat_mul(self._q0, quat_exp(rv))
        return p, v, a, q

    def _segment_index(self, t):
        idx = int(np.searchsorted(self._cum_times, t, side="right")) - 1
        return min(max(idx, 0), len(self._segment_times) - 1)
