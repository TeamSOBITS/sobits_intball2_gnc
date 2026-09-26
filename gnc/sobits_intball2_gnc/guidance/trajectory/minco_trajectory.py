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

``face_travel=True``の場合、MINCOを2回solveする（``docs/
2026-09-20_minco_face_travel_corner_divergence_root_cause_and_fix.md``）:
1回目は上記の通り「MINCOが実際に位置を解く前」の直線ベースの姿勢waypointで
solveし、2回目はその1回目の解けた位置経路の実接線（``sample()``の``v``）で
姿勢waypointを再導出してsolveし直す。1回目だけだと、直線ベースで決め打ちした
姿勢目標と、実際にMINCOが解く（コーナーを滑らかに丸めた）位置経路の接線が
コーナー付近で一致せず、setpoint自身の姿勢がsetpoint自身の速度方向を
最大60度近く向かない、という不具合になる（``static``/TOPP-RA側で過去に一度
見つかって直したのと同じ設計上のアンチパターンの再発、``docs/archive/achieved/
2026-08-28_attitude_waypoint_premature_rotation_root_cause.md``）。2回目の
solveで実測上は不動点に収束し、所要時間への影響は1%未満（オフライン検証は
上記md参照）。
"""
import time

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    rotvec_rates_to_body_rates,
    unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

_DEGENERATE_TANGENT_THRESHOLD = 1e-9
_N_DIMS = 6
_N_COEFFS = 6


class MincoInfeasibleError(ValueError):
    """``plan_minco`` reported ``success=False`` (最適化がwrench envelope制約を
    満たす解に収束しなかった、またはC++側で例外が発生した).

    呼び出し側は捕捉してフォールバックまたはabortすべき、"genuine
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
            waypointの数だけしか経由点をdensifyしない。正の値を渡すと、
            各waypoint間をこの間隔以下になるよう直線分割してから
            ``plan_minco``に渡す（``via_half_width``も同じ値が全分割点に
            適用されるので、``0.0``と組み合わせれば位置経路の形は変えずに
            経由点のみ密にできる）。``face_travel``の値には依存しない
            （以前は``face_travel=True``のときしか効かない結合バグが
            あったが解消済み、``docs/
            2026-09-20_ego_v2_style_replan_migration_plan.md``参照）。
            ``face_travel=True``のときは区間が長いほど飛行中に姿勢が進行
            方向から外れていく問題（``docs/
            2026-08-30_static_minco_face_travel_gap.md``追記4参照）への
            対処にもなる。分割点数だけ``K``（区間数）が増えるため、solve
            時間は増える。
        wrench_safety_margin: ロード済みのwrench envelopeをこの係数
            （``(0, 1]``）で縮小してから制約評価する。``1.0``（既定）は
            無効化（従来の挙動と同一）。``static``（TOPPRA）パスの
            ``guidance.wrench_envelope_safety_margin``と同じ、フィードバック
            余力確保のためのマージンをMINCO側にも適用できるようにしたもの
            （``docs/2026-08-30_static_minco_face_travel_gap.md`` 追記2）。
        target_speed, max_accel: 両方とも``None``でない場合、``plan_minco``
            （セグメント時間も自由変数、``INITIAL_SEGMENT_TIME``一律スタート）
            の代わりに``plan_minco_heuristic_time``（弧長比配分のヒューリス
            ティックT＋fixed-T solve＋wrench違反時の解析的伸長ループをC++
            内で完結、``docs/2026-09-01_replanning_minco_v4_production_port_plan.md``
            Phase 1）を使う。デフォルト``None``（両方Noneのまま）は既存の
            ``plan_minco``経路を使う既存挙動と完全に同一——オプトインの
            新経路であり、呼び出し側が明示的に指定しない限り挙動は変わらない。
        body_frame_wrench: ``True``なら``q0``をC++に渡し、wrench envelope（機体座標）を
            機体座標の力で評価する。``False``（既定）は従来通りreference系の加速度をそのまま当てる。

    Raises:
        MincoInfeasibleError: ``plan_minco``/``plan_minco_heuristic_time``が
            ``success=False``を返した場合。
    """

    def __init__(self, position_waypoints, q0, v0=None, w0=None,
                 forward_axis=(1.0, 0.0, 0.0), face_travel=True,
                 via_half_width=0.3, attitude_resample_spacing_m=None,
                 wrench_safety_margin=1.0, target_speed=None, max_accel=None,
                 a0=None, v_tail=None, body_frame_wrench=False):
        if (target_speed is None) != (max_accel is None):
            raise ValueError(
                "target_speed and max_accel must be given together (both "
                "None selects the existing plan_minco path, both non-None "
                "selects plan_minco_heuristic_time)"
            )
        if target_speed is not None and (a0 is not None or v_tail is not None):
            raise ValueError("a0/v_tail are only supported on the free-time plan_minco path")
        a0 = None if a0 is None else [float(c) for c in a0]
        v_tail = None if v_tail is None else [float(c) for c in v_tail]
        position_waypoints = np.asarray(position_waypoints, dtype=float)
        if position_waypoints.ndim != 2 or position_waypoints.shape[1] != 3 \
                or position_waypoints.shape[0] < 2:
            raise ValueError("position_waypoints must have shape (N, 3), N>=2")

        self._q0 = np.asarray(q0, dtype=float)
        wrench_q0 = [float(c) for c in self._q0] if body_frame_wrench else None
        v0 = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float)
        w0 = np.zeros(3) if w0 is None else np.asarray(w0, dtype=float)

        if attitude_resample_spacing_m is not None:
            position_waypoints = self._densify(
                position_waypoints, float(attitude_resample_spacing_m)
            )

        rotvecs = self._waypoint_rotvecs(
            position_waypoints, self._q0, forward_axis, face_travel
        )

        # wall-clock, not sim time: this measures actual solve compute cost
        # (the sim clock doesn't advance while this synchronous call blocks
        # the node's spin loop anyway, so it couldn't measure this even in
        # principle) -- a passive diagnostic, not used for any control/
        # timing decision, so CLAUDE.mdのreal-time禁止の対象外。
        solve_t0 = time.perf_counter()
        segment_times, coeffs, duration = self._solve(
            position_waypoints, rotvecs, v0, w0, via_half_width,
            wrench_safety_margin, target_speed, max_accel, a0, v_tail, wrench_q0
        )

        if face_travel:
            # Pass 2 (module docstring): reseed rotvecs from pass 1's own
            # solved position path's actual local tangent instead of the
            # pre-solve straight-line direction, then re-solve once.
            cum_times = np.concatenate([[0.0], np.cumsum(segment_times)])
            n_segments = len(segment_times)
            pos_coeffs = coeffs[:, 0:3, :]
            actual_directions = np.zeros_like(position_waypoints)
            for i in range(1, len(position_waypoints)):
                seg = self._segment_index_for(cum_times, n_segments, cum_times[i])
                tau = cum_times[i] - cum_times[seg]
                actual_directions[i] = evaluate_vector(pos_coeffs[seg], tau, order=1)
            rotvecs = self._rotvecs_from_directions(
                actual_directions, self._q0, forward_axis
            )
            segment_times, coeffs, duration = self._solve(
                position_waypoints, rotvecs, v0, w0, via_half_width,
                wrench_safety_margin, target_speed, max_accel, a0, v_tail, wrench_q0
            )

        self._set_solution(segment_times, coeffs, duration,
                           time.perf_counter() - solve_t0, len(position_waypoints))

    @classmethod
    def from_rotvec_waypoints(cls, position_waypoints, rotvecs, q0, v0, rotvec_rate0,
                              a0, rotvec_accel0, v_tail=None, via_half_width=0.0,
                              wrench_safety_margin=1.0, max_vel=None,
                              warm_start_segment_times=None, body_frame_wrench=False,
                              obstacle_grid=None, obstacle_touch_goal=False,
                              obstacle_clearance_soft=0.5):
        """Free-time ``plan_minco`` solve with caller-given ``q0``-relative
        ``rotvecs`` (``rotvecs[0]`` is the head attitude) and head rotvec
        rate/accel, for a segment that starts mid-rotation (the constructor
        always starts at ``q0`` at rest). ``max_vel=None`` disables the speed cap.
        ``obstacle_grid`` (``minco_native_py.OccupancyGrid``) runs EGO-Planner v2's
        rebound loop in the solve; a collision left over raises ``MincoInfeasibleError``."""
        position_waypoints = np.asarray(position_waypoints, dtype=float)
        rotvecs = np.asarray(rotvecs, dtype=float)
        if position_waypoints.ndim != 2 or position_waypoints.shape[1] != 3 \
                or position_waypoints.shape[0] < 2 or rotvecs.shape != position_waypoints.shape:
            raise ValueError("position_waypoints/rotvecs must both have shape (N, 3), N>=2")
        self = cls.__new__(cls)
        self._q0 = np.asarray(q0, dtype=float)
        wrench_q0 = [float(c) for c in self._q0] if body_frame_wrench else None
        solve_t0 = time.perf_counter()
        segment_times, coeffs, duration = cls._solve(
            position_waypoints, rotvecs, np.asarray(v0, dtype=float),
            np.asarray(rotvec_rate0, dtype=float), float(via_half_width),
            wrench_safety_margin, None, None, [float(c) for c in a0],
            None if v_tail is None else [float(c) for c in v_tail], wrench_q0,
            rot_a0=[float(c) for c in rotvec_accel0],
            max_vel=-1.0 if max_vel is None else float(max_vel),
            warm_start_T=None if warm_start_segment_times is None
            else [float(t) for t in warm_start_segment_times],
            **({} if obstacle_grid is None else dict(
                grid=obstacle_grid, obstacle_touch_goal=bool(obstacle_touch_goal),
                obstacle_clearance_soft=float(obstacle_clearance_soft))),
        )
        self._set_solution(segment_times, coeffs, duration,
                           time.perf_counter() - solve_t0, len(position_waypoints))
        return self

    def _set_solution(self, segment_times, coeffs, duration, solve_wall_seconds, num_waypoints):
        self.solve_wall_seconds = solve_wall_seconds
        self.num_waypoints = num_waypoints
        self._segment_times = segment_times
        self._cum_times = np.concatenate([[0.0], np.cumsum(self._segment_times)])
        self._pos_coeffs = coeffs[:, 0:3, :]
        self._rot_coeffs = coeffs[:, 3:6, :]
        self._duration = float(duration)

    @classmethod
    def _solve(cls, position_waypoints, rotvecs, v0, w0, via_half_width,
               wrench_safety_margin, target_speed, max_accel, a0=None, v_tail=None,
               wrench_q0=None, **plan_minco_kwargs):
        """Build ``waypoints_flat`` from ``position_waypoints``/``rotvecs`` and run
        one :func:`_call_minco` solve. Returns ``(segment_times, coeffs, duration)``,
        ``coeffs`` already reshaped to ``(n_segments, _N_DIMS, _N_COEFFS)``.
        Called twice by ``__init__`` when ``face_travel`` (module docstring)."""
        waypoints_flat = []
        for pos, rv in zip(position_waypoints, rotvecs):
            waypoints_flat.extend(float(c) for c in pos)
            waypoints_flat.extend(float(c) for c in rv)
        success, error_code, segment_times, coeffs_flat, duration = cls._call_minco(
            waypoints_flat, v0.tolist(), w0.tolist(), via_half_width,
            wrench_safety_margin, target_speed, max_accel, a0, v_tail, wrench_q0,
            **plan_minco_kwargs
        )
        if not success:
            raise MincoInfeasibleError(
                "plan_minco%s failed (error_code=%d) for %d waypoints"
                % ("_heuristic_time" if target_speed is not None else "",
                   error_code, len(position_waypoints))
            )
        segment_times = np.asarray(segment_times, dtype=float)
        n_segments = len(segment_times)
        coeffs = np.asarray(coeffs_flat, dtype=float).reshape(
            n_segments, _N_DIMS, _N_COEFFS
        )
        return segment_times, coeffs, float(duration)

    @staticmethod
    def _call_minco(waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
                     target_speed, max_accel, a0=None, v_tail=None, wrench_q0=None,
                     **plan_minco_kwargs):
        """``minco_native_py``への単一の呼び出し口（モジュール docstring参照）。
        いずれ（別の"Phase 2"、``docs/archive/achieved/
        2026-08-30_minco_attitude_torque_status_and_next_steps.md``）この
        関数の中身をIPC呼び出しに差し替える計画とは無関係——
        ``target_speed``/``max_accel``（``None``でなければ）は、呼び出す
        ネイティブ関数を``plan_minco``から``plan_minco_heuristic_time``へ
        切り替えるためのもの（``docs/
        2026-09-01_replanning_minco_v4_production_port_plan.md`` Phase 1）。"""
        import minco_native_py  # 遅延import: 拡張未ビルド環境でもこのモジュール自体はimportできるように
        if target_speed is None:
            return minco_native_py.plan_minco(
                waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
                a0=a0, v_tail=v_tail, q0=wrench_q0, **plan_minco_kwargs,
            )
        return minco_native_py.plan_minco_heuristic_time(
            waypoints_flat, v0, w0, target_speed, max_accel,
            via_half_width, wrench_safety_margin, q0=wrench_q0
        )

    @staticmethod
    def _densify(position_waypoints, spacing_m):
        """各区間を``spacing_m``以下の間隔になるよう等分割し、元のwaypointは
        分割境界としてそのまま残す（経路の直線形状は変えない、姿勢のseed点や
        global軌道の空間的な滑らかさのためにvia点を増やす前処理。
        ``face_travel``の値には依存しない——姿勢を使わない場合でも
        密なvia点は経路形状の滑らかさに寄与する、
        docs/2026-09-20_ego_v2_style_replan_migration_plan.md参照）。"""
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
        """``__init__``のpass 1（module docstring）: 各waypointの``q0``相対
        回転ベクトルを、まだMINCOが位置を解く前の**直線**方向（waypoint間の
        位置差分）からface-travelヒューリスティックで導出する（``via_pos``
        通過付近でこの直線方向と実際に解けた位置経路の接線がズレる問題への
        対処が``__init__``のpass 2、``_rotvecs_from_directions``参照）。
        """
        n = len(position_waypoints)
        if not face_travel:
            return np.zeros((n, 3))
        directions = np.zeros_like(position_waypoints)
        directions[1:] = position_waypoints[1:] - position_waypoints[:-1]
        return MincoTrajectory._rotvecs_from_directions(directions, q0, forward_axis)

    @staticmethod
    def _rotvecs_from_directions(directions, q0, forward_axis, rv_head=None):
        """``_waypoint_rotvecs``（pass 1、直線方向）と``__init__``のpass 2
        （実際に解けた位置経路の接線、``sample()``の``v``）が共有する
        face-travelの中核ロジック（``ToppraTrajectory._dense_travel_rotvecs``
        と同じ考え方、waypointごとに1回だけ計算する版）。``directions[0]``は
        使わない（サンプル0は``rv_head``、省略時は``q0``・rotvec 0のまま、
        ``ToppraTrajectory``の``initial_q_des``規約と同じ）。

        ``compute_q_des``自体は絶対姿勢（reference frame）を返すため、
        ``sample()``が期待する``q0``相対のrotvecにするには``quat_conj(q0)``を
        掛けてから``quat_log``する必要がある（``_dense_travel_rotvecs``と同じ）。
        これを忘れると``q0``が単位姿勢から離れているほど姿勢が大きく破綻する
        （``docs/2026-08-30_static_minco_face_travel_gap.md``参照）。

        各サンプルのrotvecは直前サンプルに対してunwrapする
        (:func:`~sobits_intball2_gnc.control.utils.quat_math.unwrap_rotvec`)。
        ``quat_log``単体では出力が``[0, pi]``にクランプされるため、``q0``
        からの累積回転が180度を超えるルート（鋭角ターンが複数連続するなど）
        では、その境界をまたぐ1サンプルだけ回転軸が反転し、後段のスプライン
        フィットが破綻する
        （``docs/2026-08-31_multi_via_waypoints_static_test_near_dock_anomaly.md``、
        ``ToppraTrajectory._dense_travel_rotvecs``と同じ不具合）。
        """
        n = len(directions)
        rotvecs = np.zeros((n, 3))
        q0 = np.asarray(q0, dtype=float)
        if rv_head is not None:
            rotvecs[0] = rv_head
        q_prev = quat_mul(q0, quat_exp(rotvecs[0]))
        for i in range(1, n):
            q_prev = compute_q_des(
                directions[i], q_prev, _DEGENERATE_TANGENT_THRESHOLD, forward_axis
            )
            raw_rotvec = quat_log(quat_mul(quat_conj(q0), q_prev))
            rotvecs[i] = unwrap_rotvec(raw_rotvec, rotvecs[i - 1])
        return rotvecs

    @staticmethod
    def _segment_index_for(cum_times, n_segments, t):
        """Same lookup as the instance ``_segment_index``, but usable in
        ``__init__`` before ``self._cum_times``/``self._segment_times`` exist
        (pass 2's reseed needs to sample pass 1's own solved coefficients)."""
        idx = int(np.searchsorted(cum_times, t, side="right")) - 1
        return min(max(idx, 0), n_segments - 1)

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

    def sample_rotvec_derivatives(self, t):
        """Return ``(rv, rv_vel, rv_accel)`` at time ``t`` -- the ``q0``-
        relative rotation vector and its first two derivatives (``sample()``
        only returns the absolute quaternion ``q``, not enough to recover a
        rotational analog of "velocity"/"acceleration" for a caller that
        needs them, e.g. the replanning_minco v4 local layer's rotation
        boundary condition, ``docs/
        2026-09-01_replanning_minco_v4_production_port_plan.md`` Phase 4).
        Same clamping as ``sample()``."""
        t = min(max(float(t), 0.0), self._duration)
        seg_idx = self._segment_index(t)
        tau = t - self._cum_times[seg_idx]
        rv = evaluate_vector(self._rot_coeffs[seg_idx], tau, order=0)
        rv_vel = evaluate_vector(self._rot_coeffs[seg_idx], tau, order=1)
        rv_accel = evaluate_vector(self._rot_coeffs[seg_idx], tau, order=2)
        return rv, rv_vel, rv_accel

    def sample_body_angular(self, t):
        """Return body-frame ``(omega, omega_dot)`` of ``sample(t)``'s ``q``;
        zero from the end on, where ``sample()`` holds ``q`` fixed."""
        if float(t) >= self._duration:
            return np.zeros(3), np.zeros(3)
        return rotvec_rates_to_body_rates(*self.sample_rotvec_derivatives(t))

    def _segment_index(self, t):
        return self._segment_index_for(self._cum_times, len(self._segment_times), t)
