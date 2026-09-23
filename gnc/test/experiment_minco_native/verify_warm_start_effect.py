"""warm_start_qvia/warm_start_T引数の効果をオフライン検証する。

docs/2026-09-20_ego_v2_style_replan_migration_plan.md 「やるべき内容3」
（warm start）向けにminco_native_py.plan_mincoへ追加した引数の効果測定。
本番コード（replanning_minco_v2_tracker.py等）は未変更。

シナリオ: EGO-v2 style local replanを模して、同じvia点構成（数waypoint）に
対して境界条件（v0, 経路形状）を毎tick少しずつずらしながら繰り返しsolveし、
(a) warm startなし（毎回ゼロ初期化）と(b) 前回解をwarm startした場合の
solve時間・反復回数を比較する。
"""
import time

import numpy as np

import minco_native_py


def make_waypoints(offset: float) -> list[float]:
    # head, via x2, tail の4waypoint。offsetでvia点をわずかに動かし
    # 「replanごとに経路がわずかに変わる」状況を模す。
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.3 + 0.01 * offset, 0.0),
        (2.0, -0.3 - 0.01 * offset, 0.0),
        (3.0, 0.0, 0.0),
    ]
    flat: list[float] = []
    for p in pts:
        flat.extend(p)
        flat.extend((0.0, 0.0, 0.0))  # rx,ry,rz = 0（姿勢はスコープ外）
    return flat


def extract_qvia(coeffs_flat: list[float], num_via: int) -> list[float]:
    # coeffs_flat layout: segment x [pos_dim(0..2) x c0..c5][rot_dim(0..2) x c0..c5]
    # via点i(0-indexed)はsegment(i+1)の始点=c0係数(t=0での値)に等しい。
    qvia: list[float] = []
    for i in range(num_via):
        seg = i + 1
        for d in range(3):
            c0_idx = seg * 36 + d * 6 + 0
            qvia.append(coeffs_flat[c0_idx])
    return qvia


def run_sequence(n_ticks: int, use_warm_start: bool) -> tuple[list[float], list[int]]:
    times: list[float] = []
    prev_qvia: list[float] | None = None
    prev_T: list[float] | None = None
    for k in range(n_ticks):
        waypoints = make_waypoints(float(k))
        v0 = [0.3, 0.0, 0.0]
        w0 = [0.0, 0.0, 0.0]

        kwargs = {}
        if use_warm_start and prev_qvia is not None:
            kwargs["warm_start_qvia"] = prev_qvia
            kwargs["warm_start_T"] = prev_T

        t0 = time.perf_counter()
        success, error_code, segment_times, coeffs_flat, duration = minco_native_py.plan_minco(
            waypoints, v0, w0, 0.3, 1.0, **kwargs
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        assert success, f"tick {k}: infeasible (error_code={error_code})"

        n = len(waypoints) // 6
        num_via = n - 2
        prev_qvia = extract_qvia(coeffs_flat, num_via)
        prev_T = list(segment_times)

    return times, []


def main() -> None:
    n_ticks = 30
    times_cold = run_sequence(n_ticks, use_warm_start=False)[0]
    times_warm = run_sequence(n_ticks, use_warm_start=True)[0]

    # 最初のtickはwarm start適用不可（prev解なし）なので除外して比較
    cold = np.array(times_cold[1:])
    warm = np.array(times_warm[1:])

    print(f"n_ticks(excl. first)={len(cold)}")
    print(f"cold  : mean={cold.mean()*1e3:.3f}ms  median={np.median(cold)*1e3:.3f}ms  "
          f"total={cold.sum()*1e3:.2f}ms")
    print(f"warm  : mean={warm.mean()*1e3:.3f}ms  median={np.median(warm)*1e3:.3f}ms  "
          f"total={warm.sum()*1e3:.2f}ms")
    speedup = cold.sum() / warm.sum() if warm.sum() > 0 else float("nan")
    print(f"speedup (cold_total/warm_total) = {speedup:.3f}x")


if __name__ == "__main__":
    main()
