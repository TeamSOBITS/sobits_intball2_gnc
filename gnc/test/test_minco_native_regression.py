"""minco_native_py.plan_minco()の回帰テスト。

test/experiment_minco_native/main_attitude.cpp のCLI実験結果（3-waypoint固定、
K=2、q1HalfWidth=0.3、フル解像度wrench envelope使用時: 元の参照値
T1,T2=20.0351,14.4978）と、一般化後の実装（minco_native_py、Kや via点数を
任意に受け取れる汎用API）を突き合わせる。minco_native_pyは別リポジトリ・
別colconパッケージなので、未ビルド環境ではskipする。

2026-09-17: `minco_solver.cpp`のaccRotバグ修正（回転ベクトルの2階微分を
そのままomega_dotとして扱っていたのを、SO(3)の右ヤコビアン補正込みの
正しい式に修正、docs/archive/achieved/
2026-09-17_accrot_jacobian_bug_offline_verification.md）で本番の物理計算
自体が変わったため、上記の元の参照値（`main_attitude.cpp`が今も使っている
修正前の素朴な近似`omega_dot=rDdot`で生成されたもの）はもう正しい比較対象
ではない。`main_attitude.cpp`自体は当時のまま更新していない（古い実験の
再現性を優先、修正後の値で上書きしない）ため、以下は修正後の本番出力
（T1,T2=17.5385,11.2856）を新しい期待値として使う。
"""
import numpy as np
import pytest

minco_native_py = pytest.importorskip("minco_native_py")


NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5
RV1 = np.array([0.0, 0.35864857, 1.6255471])


def _flatten_waypoint(pos, rot):
    return list(pos) + list(rot)


def test_three_waypoint_matches_cli_scenario_shape():
    midpoint = 0.5 * (NEAR_DOCK + ABOVE_DOCK)
    bulge = NAV_ENTRY - midpoint
    q1_given = midpoint + BULGE_SCALE * bulge

    waypoints = (
        _flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0])
        + _flatten_waypoint(q1_given, RV1)
        + _flatten_waypoint(ABOVE_DOCK, RV1)
    )

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco(waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    )

    assert success is True
    assert error_code == 0
    assert len(segment_times) == 2
    assert len(coeffs_flat) == 2 * 6 * 6
    assert duration == pytest.approx(sum(segment_times))
    # インストールされたwrench envelopeは面数削減近似（doc記載の許容誤差3.2%以内）
    # なので緩い許容で比較する。値自体はaccRotバグ修正後の本番出力
    # （モジュールdocstring参照、`main_attitude.cpp`の元の参照値20.0351/
    # 14.4978はもう正しい比較対象ではない）。
    assert segment_times[0] == pytest.approx(17.5385, rel=0.05)
    assert segment_times[1] == pytest.approx(11.2856, rel=0.05)


def test_two_waypoint_no_via_point():
    waypoints = _flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0]) + _flatten_waypoint(
        ABOVE_DOCK, RV1
    )

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco(waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    )

    assert success is True
    assert error_code == 0
    assert len(segment_times) == 1
    assert len(coeffs_flat) == 1 * 6 * 6
    assert duration == pytest.approx(segment_times[0])


def test_multi_waypoint_generalizes_segment_count():
    waypoints_pts = [
        (NEAR_DOCK, [0.0, 0.0, 0.0]),
        ([10.98, -3.9, 4.4], [0.0, 0.1, 0.4]),
        ([11.0, -4.1, 4.7], [0.0, 0.2, 0.9]),
        ([10.99, -4.25, 4.9], [0.0, 0.3, 1.3]),
        (ABOVE_DOCK, RV1),
    ]
    waypoints = []
    for pos, rot in waypoints_pts:
        waypoints += _flatten_waypoint(pos, rot)

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco(waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    )

    K = len(waypoints_pts) - 1
    assert success is True
    assert error_code == 0
    assert len(segment_times) == K
    assert len(coeffs_flat) == K * 6 * 6
    assert duration == pytest.approx(sum(segment_times))


def test_malformed_waypoints_flat_length_reports_failure():
    # 6の倍数でない -> C++側でstd::invalid_argument -> success=false, error_code=1
    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    )

    assert success is False
    assert error_code == 1
    assert segment_times == []
    assert coeffs_flat == []


# plan_minco_heuristic_time()の回帰テスト（docs/
# 2026-09-01_replanning_minco_v4_production_port_plan.md Phase 1）。
# gnc/test/experiment_minco_native/bench_v5_multiscenario.cppの4シナリオ
# （経由点数・直行・鋭角ターンをカバー、全PASS確認済み）をそのまま移植。
_TARGET_SPEED = 0.5
_MAX_ACCEL = 0.0996 / 4.5

_BENCH_V5_SCENARIOS = [
    (
        "regression_original",
        np.array([10.997, -4.297, 5.006]),
        [np.array([10.936, -9.000, 5.000]), np.array([10.269, -9.435, 5.236])],
    ),
    (
        "two_via_zigzag",
        np.array([10.997, -4.297, 5.006]),
        [
            np.array([11.5, -6.5, 5.1]),
            np.array([10.5, -8.0, 4.9]),
            np.array([10.269, -9.435, 5.236]),
        ],
    ),
    (
        "direct_no_via_long",
        np.array([10.997, -4.297, 5.006]),
        [np.array([9.5, -12.0, 5.5])],
    ),
    (
        "sharp_turn_single_via",
        np.array([10.997, -4.297, 5.006]),
        [np.array([11.8, -7.0, 5.006]), np.array([9.0, -7.0, 5.006])],
    ),
]


@pytest.mark.parametrize("name,start,route", _BENCH_V5_SCENARIOS)
@pytest.mark.parametrize("init_vy", [0.0, 0.3])
def test_heuristic_time_matches_bench_v5_scenarios(name, start, route, init_vy):
    waypoints = _flatten_waypoint(start, [0.0, 0.0, 0.0])
    for p in route:
        waypoints += _flatten_waypoint(p, [0.0, 0.0, 0.0])
    v0 = [0.0, -abs(init_vy), 0.0]

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco_heuristic_time(
            waypoints, v0, [0.0, 0.0, 0.0],
            target_speed=_TARGET_SPEED, max_accel=_MAX_ACCEL,
        )
    )

    K = len(route)
    assert success is True
    assert error_code == 0
    assert len(segment_times) == K
    assert all(t > 0.0 for t in segment_times)
    assert len(coeffs_flat) == K * 6 * 6
    assert duration == pytest.approx(sum(segment_times))


def test_heuristic_time_invalid_target_speed_reports_failure():
    waypoints = _flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0]) + _flatten_waypoint(
        ABOVE_DOCK, [0.0, 0.0, 0.0]
    )

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco_heuristic_time(
            waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            target_speed=0.0, max_accel=_MAX_ACCEL,
        )
    )

    assert success is False
    assert error_code == 1
    assert segment_times == []
    assert coeffs_flat == []


def test_heuristic_time_invalid_max_accel_reports_failure():
    waypoints = _flatten_waypoint(NEAR_DOCK, [0.0, 0.0, 0.0]) + _flatten_waypoint(
        ABOVE_DOCK, [0.0, 0.0, 0.0]
    )

    success, error_code, segment_times, coeffs_flat, duration = (
        minco_native_py.plan_minco_heuristic_time(
            waypoints, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            target_speed=_TARGET_SPEED, max_accel=0.0,
        )
    )

    assert success is False
    assert error_code == 1
    assert segment_times == []
    assert coeffs_flat == []
