"""minco_native_py.plan_minco()の回帰テスト。

test/experiment_minco_native/main_attitude.cpp のCLI実験結果（3-waypoint固定、
K=2、q1HalfWidth=0.3、フル解像度wrench envelope使用時: T1,T2=20.0351,14.4978）と、
一般化後の実装（minco_native_py、Kや via点数を任意に受け取れる汎用API）を突き合わせる。
minco_native_pyは別リポジトリ・別colconパッケージなので、未ビルド環境ではskipする。
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
    # なので、フル解像度CLI実験値(20.0351, 14.4978)とは緩い許容で比較する
    assert segment_times[0] == pytest.approx(20.0351, rel=0.05)
    assert segment_times[1] == pytest.approx(14.4978, rel=0.05)


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
