import numpy as np
import pytest

from experiment_jaxa_smoothing_offline import (
    METHODS, corner_spline, distinct, make_curve, dense_samples, fig4_case, fig4_paper_case,
    OmplSimplifier, ompl_path, smoothing_spline, tracking_switch,
)
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import (
    bspline_waypoints, tracking_point,
)


@pytest.mark.parametrize('method', METHODS)
def test_dense_sampling_bounds_and_endpoints(method):
    case = fig4_case()
    curve, _, _ = make_curve(case['raw'], method, case['grid'], case['bounds'])
    _, dense = dense_samples(curve)
    np.testing.assert_allclose(dense[[0, -1]], case['raw'][[0, -1]], atol=1e-10)
    assert np.linalg.norm(np.diff(dense, axis=0), axis=1).max() <= .01 + 1e-12


def test_reference_curve_matches_production():
    case = fig4_case()
    curve, _, length = make_curve(case['raw'], METHODS[0], case['grid'], case['bounds'])
    for spacing in (.5, .1):
        count = int(np.ceil(length / spacing)) + 1
        np.testing.assert_allclose(curve(np.linspace(0, 1, count)),
                                   bspline_waypoints(case['raw'], spacing), atol=1e-10)


def test_switch_metric_matches_native_tracker():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    positions = np.array([[.49, .1, 0.], [.51, .1, 0.], [1., .49, 0.], [1., .51, 0.]])
    nearest = np.argmin(np.linalg.norm(points[None] - positions[:, None], axis=2), axis=1)
    index = np.minimum(nearest, len(points) - 2)
    targets = np.array([tracking_point(points, p, .11)[0] for p in positions])
    expected = np.linalg.norm(np.diff(targets, axis=0), axis=1)[index[1:] != index[:-1]].max()
    assert tracking_switch(points, positions) == pytest.approx(expected)


@pytest.mark.parametrize('points', [np.empty((0, 3)), np.zeros((2, 3)), [[np.nan, 0., 0.], [1., 0., 0.]]])
def test_reject_degenerate_input(points):
    with pytest.raises(ValueError):
        distinct(points)


def test_representative_control_points_stay_inside_control_polygon_box():
    # Clamped B-spline convex-hull property: unlike interpolation it cannot bulge outward.
    case = fig4_case()
    curve, points, _ = make_curve(case['raw'], 'representative_control_points', case['grid'], case['bounds'])
    _, dense = dense_samples(curve)
    assert len(points) < len(case['raw'])
    assert np.all(dense >= points.min(axis=0) - 1e-9) and np.all(dense <= points.max(axis=0) + 1e-9)
    np.testing.assert_allclose(dense[[0, -1]], case['raw'][[0, -1]], atol=1e-10)


def test_corner_spline_keeps_edges_straight_and_bounds_corner_cut():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [3., 2., 0.]])
    curve, _, _, waypoints = corner_spline(points, .4)
    _, dense = dense_samples(curve)
    np.testing.assert_allclose(waypoints[[0, -1]], points[[0, -1]], atol=1e-10)
    assert np.allclose(curve(np.linspace(0, 1, 1001))[:, 2], 0.)
    assert len(waypoints) == 1 + 3 + 2 * 6  # 3 straight ends, 2 corners x 6 samples
    np.testing.assert_allclose(dense[[0, -1]], points[[0, -1]], atol=1e-10)
    on_first_edge = dense[dense[:, 0] <= 1.6 - 1e-9]
    np.testing.assert_allclose(on_first_edge[:, 1], 0., atol=1e-10)
    # Second corner trims to half of the 1 m edge; cut depth is trim*sin(turn/2)/2.
    for corner, trim in (([2., 0., 0.], .4), ([2., 2., 0.], .4)):
        depth = np.linalg.norm(dense - corner, axis=1).min()
        assert depth == pytest.approx(trim * np.sin(np.pi / 4) / 2, abs=2e-3)


def test_smoothing_spline_pins_ends_and_averages_fig4_zigzag():
    case = fig4_paper_case()
    curve, _, _, waypoints = smoothing_spline(case['raw'], .3)
    _, dense = dense_samples(curve)
    np.testing.assert_allclose(dense[[0, -1]], case['raw'][[0, -1]], atol=1e-10)
    np.testing.assert_allclose(waypoints[[0, -1]], case['raw'][[0, -1]], atol=1e-10)
    assert np.linalg.norm(np.diff(waypoints, axis=0), axis=1).max() <= .2 + 1e-12
    # The zigzag spans x = 0.45..1.35 m in rows 9-13; the averaged curve stays well inside it.
    zigzag = dense[(dense[:, 1] > .75) & (dense[:, 1] < 1.95)]
    assert np.ptp(zigzag[:, 0]) < .5


def test_ompl_simplify_keeps_ends_validity_and_never_lengthens():
    case = fig4_paper_case()
    _, states, length, waypoints = ompl_path(case['raw'], case['grid'], case['bounds'])
    raw = case['raw']
    np.testing.assert_allclose(states[[0, -1]], raw[[0, -1]], atol=1e-12)
    # OMPL guarantees validity only at its own motion resolution; a finer check may
    # find slight touches (the paper's "may collide" before its final check).
    checker = OmplSimplifier(case['grid'], case['bounds'], 0)
    assert all(checker.motion(a, b) for a, b in zip(waypoints[:-1], waypoints[1:]))
    assert length <= np.linalg.norm(np.diff(raw, axis=0), axis=1).sum() + 1e-9
    again = ompl_path(raw, case['grid'], case['bounds'])[1]
    np.testing.assert_allclose(again, states)  # seeded from the input path
