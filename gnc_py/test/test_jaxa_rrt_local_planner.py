"""Unit tests for the JAXA IAC-22 baseline local planner (jaxa_rrt_local_planner)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import (
    JaxaPlanError,
    JaxaPlannerConfig,
    RRTStar,
    bspline_waypoints,
    path_is_free,
    plan_local_path,
    tracking_point,
)

BOUNDS = ([0.0, -1.0, -1.0], [4.0, 1.0, 1.0])


def make_grid(boxes=()):
    import sobits_intball2_gnc_cpp as core

    grid = core.OccupancyGrid(0.1, 0.0)
    for center, half_extent in boxes:
        grid.add_box(list(center), list(half_extent))
    return grid


def test_tracking_point_on_straight_path_is_lookahead_ahead_of_the_foot():
    path = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    target, i = tracking_point(path, np.array([0.4, 0.1, 0.0]), 0.3)
    assert i == 0
    np.testing.assert_allclose(target, [0.7, 0.0, 0.0], atol=1e-12)


def test_tracking_point_stops_at_the_goal_on_the_last_segment():
    path = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    target, i = tracking_point(path, np.array([1.9, 0.0, 0.0]), 0.3)
    assert i == 1
    np.testing.assert_allclose(target, [2.0, 0.0, 0.0])


def test_bspline_passes_through_points_with_spacing_count_and_drops_duplicates():
    pts = np.array([[0., 0., 0.], [1., 0.5, 0.], [2., 0., 0.], [2., 0., 0.]])
    wp = bspline_waypoints(pts, 0.5)
    np.testing.assert_allclose(wp[0], pts[0])
    np.testing.assert_allclose(wp[-1], pts[-1])
    length = 2 * np.hypot(1.0, 0.5)
    assert len(wp) == int(np.ceil(length / 0.5)) + 1


def test_rrt_star_avoids_a_wall_and_raises_on_occupied_start():
    grid = make_grid([([2.0, 0.0, 0.0], [0.2, 0.6, 1.0])])  # gap only at |y| > 0.6
    path = RRTStar(grid, BOUNDS, seed=0).plan([0.5, 0.0, 0.0], [3.5, 0.0, 0.0])
    assert path_is_free(path, grid, BOUNDS)
    with pytest.raises(JaxaPlanError, match="start occupied"):
        RRTStar(grid, BOUNDS, seed=0).plan([2.0, 0.0, 0.0], [3.5, 0.0, 0.0])


def test_plan_local_path_returns_collision_free_waypoints():
    grid = make_grid([([2.0, 0.0, 0.0], [0.2, 0.4, 1.0])])
    for planner in ("ompl", "rrt"):
        path, info = plan_local_path(np.array([0.5, 0., 0.]), np.array([3.5, 0., 0.]), grid, BOUNDS,
                                     seed=0, config=JaxaPlannerConfig(planner=planner))
        assert path_is_free(path, grid, BOUNDS)
        np.testing.assert_allclose(path[0], [0.5, 0., 0.])
        np.testing.assert_allclose(path[-1], [3.5, 0., 0.])
        assert 1 <= info["attempts"] <= 50 and info["plan_s"] >= 0.0


def test_ompl_plan_is_the_default_and_raises_after_max_attempts():
    assert JaxaPlannerConfig().planner == "ompl"
    wall = make_grid([([2.0, 0.0, 0.0], [0.2, 1.5, 1.5])])  # no way around inside BOUNDS
    with pytest.raises(JaxaPlanError, match="after 3 attempts"):
        plan_local_path([.5, 0, 0], [3.5, 0, 0], wall, BOUNDS, 0,
                        JaxaPlannerConfig(max_attempts=3, ompl_solve_time_s=0.02))
    with pytest.raises(JaxaPlanError, match="start occupied"):
        plan_local_path([2.0, 0, 0], [3.5, 0, 0], wall, BOUNDS, 0, JaxaPlannerConfig())


@pytest.mark.parametrize("count", [2, 3, 4, 5, 10, 30])
def test_native_bspline_matches_frozen_python_reference(count):
    import jaxa_python_reference as reference

    points = np.random.RandomState(count).normal(size=(count, 3)).cumsum(axis=0)
    points = np.vstack((points, points[-1]))
    np.testing.assert_allclose(bspline_waypoints(points), reference.bspline_waypoints(points),
                               atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 100])
def test_native_rrt_and_tracking_match_frozen_python_reference(seed):
    import jaxa_python_reference as reference

    grid = make_grid([([2, 0, 0], [.2, .4, 1])])
    start, goal = np.array([.5, 0, 0]), np.array([3.5, 0, 0])
    raw = RRTStar(grid, BOUNDS, seed=seed).plan(start, goal)
    expected = reference.RRTStar(grid, BOUNDS, seed=seed).plan(start, goal)
    np.testing.assert_allclose(raw, expected, atol=1e-10, rtol=0)
    path = bspline_waypoints(raw)
    for position in ([.5, .2, 0], [2.1, -.3, .1], [3.6, 0, 0]):
        target, index = tracking_point(path, np.array(position), .3)
        ref_target, ref_index = reference.tracking_point(path, np.array(position), .3)
        assert index == ref_index
        np.testing.assert_allclose(target, ref_target, atol=1e-10, rtol=0)
    assert path_is_free(path, grid, BOUNDS) == reference.path_is_free(path, grid, BOUNDS)


@pytest.mark.parametrize("layout", range(5))
@pytest.mark.parametrize("seed", [100, 200, 300])
def test_native_full_plan_matches_reference_across_paper_layouts(layout, seed):
    import sobits_intball2_gnc_cpp as core
    import offline_common as base
    import experiment_jaxa_baseline_offline as experiment
    import jaxa_python_reference as reference

    _, static = core.load_octomap_points(base.MAP)
    grid = core.OccupancyGrid(base.RESOLUTION, base.INFLATION)
    grid.add_points(static)
    scenario = experiment.paper_scenario(layout)
    for center, half in scenario.boxes:
        grid.add_box(center.tolist(), half.tolist())
    config = JaxaPlannerConfig(planner="rrt", max_attempts=5)
    try:
        expected, ref_info = reference.plan_local_path(
            scenario.start, scenario.goal, grid, base.BOUNDS, seed, config)
    except reference.JaxaPlanError as error:
        with pytest.raises(JaxaPlanError) as native_error:
            plan_local_path(scenario.start, scenario.goal, grid, base.BOUNDS, seed, config)
        assert str(native_error.value) == str(error)
    else:
        path, info = plan_local_path(scenario.start, scenario.goal, grid, base.BOUNDS, seed, config)
        assert info["attempts"] == ref_info["attempts"]
        np.testing.assert_allclose(path, expected, atol=1e-10, rtol=0)
        assert path_is_free(path, grid, base.BOUNDS)


@pytest.mark.parametrize("options", [
    {"rrt_step_m": .2, "rrt_radius_m": .4, "rrt_iterations": 300},
    {"rrt_goal_bias": .3, "rrt_goal_tolerance_m": .5, "waypoint_spacing_m": .2, "max_attempts": 2},
])
def test_native_forwards_nondefault_configuration(options):
    import jaxa_python_reference as reference

    grid = make_grid()
    config = JaxaPlannerConfig(planner="rrt", **options)
    start, goal = np.array([.5, 0, 0]), np.array([3.5, 0, 0])
    actual, info = plan_local_path(start, goal, grid, BOUNDS, 0, config)
    expected, ref_info = reference.plan_local_path(start, goal, grid, BOUNDS, 0, config)
    assert info["attempts"] == ref_info["attempts"]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("options", [
    {"rrt_step_m": 0}, {"rrt_radius_m": float("nan")}, {"rrt_iterations": 0},
    {"rrt_goal_bias": 1.1}, {"rrt_goal_tolerance_m": -1},
    {"waypoint_spacing_m": float("inf")}, {"max_attempts": 0},
    {"planner": "prm"}, {"ompl_solve_time_s": 0.0}, {"ompl_solve_time_s": float("nan")},
])
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError, match="configuration"):
        plan_local_path([.5, 0, 0], [3.5, 0, 0], make_grid(), BOUNDS, 0, JaxaPlannerConfig(**options))


def test_same_start_and_goal_returns_stationary_path():
    point = [.5, 0, 0]
    path, info = plan_local_path(point, point, make_grid(), BOUNDS, 0, JaxaPlannerConfig())
    np.testing.assert_allclose(path, [point, point])
    assert info["attempts"] == 1
    target, index = tracking_point(path, point, .3)
    np.testing.assert_allclose(target, point)
    assert index == 0


@pytest.mark.parametrize("start,goal,bounds", [
    ([float("nan"), 0, 0], [3.5, 0, 0], BOUNDS),
    ([.5, 0], [3.5, 0, 0], BOUNDS),
    ([.5, 0, 0], [3.5, 0, 0], ([4, -1, -1], [0, 1, 1])),
])
def test_invalid_coordinates_are_rejected(start, goal, bounds):
    with pytest.raises(ValueError):
        plan_local_path(start, goal, make_grid(), bounds, 0, JaxaPlannerConfig())
