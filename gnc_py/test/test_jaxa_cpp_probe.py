"""Numerical and occupancy checks for the offline-only native prototype."""
import numpy as np
import pytest
import sobits_intball2_gnc_cpp as core
native = pytest.importorskip("jaxa_cpp_probe", reason="build test/jaxa_cpp_probe and add its build directory to PYTHONPATH")
from jaxa_python_reference import (
    RRTStar, bspline_waypoints, path_is_free,
)


@pytest.mark.parametrize("count", [2, 3, 4, 5, 10, 30])
def test_spline_matches_scipy(count):
    points = np.random.RandomState(count).normal(size=(count, 3)).cumsum(axis=0)
    points = np.vstack((points, points[-1]))
    np.testing.assert_allclose(native.spline(points.tolist()), bspline_waypoints(points), atol=1e-10, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 100])
def test_rrt_and_smoothing_match_python(seed):
    grid = core.OccupancyGrid(0.1, 0.0)
    grid.add_box([2, 0, 0], [.2, .4, 1])
    bounds = ([0, -1, -1], [4, 1, 1])
    start, goal = np.array([.5, 0, 0]), np.array([3.5, 0, 0])
    raw = RRTStar(grid, bounds, seed=seed).plan(start, goal)
    cpp_raw, cpp_path, free = native.attempt(grid, start.tolist(), goal.tolist(), *bounds, seed)
    np.testing.assert_allclose(cpp_raw, raw, atol=1e-10, rtol=0)
    np.testing.assert_allclose(cpp_path, bspline_waypoints(raw), atol=1e-10, rtol=0)
    assert path_is_free(cpp_raw, grid, bounds)
    assert free == path_is_free(cpp_path, grid, bounds)


@pytest.mark.parametrize("start,goal,message", [
    ([2, 0, 0], [3.5, 0, 0], "start occupied"),
    ([.5, 0, 0], [2, 0, 0], "goal occupied"),
    ([.5, 0, 0], [3.5, 0, 0], "RRT.*found no path"),
])
def test_blocked_inputs_raise(start, goal, message):
    grid = core.OccupancyGrid(0.1, 0.0)
    grid.add_box([2, 0, 0], [.2, 1.5, 1.5])
    with pytest.raises(RuntimeError, match=message):
        native.attempt(grid, start, goal, [0, -1, -1], [4, 1, 1], 0)
