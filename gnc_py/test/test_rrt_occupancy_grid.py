"""RRT occupancy-grid mode: same map/bounds contract as global A*."""
import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

from sobits_intball2_gnc.guidance.global_planner.rrt_planner import RRTPlanner


def make_grid():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(0.1, 0.05)
    grid.add_box([0.5, 0.0, 0.0], [0.05, 0.15, 0.15])
    return grid


def make_planner(seed):
    return RRTPlanner(
        grid=make_grid().snapshot(), search_bounds=([0.0, -0.5, -0.5], [1.0, 0.5, 0.5]),
        step_size=0.1, goal_tolerance=0.15, goal_bias=0.2, max_iterations=10000, seed=seed,
    )


def test_grid_rrt_detours_and_all_edges_are_free():
    planner = make_planner(3)
    path = planner.plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])
    assert len(path) > 2
    for a, b in zip(path[:-1], path[1:]):
        assert planner._segment_free(a, b)


def test_grid_rrt_is_reproducible_for_a_fixed_seed():
    a = make_planner(5).plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])
    b = make_planner(5).plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])
    np.testing.assert_allclose(a, b)
