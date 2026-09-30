"""OccupancyGrid mode of AStarPlanner: pure offline tests, no ROS node."""
import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

from sobits_intball2_gnc.guidance.global_planner.astar_planner import (
    AStarPlanner,
    AStarPlanningError,
    GoalOccupiedError,
    SearchBoundsError,
    StartOccupiedError,
)


RESOLUTION = 0.1
BOUNDS = (np.array([0.0, -0.5, -0.5]), np.array([1.0, 0.5, 0.5]))


def make_grid(inflation=0.0):
    return sobits_intball2_gnc_cpp.OccupancyGrid(RESOLUTION, inflation)


def make_planner(grid):
    return AStarPlanner(resolution=RESOLUTION, grid=grid.snapshot(), search_bounds=BOUNDS)


def path_cells(path):
    return [tuple(np.floor(np.asarray(point) / RESOLUTION).astype(int)) for point in path]


def test_occupancy_mode_uses_floor_cells_and_cell_centers():
    planner = make_planner(make_grid())
    start, goal = np.array([0.01, 0.01, 0.01]), np.array([0.91, 0.01, 0.01])
    path = planner.plan(start, goal)

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    assert path_cells(path) == [(i, 0, 0) for i in range(10)]
    assert np.allclose(path[1], [0.15, 0.05, 0.05])


def test_inflated_box_forces_a_safe_detour_inside_bounds():
    grid = make_grid(inflation=0.1)
    grid.add_box([0.5, 0.0, 0.0], [0.01, 0.01, 0.01])
    planner = make_planner(grid)
    path = planner.plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])

    assert any(np.linalg.norm(np.asarray(point)[1:] - [0.05, 0.05]) > 1e-9
               for point in path[1:-1])
    for point in path:
        assert not grid.inflated_occupied(list(point))
        assert np.all(np.asarray(point) >= BOUNDS[0])
        assert np.all(np.asarray(point) <= BOUNDS[1])


def test_start_goal_and_bounds_fail_with_specific_errors():
    start_grid = make_grid()
    start_grid.add_box([0.05, 0.05, 0.05], [0.01, 0.01, 0.01])
    with pytest.raises(StartOccupiedError):
        make_planner(start_grid).plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])

    goal_grid = make_grid()
    goal_grid.add_box([0.95, 0.05, 0.05], [0.01, 0.01, 0.01])
    with pytest.raises(GoalOccupiedError):
        make_planner(goal_grid).plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])

    with pytest.raises(SearchBoundsError):
        make_planner(make_grid()).plan([-0.01, 0.05, 0.05], [0.95, 0.05, 0.05])


def test_no_corridor_fails_and_result_is_deterministic():
    grid = make_grid()
    grid.add_box([0.5, 0.0, 0.0], [0.01, 0.5, 0.5])
    planner = make_planner(grid)
    with pytest.raises(AStarPlanningError):
        planner.plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])

    detour_grid = make_grid(inflation=0.1)
    detour_grid.add_box([0.5, 0.0, 0.0], [0.01, 0.01, 0.01])
    planner_a = make_planner(detour_grid)
    planner_b = make_planner(detour_grid)
    path_a = planner_a.plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])
    path_b = planner_b.plan([0.05, 0.05, 0.05], [0.95, 0.05, 0.05])
    np.testing.assert_allclose(path_a, path_b)


def test_diagonal_cannot_cut_between_occupied_voxels():
    grid = make_grid()
    # A 26-neighbor search could go directly from (0, 0, 0) to (1, 1, 0),
    # but that edge touches both occupied face-adjacent cells at their corner.
    grid.add_box([0.15, 0.05, 0.05], [0.01, 0.01, 0.01])
    grid.add_box([0.05, 0.15, 0.05], [0.01, 0.01, 0.01])
    planner = AStarPlanner(
        resolution=RESOLUTION,
        grid=grid.snapshot(),
        search_bounds=([0.0, 0.0, 0.0], [0.2, 0.2, 0.1]),
    )
    with pytest.raises(AStarPlanningError):
        planner.plan([0.05, 0.05, 0.05], [0.15, 0.15, 0.05])


def test_six_connected_mode_is_available_and_not_shorter_than_26_connected():
    grid = make_grid()
    start, goal = [0.05, 0.05, 0.05], [0.95, 0.45, 0.45]
    six = AStarPlanner(RESOLUTION, grid=grid.snapshot(), search_bounds=BOUNDS, connectivity=6)
    diagonal = AStarPlanner(RESOLUTION, grid=grid.snapshot(), search_bounds=BOUNDS, connectivity=26)
    path_six = six.plan(start, goal)
    path_diagonal = diagonal.plan(start, goal)
    length = lambda path: sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
    assert length(path_six) >= length(path_diagonal)
