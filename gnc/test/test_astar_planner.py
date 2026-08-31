"""Unit tests for guidance/global_planner/astar_planner.py (plain-value, no ROS)."""
import numpy as np

from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner


def test_same_cell_returns_direct_pair():
    planner = AStarPlanner(resolution=0.1)
    start, goal = [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]
    path = planner.plan(start, goal)
    assert len(path) == 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


def test_free_space_path_is_direct():
    planner = AStarPlanner(resolution=0.2)
    start, goal = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    path = planner.plan(start, goal)
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    # No obstacles: A* should find the straight diagonal-step line, i.e. the
    # minimum number of grid steps (5 cells of 0.2 = 1.0m -> 6 waypoints).
    assert len(path) == 6


def test_obstacle_forces_detour():
    resolution = 0.2
    planner = AStarPlanner(resolution=resolution)
    start, goal = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]

    # Block every cell on the direct line between start and goal.
    blocked = {(i, 0, 0) for i in range(1, 5)}
    path = planner.plan(start, goal, obstacles=blocked)

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    for point in path[1:-1]:
        idx = tuple(int(round(c / resolution)) for c in point)
        assert idx not in blocked


def test_unreachable_goal_raises():
    planner = AStarPlanner(resolution=0.2, search_margin=0)
    # Fully enclose the goal cell so no neighbor can reach it.
    goal_idx = (5, 0, 0)
    enclosing = {
        tuple(g + o for g, o in zip(goal_idx, offset))
        for offset in [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
    }
    start = [0.0, 0.0, 0.0]
    goal = [c * 0.2 for c in goal_idx]
    try:
        planner.plan(start, goal, obstacles=enclosing)
        assert False, "expected RuntimeError for unreachable goal"
    except RuntimeError:
        pass
