"""Unit tests for guidance/global_planner/rrt_planner.py (plain-value, no ROS)."""
import numpy as np

from sobits_intball2_gnc.guidance.global_planner.rrt_planner import RRTPlanner


def test_free_space_returns_direct_pair():
    planner = RRTPlanner(seed=0)
    start, goal = [0.0, 0.0, 0.0], [1.0, 2.0, 0.0]
    path = planner.plan(start, goal)
    assert len(path) == 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


def test_obstacle_forces_detour_and_reaches_goal():
    planner = RRTPlanner(
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        step_size=0.2,
        goal_tolerance=0.15,
        max_iterations=20000,
        seed=42,
    )
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    obstacle = [(np.array([0.5, 0.0, 0.0]), 0.3)]

    path = planner.plan(start, goal, obstacles=obstacle)

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    for a, b in zip(path[:-1], path[1:]):
        assert not planner._segment_collides(a, b, obstacle)


def test_reproducible_with_same_seed():
    start, goal = [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]
    obstacle = [(np.array([0.5, 0.5, 0.0]), 0.3)]

    path_a = RRTPlanner(seed=7, max_iterations=20000).plan(start, goal, obstacles=obstacle)
    path_b = RRTPlanner(seed=7, max_iterations=20000).plan(start, goal, obstacles=obstacle)

    assert len(path_a) == len(path_b)
    for a, b in zip(path_a, path_b):
        assert np.allclose(a, b)


def test_unreachable_goal_raises():
    # Goal fully enclosed by a large obstacle -> no collision-free connection possible.
    planner = RRTPlanner(
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        max_iterations=500,
        seed=1,
    )
    start = [0.0, 0.0, 0.0]
    goal = [1.0, 0.0, 0.0]
    obstacle = [(np.array([1.0, 0.0, 0.0]), 1.5)]  # radius covers start too, but segment check dominates
    try:
        planner.plan(start, goal, obstacles=obstacle)
        assert False, "expected RuntimeError for unreachable goal"
    except RuntimeError:
        pass
