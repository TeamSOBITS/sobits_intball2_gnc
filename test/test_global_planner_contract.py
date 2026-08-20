"""Contract test shared by every BaseGlobalPlanner implementation.

See docs/architecture_guidelines.md 5 節: every swap-point with 2+ concrete
implementations gets one parametrized test asserting the behavioral
properties all implementations must satisfy, not just structural ones. Add a
new planner to ``PLANNER_FACTORIES`` when it's implemented.
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.rrt_planner import RRTPlanner

PLANNER_FACTORIES = {
    "astar": lambda: AStarPlanner(resolution=0.2),
    "rrt": lambda: RRTPlanner(seed=0, max_iterations=20000),
}


@pytest.mark.parametrize("name", PLANNER_FACTORIES)
def test_planner_starts_and_ends_at_requested_points(name):
    planner = PLANNER_FACTORIES[name]()
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 0.5])

    path = planner.plan(start, goal)

    assert len(path) >= 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


@pytest.mark.parametrize("name", PLANNER_FACTORIES)
def test_planner_accepts_no_obstacles(name):
    planner = PLANNER_FACTORIES[name]()
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    path = planner.plan(start, goal, obstacles=None)

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


@pytest.mark.parametrize("name", PLANNER_FACTORIES)
def test_planner_zero_distance_is_trivial(name):
    planner = PLANNER_FACTORIES[name]()
    point = np.array([0.3, -0.1, 0.2])

    path = planner.plan(point, point)

    assert np.allclose(path[0], point)
    assert np.allclose(path[-1], point)
