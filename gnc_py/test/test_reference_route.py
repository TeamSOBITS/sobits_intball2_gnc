"""The shared pre-departure route and its position-only min-jerk reference."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.search.reference_route import (
    AStarPlanningError,
    densify,
    plan_reference_route,
    route_is_free,
)
from sobits_intball2_gnc.guidance.trajectory.reference_polynomial import MinJerkReference

core = pytest.importorskip("sobits_intball2_gnc_cpp")

RESOLUTION = 0.05
INFLATION = 0.2
BOUNDS = (np.array([-2.0, -2.0, -2.0]), np.array([6.0, 6.0, 6.0]))
SPEED = 0.15
Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def _grid(points=()):
    grid = core.OccupancyGrid(RESOLUTION, INFLATION)
    points = np.asarray(points, dtype=float)
    if points.size:
        grid.add_points(points.ravel().tolist())
    return grid.snapshot()


def _box(center, half, step=RESOLUTION / 2.0):
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    axes = [np.arange(c - h, c + h + step, step) for c, h in zip(center, half)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


# --- plan_reference_route / route_is_free / densify ---------------------------

def test_empty_space_gives_the_straight_two_point_route():
    route = plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], _grid(), BOUNDS)
    assert len(route) == 2
    np.testing.assert_allclose(route[0], [0.0, 0.0, 0.0], atol=RESOLUTION)
    np.testing.assert_allclose(route[-1], [3.0, 0.0, 0.0], atol=RESOLUTION)


def test_a_box_on_the_chord_is_detoured_and_the_route_stays_free():
    grid = _grid(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    route = plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], grid, BOUNDS)
    assert len(route) > 2, "the straight chord would cross the box"
    assert route_is_free(route, grid, BOUNDS)
    assert not route_is_free([route[0], route[-1]], grid, BOUNDS)


def test_a_route_planned_before_the_box_appeared_is_reported_blocked():
    route = plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], _grid(), BOUNDS)
    assert not route_is_free(route, _grid(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3])), BOUNDS)


def test_a_wall_across_the_search_box_leaves_no_path():
    wall = _box([1.5, 0.0, 0.0], [0.1, 7.0, 7.0], step=RESOLUTION / 2.0)
    with pytest.raises(AStarPlanningError):
        plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], _grid(wall), BOUNDS)


def test_the_route_is_deterministic():
    grid = _grid(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    a = plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], grid, BOUNDS)
    b = plan_reference_route([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], grid, BOUNDS)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("start, goal", [
    ([1.5, 0.0, 0.0], [3.0, 0.0, 0.0]),  # start inside the box
    ([0.0, 0.0, 0.0], [1.5, 0.0, 0.0]),  # goal inside the box
])
def test_an_occupied_end_raises(start, goal):
    grid = _grid(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    with pytest.raises(AStarPlanningError):
        plan_reference_route(start, goal, grid, BOUNDS)


def test_a_goal_outside_the_search_bounds_raises():
    with pytest.raises(AStarPlanningError):
        plan_reference_route([0.0, 0.0, 0.0], [99.0, 0.0, 0.0], _grid(), BOUNDS)


def test_densify_keeps_the_vertices_and_bounds_the_spacing():
    route = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.6, 0.0])]
    points = np.asarray(densify(route, 0.25))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert steps.max() <= 0.25 + 1e-9
    for vertex in route:
        assert np.min(np.linalg.norm(points - vertex, axis=1)) < 1e-9


def test_densify_leaves_a_short_segment_whole():
    assert len(densify([np.zeros(3), np.array([0.1, 0.0, 0.0])], 0.25)) == 2


def test_densify_rejects_a_non_positive_spacing():
    with pytest.raises(ValueError):
        densify([np.zeros(3), np.ones(3)], 0.0)


# --- MinJerkReference ---------------------------------------------------------

def _square_route():
    return densify([np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                    np.array([1.0, 1.0, 0.0])], 0.25)


def test_the_reference_passes_through_every_point():
    points = _square_route()
    ref = MinJerkReference(points, SPEED, Q0)
    starts = np.concatenate([[0.0], np.cumsum(ref._times)])
    for t, expected in zip(starts, points):
        np.testing.assert_allclose(ref.sample(t)[0], expected, atol=1e-7)


def test_the_reference_starts_and_ends_at_rest():
    ref = MinJerkReference(_square_route(), SPEED, Q0)
    for t in (0.0, ref.global_total_duration):
        _p, v, a, _q = ref.sample(t)
        np.testing.assert_allclose(v, np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(a, np.zeros(3), atol=1e-7)


def test_velocity_and_acceleration_are_continuous_across_a_point():
    ref = MinJerkReference(_square_route(), SPEED, Q0)
    t = float(np.cumsum(ref._times)[2])
    for before, after in zip(ref.sample(t - 1e-6)[1:3], ref.sample(t + 1e-6)[1:3]):
        np.testing.assert_allclose(before, after, atol=1e-4)


def test_the_sampled_speed_is_capped():
    ref = MinJerkReference(_square_route(), SPEED, Q0)
    ts = np.linspace(0.0, ref.global_total_duration, 400)
    assert max(np.linalg.norm(ref.sample(t)[1]) for t in ts) <= SPEED + 1e-9


def test_segment_times_are_distance_over_speed_with_doubled_ends():
    points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
              np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
    ref = MinJerkReference(points, SPEED, Q0)
    np.testing.assert_allclose(ref._times, np.array([2.0, 1.0, 2.0]) / SPEED)
    assert ref.num_waypoints == 4


def test_a_single_segment_is_doubled_only_once():
    """move_goal_out_of_obstacle rebuilds the global from two points; doubling
    the first and the last separately would stretch it fourfold."""
    ref = MinJerkReference([np.zeros(3), np.array([1.0, 0.0, 0.0])], SPEED, Q0)
    np.testing.assert_allclose(ref.global_total_duration, 2.0 / SPEED)


def test_sampling_outside_the_duration_is_clamped():
    ref = MinJerkReference(_square_route(), SPEED, Q0)
    np.testing.assert_allclose(ref.sample(-1.0)[0], ref.sample(0.0)[0])
    end = ref.global_total_duration
    np.testing.assert_allclose(ref.sample(end + 10.0)[0], ref.sample(end)[0])


def test_the_attitude_is_the_given_one():
    ref = MinJerkReference(_square_route(), SPEED, Q0)
    np.testing.assert_allclose(ref.sample(1.0)[3], Q0)


def test_repeated_points_do_not_make_the_solve_singular():
    ref = MinJerkReference([np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0])], SPEED, Q0)
    assert ref.num_waypoints == 2


@pytest.mark.parametrize("points", [
    [np.zeros(3)],
    [np.zeros(3), np.zeros(3)],
])
def test_too_few_distinct_points_raises(points):
    with pytest.raises(ValueError):
        MinJerkReference(points, SPEED, Q0)


def test_a_non_positive_speed_raises():
    with pytest.raises(ValueError):
        MinJerkReference(_square_route(), 0.0, Q0)
