"""Unit tests for JaxaTrackingPointTracker (JAXA IAC-22 baseline)."""
import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_rotate
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import JaxaPlannerConfig
from sobits_intball2_gnc.guidance.trajectory_tracking import jaxa_tracking_point_tracker as jt
from test_jaxa_rrt_local_planner import BOUNDS, make_grid

GOAL = np.array([3.5, 0.0, 0.0])
Q_LEVEL_UPSIDE_DOWN = np.array([1.0, 0.0, 0.0, 0.0])  # 180 deg roll: body z = -world z


class Pose:
    def __init__(self, p, q=(0., 0., 0., 1.)):
        self.p, self.q, self.stamp = np.asarray(p, float), np.asarray(q, float), 0.0

    def __call__(self):
        return self.p, self.q, self.stamp


class ManualThread:
    last = None

    """Background solve that finishes only when the test says so."""

    def __init__(self, target, args=(), daemon=None):
        self.target, self.args, self.done = target, args, False
        ManualThread.last = self

    def start(self):
        self.target(*self.args)

    def is_alive(self):
        return not self.done


def make(grid, pose, q0=(0., 0., 0., 1.), async_replan=True):
    return jt.JaxaTrackingPointTracker(
        pose.p, GOAL, pose, lambda _s: True, q0, grid, BOUNDS, lookahead_m=0.3,
        config=JaxaPlannerConfig(planner="rrt", rrt_iterations=300, max_attempts=5), collision_check_period=0.05,
        goal_facing_hold_m=0.3, async_replan=async_replan)


def test_initial_straight_path_tracks_lookahead_and_faces_goal():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make(make_grid(), pose)
    p, v, a, q = tracker.sample(0.0)
    assert np.isinf(tracker.total_duration)
    assert np.linalg.norm(p - pose.p) > 0.25 and np.all(v == 0) and np.all(a == 0)
    np.testing.assert_allclose(quat_rotate(q, [1., 0., 0.]), [1., 0., 0.], atol=1e-9)


def test_free_global_path_is_followed_without_planning():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make(make_grid(), pose)
    np.testing.assert_allclose(tracker.path, [pose.p, GOAL])
    assert tracker.plans[0]["attempts"] == 0


def test_blocked_global_path_runs_the_local_planner():
    pose = Pose([0.5, 0.0, 0.0])
    grid = make_grid([([2.0, 0.0, 0.0], [0.2, 0.4, 1.0])])
    tracker = make(grid, pose)
    assert tracker.plans[0]["attempts"] >= 1 and len(tracker.path) > 2
    np.testing.assert_allclose(tracker.path[0], pose.p)
    np.testing.assert_allclose(tracker.path[-1], GOAL)
    rejoin = tracker.path[-2]  # local path rejoins the global line past the box
    np.testing.assert_allclose(rejoin[1:], [0.0, 0.0], atol=1e-9)
    assert 3.15 <= rejoin[0] <= 3.35  # 1 m past the last blocked sample


def test_goal_facing_keeps_an_upside_down_level_roll():
    pose = Pose([0.5, 0.0, 0.0], Q_LEVEL_UPSIDE_DOWN)
    tracker = make(make_grid(), pose, q0=Q_LEVEL_UPSIDE_DOWN)
    pose.p = np.array([0.5, 0.3, 0.0])
    _p, _v, _a, q = tracker.sample(0.0)
    np.testing.assert_allclose(quat_rotate(q, [0., 0., 1.])[2], -1.0, atol=1e-6)


def test_attitude_is_held_near_the_goal_and_total_duration_set_on_reaching_it():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make(make_grid(), pose)
    pose.p = np.array([3.3, 0.05, 0.0])
    _p, _v, _a, q_before = tracker.sample(1.0)
    pose.p = np.array([3.6, -0.05, 0.0])  # passed the goal: direction to it flips
    p, _v, _a, q = tracker.sample(2.0)
    np.testing.assert_allclose(q, q_before)
    np.testing.assert_allclose(p, GOAL)
    assert tracker.total_duration == 1.0  # tracking point 3.6 clamps to the goal at t=1


def test_async_replan_follows_old_path_until_the_solve_finishes(monkeypatch):
    monkeypatch.setattr(jt.threading, "Thread", ManualThread)
    pose = Pose([0.5, 0.0, 0.0])
    grid = make_grid()
    tracker = make(grid, pose)
    old_path = tracker.path.copy()
    grid.add_box([2.0, 0.0, 0.0], [0.2, 0.4, 1.0])
    tracker.sample(0.0)
    assert ManualThread.last is not None
    np.testing.assert_allclose(tracker.path, old_path)
    tracker.sample(0.1)
    np.testing.assert_allclose(tracker.path, old_path)
    ManualThread.last.done = True
    tracker.sample(0.2)
    assert tracker.last_replan_occurred and tracker.last_replan_source == "async"
    assert not np.array_equal(tracker.path, old_path)


def test_failed_replan_sets_failed_and_holds():
    pose = Pose([0.5, 0.0, 0.0])
    grid = make_grid()
    tracker = make(grid, pose, async_replan=False)
    grid.add_box([2.0, 0.0, 0.0], [0.2, 2.0, 2.0])  # wall across bounds
    p, _v, _a, _q = tracker.sample(0.0)
    assert tracker.failed and tracker.replanning_stopped
    np.testing.assert_allclose(p, pose.p)


class Log:
    def info(self, *_a):
        pass

    warn = error = info


class ObstacleMap:
    def __init__(self, grid):
        self.grid, self.listeners = grid, []

    def add_listener(self, fn):
        self.listeners.append(fn)


def builder_options():
    # Same keys as GuidanceExecutor passes (production OMPL planner).
    return dict(lookahead_m=0.3, ompl_solve_time_s=0.1, max_attempts=50,
                collision_check_period=0.05, goal_facing_hold_m=0.3,
                bounds=BOUNDS[0] + BOUNDS[1])


def test_tracker_builder_builds_jaxa_rrt_and_forwards_grid_updates():
    from sobits_intball2_gnc.guidance.executor.tracker_builder import TrackerBuilder

    pose = Pose([0.5, 0.0, 0.0])
    tf = type("TF", (), {"get_pose": staticmethod(pose)})()
    obstacle_map = ObstacleMap(make_grid())
    builder = TrackerBuilder(tf, lambda _s: True, Log(), 0.5, 0.05, 1.0, None, 3.2, None,
                             obstacle_map=obstacle_map)
    tracker, traj = builder.build(pose.p, pose.q, GOAL, [], "jaxa_rrt", np.array([1., 0., 0.]), True,
                                  jaxa_options=builder_options())
    assert isinstance(tracker, jt.JaxaTrackingPointTracker) and traj is None
    new_grid = make_grid()
    obstacle_map.listeners[0](new_grid)
    assert tracker._grid is new_grid


# --- shared reference route as the global path ---------------------------------

L_ROUTE = [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.8, 0.0], [3.5, 0.8, 0.0]]


def make_route(grid, pose, route=L_ROUTE, goal=(3.5, 0.8, 0.0)):
    return jt.JaxaTrackingPointTracker(
        pose.p, goal, pose, lambda _s: True, (0., 0., 0., 1.), grid, BOUNDS, lookahead_m=0.3,
        config=JaxaPlannerConfig(planner="rrt", rrt_iterations=300, max_attempts=5),
        collision_check_period=0.05, goal_facing_hold_m=0.3, async_replan=False,
        reference_route=route)


def test_reference_route_is_densified_with_its_shape_and_ends_unchanged():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make_route(make_grid(), pose)
    path = tracker.path
    legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert legs.max() <= jt.REFERENCE_SPACING_M + 1e-9
    for vertex in L_ROUTE:
        assert np.min(np.linalg.norm(path - vertex, axis=1)) < 1e-9
    assert tracker.plans[0]["attempts"] == 0


def test_reference_route_ends_are_set_to_the_start_and_the_goal():
    pose = Pose([0.53, 0.02, 0.0])
    tracker = make_route(make_grid(), pose, goal=(3.52, 0.8, 0.0))
    np.testing.assert_allclose(tracker.path[0], pose.p)
    np.testing.assert_allclose(tracker.path[-1], [3.52, 0.8, 0.0])


def test_reference_route_needs_two_points():
    import pytest
    with pytest.raises(ValueError):
        make_route(make_grid(), Pose([0.5, 0.0, 0.0]), route=[[0.5, 0.0, 0.0]])


def test_tracking_point_stays_on_the_densified_corner():
    pose = Pose([1.9, 0.0, 0.0])
    tracker = make_route(make_grid(), pose)
    target, _v, _a, _q = tracker.sample(0.0)
    # Straight past the corner at x = 2.0 would leave y = 0 only after it; a sparse
    # polyline would already aim at (2.0, 0.8) from here.
    assert target[1] < 0.35


def test_blocked_route_rejoins_it_and_keeps_the_rest_of_the_route():
    pose = Pose([0.5, 0.0, 0.0])
    grid = make_grid([((1.2, 0.0, 0.0), (0.15, 0.3, 0.5))])
    tracker = make_route(grid, pose)
    assert tracker.plans[0]["attempts"] >= 1
    path = tracker.path
    np.testing.assert_allclose(path[-1], [3.5, 0.8, 0.0])
    # the route's corner survives after the rejoin: the path does not jump to the goal
    assert np.min(np.linalg.norm(path - np.array([2.0, 0.8, 0.0]), axis=1)) < 1e-9
    assert np.min(np.linalg.norm(path - np.array([2.0, 0.0, 0.0]), axis=1)) < 0.6


# --- attitude: face the goal (default) or the path ahead ------------------------

def make_attitude(pose, mode, route=L_ROUTE, goal=(3.5, 0.8, 0.0), **kwargs):
    return jt.JaxaTrackingPointTracker(
        pose.p, goal, pose, lambda _s: True, (0., 0., 0., 1.), make_grid(), BOUNDS, lookahead_m=0.3,
        config=JaxaPlannerConfig(planner="rrt", rrt_iterations=300, max_attempts=5),
        collision_check_period=0.05, goal_facing_hold_m=0.3, async_replan=False,
        reference_route=route, attitude_mode=mode, **kwargs)


def camera_axis(q):
    return quat_rotate(q, [1., 0., 0.])


def test_goal_mode_faces_the_goal_and_path_mode_faces_the_path_ahead():
    goal_dir = np.array([3.0, 0.8, 0.0]) / np.linalg.norm([3.0, 0.8, 0.0])
    pose = Pose([0.5, 0.0, 0.0])
    q_goal = make_attitude(pose, "goal").sample(0.0)[3]
    q_path = make_attitude(pose, "path").sample(0.0)[3]
    np.testing.assert_allclose(camera_axis(q_goal), goal_dir, atol=1e-9)
    np.testing.assert_allclose(camera_axis(q_path), [1., 0., 0.], atol=1e-9)  # first leg


def test_path_mode_turn_rate_is_limited():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make_attitude(pose, "path", path_facing_max_rate_deg=20.0)
    q_prev = tracker.sample(0.0)[3]
    pose.p = np.array([2.0, 0.1, 0.0])  # the path ahead now runs along +y
    t, turned = 0.0, 0.0
    for _ in range(20):
        t += 0.05
        pose.stamp = t
        q = tracker.sample(t)[3]
        step = geodesic_angle(q_prev, q)
        assert step <= np.radians(20.0) * 0.05 + 1e-9
        turned += step
        q_prev = q
    assert turned > np.radians(20.0)  # it does turn, just not instantly


def test_path_mode_holds_the_attitude_near_the_goal():
    pose = Pose([0.5, 0.0, 0.0])
    tracker = make_attitude(pose, "path")
    q_start = tracker.sample(0.0)[3]
    pose.p = np.array([3.4, 0.8, 0.0])  # inside goal_facing_hold_m
    q_end = tracker.sample(0.1)[3]
    np.testing.assert_allclose(q_end, q_start)


def test_unknown_attitude_mode_is_rejected():
    with pytest.raises(ValueError):
        make_attitude(Pose([0.5, 0.0, 0.0]), "sideways")
    with pytest.raises(ValueError):
        make_attitude(Pose([0.5, 0.0, 0.0]), "path", path_facing_max_rate_deg=0.0)


def test_tracker_builder_forwards_the_attitude_options():
    from sobits_intball2_gnc.guidance.executor.tracker_builder import TrackerBuilder, TrajectoryBuildError

    pose = Pose([0.5, 0.0, 0.0])
    tf = type("TF", (), {"get_pose": staticmethod(pose)})()
    builder = TrackerBuilder(tf, lambda _s: True, Log(), 0.5, 0.05, 1.0, None, 3.2, None,
                             obstacle_map=ObstacleMap(make_grid()))
    options = dict(builder_options(), attitude_mode="path", path_facing_ahead_m=0.7,
                   path_facing_max_rate_deg=30.0)
    tracker, _ = builder.build(pose.p, pose.q, GOAL, [], "jaxa_rrt", np.array([1., 0., 0.]), True,
                               jaxa_options=options)
    assert tracker._attitude_mode == "path"
    assert tracker._facing_ahead == 0.7
    np.testing.assert_allclose(tracker._facing_max_rate, np.radians(30.0))
    with pytest.raises(TrajectoryBuildError):
        builder.build(pose.p, pose.q, GOAL, [], "jaxa_rrt", np.array([1., 0., 0.]), True,
                      jaxa_options=dict(builder_options(), attitude_mode="sideways"))
