"""Unit tests for JaxaTrackingPointTracker (JAXA IAC-22 baseline)."""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
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
