"""The pre-departure reference route step inside GuidanceExecutor
(docs/minco_astar_reference_global.md 3.1)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.executor.guidance_executor import (
    REFERENCE_DEPTH_FRAMES,
    STATUS_ABORTED,
    STATUS_SUCCESS,
    AlignmentAborted,
    ReferenceRouteError,
)

core = pytest.importorskip("sobits_intball2_gnc_cpp")

from test_guidance_executor import (  # noqa: E402
    FakeCheckpointPublisher,
    FakeLogger,
    FakeSetpointPublisher,
    FakeTf,
    _make_clock,
    _make_executor,
)

RESOLUTION = 0.05
INFLATION = 0.2
BOUNDS = (np.array([-2.0, -2.0, -2.0]), np.array([6.0, 6.0, 6.0]))
P0 = np.array([0.0, 0.0, 0.0])
P_TARGET = np.array([3.0, 0.0, 0.0])
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
FORWARD = (1.0, 0.0, 0.0)


def _box(center, half, step=RESOLUTION / 2.0):
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    axes = [np.arange(c - h, c + h + step, step) for c, h in zip(center, half)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class FakeObstacleMap:
    """Enough of ObstacleMap for the route step: a grid, depth bounds, and a
    depth stamp that advances once per spin."""

    def __init__(self, points=(), uses_depth=True, frames_per_spin=1):
        self.grid = core.OccupancyGrid(RESOLUTION, INFLATION)
        self.add(points)
        self.depth_bounds = BOUNDS
        self.uses_depth = uses_depth
        self.last_depth_stamp = None
        self._frames_per_spin = frames_per_spin
        self.spins = 0

    def add_listener(self, _fn):
        pass

    def add(self, points):
        points = np.asarray(points, dtype=float)
        if points.size:
            self.grid.add_points(points.ravel().tolist())

    def on_spin(self):
        self.spins += 1
        if self._frames_per_spin:
            self.last_depth_stamp = (self.last_depth_stamp or 0.0) + 1.0


def _executor(obstacle_map, **kwargs):
    clock_seconds_fn, spin_fn = _make_clock(dt_per_spin=0.05)

    def spin(seconds):
        spin_fn(seconds)
        obstacle_map.on_spin()

    return _make_executor(
        FakeTf(P0, Q0), FakeSetpointPublisher(), FakeCheckpointPublisher(),
        clock_seconds_fn, spin, FakeLogger(), target_speed=1.0,
        obstacle_map=obstacle_map, **kwargs)


def _record_alignments(executor, rotated, status=STATUS_SUCCESS):
    """Replace the real turn with one that reports whether it rotated."""
    calls = []

    def fake(point, p0, q0, _forward_axis, _is_cancel_requested):
        calls.append(np.asarray(point, dtype=float))
        return status, p0, q0, rotated

    executor._pre_align_toward = fake
    return calls


def _plan(executor, can_align=True):
    return executor._plan_reference_route(
        P0, Q0, P_TARGET, FORWARD, can_align=can_align,
        is_cancel_requested=lambda: False)


# --- the 3.1 order ------------------------------------------------------------

def test_depth_is_integrated_before_the_route_is_planned():
    obstacle_map = FakeObstacleMap()
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    route, _p0, _q0 = _plan(executor)
    assert obstacle_map.spins >= REFERENCE_DEPTH_FRAMES
    assert len(route) == 2


def test_a_second_turn_to_the_first_leg_integrates_depth_again():
    obstacle_map = FakeObstacleMap(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    executor = _executor(obstacle_map)
    calls = _record_alignments(executor, rotated=True)
    route, _p0, _q0 = _plan(executor)
    assert len(calls) == 1, "only the first leg; the goal turn is execute()'s own"
    np.testing.assert_allclose(calls[0], route[1])
    assert obstacle_map.spins >= 2 * REFERENCE_DEPTH_FRAMES, "two waits"


def test_no_second_wait_when_the_vehicle_did_not_have_to_turn():
    obstacle_map = FakeObstacleMap(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    _plan(executor)
    assert obstacle_map.spins < 2 * REFERENCE_DEPTH_FRAMES


def test_the_route_turn_is_skipped_without_pre_align():
    obstacle_map = FakeObstacleMap()
    executor = _executor(obstacle_map)
    calls = _record_alignments(executor, rotated=True)
    _plan(executor, can_align=False)
    assert calls == []


def test_an_aborted_turn_propagates_its_status():
    obstacle_map = FakeObstacleMap(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=True, status=STATUS_ABORTED)
    with pytest.raises(AlignmentAborted) as excinfo:
        _plan(executor)
    assert excinfo.value.status == STATUS_ABORTED


# --- the route itself ---------------------------------------------------------

def test_a_box_on_the_chord_is_detoured():
    executor = _executor(FakeObstacleMap(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3])))
    _record_alignments(executor, rotated=False)
    route, _p0, _q0 = _plan(executor)
    assert len(route) > 2
    assert max(abs(float(point[1])) for point in route) > 0.3, "it goes around"


def test_the_route_is_published_through_the_callback():
    executor = _executor(FakeObstacleMap())
    published = []
    executor._reference_route_callback = published.append
    _record_alignments(executor, rotated=False)
    route, _p0, _q0 = _plan(executor)
    assert len(published) == 1 and published[0] is route


def test_a_route_blocked_by_newer_depth_is_replanned_once():
    """A box that only shows up while A* is running must not be driven into."""
    obstacle_map = FakeObstacleMap()
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    real = executor._solve_route
    seen = []

    def solve_then_reveal(p0, p_target):
        route = real(p0, p_target)
        seen.append(route)
        if len(seen) == 1:  # the box appears after the first solve
            obstacle_map.add(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
        return route

    executor._solve_route = solve_then_reveal
    route, _p0, _q0 = _plan(executor)
    assert len(seen) == 2, "replanned exactly once"
    assert len(route) > 2 and route is seen[1]


def test_a_route_still_blocked_after_the_retry_fails():
    obstacle_map = FakeObstacleMap()
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    executor._solve_route = lambda p0, p_target: [np.asarray(p0, dtype=float),
                                                  np.asarray(p_target, dtype=float)]
    obstacle_map.add(_box([1.5, 0.0, 0.0], [0.3, 0.3, 0.3]))
    with pytest.raises(ReferenceRouteError):
        _plan(executor)


def test_an_unreachable_goal_fails():
    """A goal sealed inside a box has no A* route at all."""
    obstacle_map = FakeObstacleMap(_box([3.0, 0.0, 0.0], [0.4, 0.4, 0.4]))
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    with pytest.raises(ReferenceRouteError):
        _plan(executor)


def test_without_an_obstacle_map_the_route_cannot_be_planned():
    clock_seconds_fn, spin_fn = _make_clock()
    executor = _make_executor(
        FakeTf(P0, Q0), FakeSetpointPublisher(), FakeCheckpointPublisher(),
        clock_seconds_fn, spin_fn, FakeLogger(), target_speed=1.0)
    with pytest.raises(ReferenceRouteError):
        _plan(executor)


def test_a_depth_timeout_is_a_warning_not_a_failure():
    """The route is judged on the grid as it stands, so missing frames alone
    must not end the goal."""
    obstacle_map = FakeObstacleMap(frames_per_spin=0)
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    route, _p0, _q0 = _plan(executor)
    assert len(route) == 2
    assert any("depth frames arrived" in w for w in executor._log.warnings)


def test_a_map_without_depth_does_not_wait():
    obstacle_map = FakeObstacleMap(uses_depth=False)
    executor = _executor(obstacle_map)
    _record_alignments(executor, rotated=False)
    _plan(executor)
    assert obstacle_map.spins == 0
