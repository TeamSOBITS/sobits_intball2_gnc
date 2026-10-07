"""Unit tests for ReplanMincoTracker (docs/
2026-09-20_ego_v2_style_replan_migration_plan.md).
"""
import threading

import numpy as np
import pytest

pytest.importorskip("sobits_intball2_gnc_cpp")

from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

P0 = [0.0, 0.0, 0.0]
P_TARGET = [3.0, 0.0, 0.0]
Q0 = [0.0, 0.0, 0.0, 1.0]
TARGET_SPEED = 0.5
MAX_ACCEL = 0.1 / 4.5
DT = 0.05
T_CAP = 300.0


class _IdealTrackingTf:
    """Reports the tracker's own last position setpoint as the measured pose:
    replan_minco replans from its reference state, so a TF advancing independently of
    the setpoint would not exercise the real closed loop."""

    def __init__(self, p0=P0, offset=(0.0, 0.0, 0.0)):
        self.pos = np.asarray(p0, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        self.stamp = 0.0
        self.available = True

    def pose_fn(self):
        if not self.available:
            return None
        return self.pos + self.offset, list(Q0), self.stamp

    def advance(self, t, p_out):
        self.stamp = t
        self.pos = np.asarray(p_out, dtype=float)


def _make_tracker(tf, tf_fresh_fn=lambda stamp: True, p_target=P_TARGET, **kwargs):
    kwargs.setdefault("target_speed", TARGET_SPEED)
    kwargs.setdefault("max_accel", MAX_ACCEL)
    return ReplanMincoTracker(
        P0, p_target, tf.pose_fn, tf_fresh_fn, Q0, **kwargs
    )


def _run_to_end(tracker, tf):
    """Steps until ``total_duration`` is passed; returns per-tick records
    ``(t, p, v, a, q, replanned)``."""
    records = []
    t = 0.0
    while t <= tracker.total_duration and t < T_CAP:
        t += DT
        tf.stamp = t
        p, v, a, q = tracker.sample(t)
        tf.advance(t, p)
        records.append((t, p, v, a, q, tracker.last_replan_occurred))
    return records


def test_sample_at_t_zero_starts_at_rest_at_p0():
    tracker = _make_tracker(_IdealTrackingTf())
    p, v, a, q = tracker.sample(0.0)
    assert np.allclose(p, P0, atol=1e-6)
    assert np.allclose(v, 0.0, atol=1e-6)
    assert np.allclose(q, Q0)


def test_ideal_tracking_reaches_target_and_total_duration_freezes():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf)
    records = _run_to_end(tracker, tf)

    t_end, p, v, _a, _q, _r = records[-1]
    assert t_end < T_CAP
    assert np.allclose(p, P_TARGET, atol=1e-3)
    assert np.allclose(v, 0.0, atol=1e-3)
    assert tracker.last_fallback_reason is None

    frozen = tracker.total_duration
    tracker.sample(t_end + DT)
    tracker.sample(t_end + 2 * DT)
    assert tracker.total_duration == pytest.approx(frozen)


def test_replans_every_period_with_continuous_reference():
    period = 1.0
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=period)

    local, local_start_t = tracker.local_trajectory, 0.0
    replan_times = []
    t = 0.0
    while t <= tracker.total_duration and t < T_CAP:
        t += DT
        tf.stamp = t
        p, v, a, _q = tracker.sample(t)
        tf.advance(t, p)
        if tracker.last_replan_occurred:
            p_old, v_old, a_old, _ = local.sample(t - local_start_t)
            assert np.allclose(p, p_old, atol=1e-4)
            assert np.allclose(v, v_old, atol=1e-4)
            assert np.allclose(a, a_old, atol=1e-4)
            replan_times.append(t)
            local, local_start_t = tracker.local_trajectory, t

    assert len(replan_times) >= 2
    assert np.allclose(np.diff(replan_times), period, atol=DT + 1e-9)


def test_via_waypoint_is_passed():
    via = np.array([1.5, 1.0, 0.0])
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, route_waypoints=[via], via_half_width=0.0)

    global_traj = tracker.trajectory
    global_ts = np.arange(0.0, global_traj.global_total_duration, 0.01)
    global_min = min(np.linalg.norm(global_traj.sample(t)[0] - via) for t in global_ts)
    assert global_min < 1e-2

    records = _run_to_end(tracker, tf)
    tracked_min = min(np.linalg.norm(r[1] - via) for r in records)
    assert tracked_min < 0.1
    assert np.allclose(records[-1][1], P_TARGET, atol=1e-3)


def test_tf_stale_latches_permanently():
    stale_at = 1.0
    tf = _IdealTrackingTf()
    tracker = _make_tracker(
        tf, tf_fresh_fn=lambda stamp: not np.isclose(stamp, stale_at))

    t = 0.0
    frozen = None
    for _ in range(100):
        t += DT
        tf.stamp = t
        out = tracker.sample(t)
        tf.advance(t, out[0])
        if t >= stale_at - 1e-9:
            if frozen is None:
                frozen = out
            for got, want in zip(out, frozen):
                assert np.allclose(got, want)
            assert tracker.last_fallback_reason == "tf_stale"


def test_pose_none_latches_tf_stale():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf)
    before = tracker.sample(0.0)
    tf.available = False
    out = tracker.sample(0.5)
    assert tracker.last_fallback_reason == "tf_stale"
    assert tracker.replanning_stopped
    for got, want in zip(out, before):
        assert np.allclose(got, want)


def test_local_replan_failure_keeps_previous_local_and_retries():
    period = 1.0
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=period)

    real_build_local = tracker._build_local
    calls = {"n": 0}

    def fail_first_build_local(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MincoInfeasibleError("injected")
        return real_build_local(*args)

    tracker._build_local = fail_first_build_local

    initial_local = tracker.local_trajectory
    failed_at = None
    retried_at = None
    t = 0.0
    while t <= tracker.total_duration and t < T_CAP:
        t += DT
        tf.stamp = t
        p, v, _a, _q = tracker.sample(t)
        tf.advance(t, p)
        if tracker.last_local_fallback:
            failed_at = t
            assert tracker.last_fallback_reason == "minco_infeasible_local"
            assert not tracker.replanning_stopped
            assert tracker.local_trajectory is initial_local
            p_old, v_old, _, _ = initial_local.sample(t)
            assert np.allclose(p, p_old)
            assert np.allclose(v, v_old)
        elif failed_at is not None and retried_at is None and tracker.last_replan_occurred:
            retried_at = t

    assert failed_at is not None
    assert retried_at == pytest.approx(failed_at + period, abs=DT + 1e-9)
    assert np.allclose(p, P_TARGET, atol=1e-3)


def test_steady_state_offset_still_terminates():
    tf = _IdealTrackingTf(offset=(0.0, 0.02, 0.0))
    tracker = _make_tracker(tf)
    records = _run_to_end(tracker, tf)
    assert records[-1][0] < T_CAP
    assert np.allclose(records[-1][1], P_TARGET, atol=1e-3)


def test_non_increasing_t_returns_last_output():
    tracker = _make_tracker(_IdealTrackingTf())
    first = tracker.sample(1.0)
    for t in (1.0, 0.5):
        out = tracker.sample(t)
        for got, want in zip(out, first):
            assert np.allclose(got, want)


def test_attitude_stays_q0_throughout():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf)
    records = _run_to_end(tracker, tf)
    assert all(np.allclose(r[4], Q0) for r in records)


def test_target_speed_and_max_accel_must_be_given_together():
    with pytest.raises(ValueError):
        _make_tracker(_IdealTrackingTf(), target_speed=TARGET_SPEED, max_accel=None)
    with pytest.raises(ValueError):
        _make_tracker(_IdealTrackingTf(), target_speed=None, max_accel=MAX_ACCEL)


@pytest.mark.parametrize("kwargs", [
    {"local_replan_period": 0.0},
    {"planning_horizon_m": -1.0},
])
def test_non_positive_local_replan_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _make_tracker(_IdealTrackingTf(), **kwargs)


def test_freetime_global_reaches_target():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, target_speed=None, max_accel=None)
    records = _run_to_end(tracker, tf)
    assert records[-1][0] < T_CAP
    assert np.allclose(records[-1][1], P_TARGET, atol=1e-3)
    assert np.allclose(records[-1][2], 0.0, atol=1e-3)


FACE_TRAVEL_ROUTE = [[3.0, 0.0, 0.0]]
FACE_TRAVEL_TARGET = [3.0, 3.0, 0.0]
FACE_TRAVEL_MAX_VEL = 0.2


@pytest.fixture(scope="module")
def face_travel_run():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(
        tf, p_target=FACE_TRAVEL_TARGET, route_waypoints=FACE_TRAVEL_ROUTE,
        planning_horizon_m=4.0, face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL,
        attitude_resample_spacing_m=0.3,
    )
    return _run_to_end(tracker, tf)


def _angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


def test_face_travel_faces_own_velocity_through_turn(face_travel_run):
    errors = [_angle_deg(quat_rotate(q, np.array([1.0, 0.0, 0.0])), v)
              for _t, _p, v, _a, q, _r in face_travel_run if np.linalg.norm(v) > 0.05]
    # The first local already aims past the corner (leg < look-ahead) while the
    # head still faces the first leg, so the start is off by ~25 deg for a while.
    assert np.percentile(errors, 95) < 20.0
    assert np.allclose(face_travel_run[-1][1], FACE_TRAVEL_TARGET, atol=1e-2)


def test_face_travel_attitude_continuous_across_replans(face_travel_run):
    assert any(r[5] for r in face_travel_run)
    steps = [_angle_deg(quat_rotate(prev[4], np.array([1.0, 0.0, 0.0])),
                        quat_rotate(cur[4], np.array([1.0, 0.0, 0.0])))
             for prev, cur in zip(face_travel_run, face_travel_run[1:])]
    assert max(steps) < 2.0


def test_face_travel_respects_local_speed_cap(face_travel_run):
    assert max(np.linalg.norm(r[2]) for r in face_travel_run) < FACE_TRAVEL_MAX_VEL * 1.03


def _gate_local_builds(tracker):
    gate = threading.Event()
    build = tracker._build_local

    def gated_build(*args):
        gate.wait()
        return build(*args)

    tracker._build_local = gated_build
    return gate


def test_async_replan_keeps_old_local_while_solving_then_swaps_continuously():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=1.0, async_replan=True)
    gate = _gate_local_builds(tracker)
    old_local = tracker.local_trajectory

    t = 0.0
    while tracker._pending_thread is None:
        t += DT
        p, _v, _a, _q = tracker.sample(t)
    blocked_ticks = 6
    for _ in range(blocked_ticks):
        t += DT
        p, _v, _a, _q = tracker.sample(t)
        assert not tracker.last_replan_occurred
        assert np.allclose(p, old_local.sample(t)[0], atol=1e-9)

    gate.set()
    tracker._pending_thread.join()
    t += DT
    p, v, _a, _q = tracker.sample(t)
    assert tracker.last_replan_occurred
    assert tracker.last_replan_lag_seconds == pytest.approx((blocked_ticks + 1) * DT)
    p_old, v_old, _a_old, _q_old = old_local.sample(t)
    assert np.allclose(p, p_old, atol=1e-3)
    assert np.allclose(v, v_old, atol=1e-3)


def test_adopting_a_local_reports_the_share_of_the_wrench_envelope_it_uses():
    from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
    from sobits_intball2_gnc.guidance.constraints.actuation_envelope import wrench_envelope_halfspaces
    allocator = ThrustAllocator()
    envelope = wrench_envelope_halfspaces(allocator.A, allocator.fj_max)
    tracker = _make_tracker(_IdealTrackingTf(), wrench_envelope=envelope, mass=3.216, inertia=0.0136)
    tracker._solve_local_in_background(tracker._reference_start_state())
    tracker._adopt_local(tracker._pending_result, 0.0, "async")
    use = tracker.last_replan_wrench_use
    assert use is not None and 0.0 < use < 1.05   # the planner (margin 1.0) works up to the boundary
    # Half the envelope scale means twice the share.
    half = (envelope[0], envelope[1] * 0.5)
    tracker._wrench_envelope = half
    assert tracker._planned_wrench_use(tracker.local_trajectory, 0.0) == pytest.approx(2.0 * use)
    # Synchronous adoptions do not compute it (it would hold the setpoint loop).
    tracker._adopt_local(tracker._try_build_local(tracker._reference_start_state()), 0.0, "periodic")
    assert tracker.last_replan_wrench_use is None
    unconfigured = _make_tracker(_IdealTrackingTf())
    unconfigured._solve_local_in_background(unconfigured._reference_start_state())
    unconfigured._adopt_local(unconfigured._pending_result, 0.0, "async")
    assert unconfigured.last_replan_wrench_use is None


def test_scalar_limits_replace_the_wrench_envelope_in_the_local_solve():
    limit_accel, limit_ang = 0.04, 0.4
    tracker = _make_tracker(_IdealTrackingTf(), scalar_limits=(limit_accel, limit_ang))
    local = tracker.local_trajectory
    times = np.arange(0.0, local.global_total_duration, 0.1)
    accel = max(np.linalg.norm(local.sample(t)[2]) for t in times)
    ang_accel = max(np.linalg.norm(local.sample_body_angular(t)[1]) for t in times)
    assert 0.0 < accel <= limit_accel * 1.05
    assert ang_accel <= limit_ang * 1.05


def test_async_replan_face_travel_reaches_target():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(
        tf, p_target=FACE_TRAVEL_TARGET, route_waypoints=FACE_TRAVEL_ROUTE,
        planning_horizon_m=4.0, face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL,
        attitude_resample_spacing_m=0.3, async_replan=True,
    )
    replans = 0
    t = 0.0
    while t <= tracker.total_duration and t < T_CAP:
        t += DT
        tf.stamp = t
        p, _v, _a, _q = tracker.sample(t)
        tf.advance(t, p)
        replans += tracker.last_replan_occurred
        if tracker._pending_thread is not None:
            tracker._pending_thread.join()
    assert replans >= 2
    assert tracker.last_fallback_reason is None
    assert np.allclose(p, FACE_TRAVEL_TARGET, atol=1e-2)


def test_keeps_replanning_after_touch_goal_until_goal_local_plays_out():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=1.0)
    replans_after_touch = 0
    t = 0.0
    while t <= tracker.total_duration and t < T_CAP:
        touched_before_tick = tracker._local_touches_goal
        t += DT
        tf.stamp = t
        p, v, _a, _q = tracker.sample(t)
        tf.advance(t, p)
        replans_after_touch += touched_before_tick and tracker.last_replan_occurred
    assert replans_after_touch >= 2
    assert t < T_CAP
    assert np.allclose(p, P_TARGET, atol=1e-3)
    assert np.allclose(v, 0.0, atol=1e-3)


class _FreeGrid:
    resolution = 0.1

    def inflated_occupied(self, _p):
        return False


class _HoldStop:
    duration = 3.0

    def __init__(self, p):
        self._p = np.asarray(p, dtype=float)

    def sample(self, _t):
        return self._p, np.zeros(3), np.zeros(3), list(Q0), np.zeros(3), np.zeros(3)


def _async_tracker_with_scripted_collision(tf):
    tracker = _make_tracker(
        tf, local_replan_period=100.0, async_replan=True, collision_check_period=DT,
        stop_profile_fn=lambda p, v, q, omega: _HoldStop(p))
    tracker._planner.obstacle_grid = _FreeGrid()
    collision = {"ahead_s": None}
    tracker._collision_ahead_s = lambda: collision["ahead_s"]
    return tracker, collision


def _tick(tracker, tf, t):
    t += DT
    tf.stamp = t
    tf.advance(t, tracker.sample(t)[0])
    return t


def test_async_collision_keeps_flying_while_replanning_and_adopts_success():
    tf = _IdealTrackingTf()
    tracker, collision = _async_tracker_with_scripted_collision(tf)
    gate = _gate_local_builds(tracker)
    collision["ahead_s"] = 1.0
    t = 0.0
    for _ in range(6):
        t = _tick(tracker, tf, t)
        assert tracker.emergency_stops == 0
    assert tracker._pending_thread is not None

    collision["ahead_s"] = None
    gate.set()
    tracker._pending_thread.join()
    t = _tick(tracker, tf, t)
    assert tracker.last_replan_occurred
    assert tracker.emergency_stops == 0


def test_async_collision_stops_once_the_replan_fails():
    tf = _IdealTrackingTf()
    tracker, collision = _async_tracker_with_scripted_collision(tf)
    gate = _gate_local_builds(tracker)
    gated_build = tracker._build_local

    def failing_build(*args):
        gated_build(*args)
        raise MincoInfeasibleError("scripted")

    tracker._build_local = failing_build
    collision["ahead_s"] = 1.0
    t = _tick(tracker, tf, 0.0)
    assert tracker.emergency_stops == 0

    gate.set()
    tracker._pending_thread.join()
    t = _tick(tracker, tf, t)
    assert tracker.emergency_stops == 1
    assert tracker.last_fallback_reason == "emergency_stop"
    assert not tracker.replanning_stopped


def _join_pending(tracker):
    """A daemon solve still inside the native solver at interpreter exit aborts the process."""
    if tracker._pending_thread is not None:
        tracker._pending_thread.join()


def _older_solve_landing_with_collision(ahead_s_after_landing):
    tf = _IdealTrackingTf()
    tracker, collision = _async_tracker_with_scripted_collision(tf)
    gate = _gate_local_builds(tracker)
    tracker._start_background_replan()
    older = tracker._pending_thread

    collision["ahead_s"] = 1.0
    t = _tick(tracker, tf, 0.0)
    assert tracker._pending_thread is older

    collision["ahead_s"] = ahead_s_after_landing
    gate.set()
    older.join()
    _tick(tracker, tf, t)
    return tracker, older


def test_async_older_solve_landing_collision_free_on_the_current_grid_is_kept():
    tracker, _older = _older_solve_landing_with_collision(None)
    assert tracker.last_replan_occurred
    assert tracker.emergency_stops == 0
    assert tracker._pending_thread is None
    _join_pending(tracker)


def test_async_older_solve_landing_still_colliding_far_away_replans_without_stopping():
    tracker, older = _older_solve_landing_with_collision(_HoldStop.duration + 1.0)
    assert tracker.emergency_stops == 0
    assert tracker._pending_thread is not None and tracker._pending_thread is not older
    _join_pending(tracker)


def test_async_older_solve_landing_still_colliding_close_stops_without_solving_again():
    tracker, _older = _older_solve_landing_with_collision(1.0)
    assert tracker.emergency_stops == 1
    assert tracker._stop_profile is not None
    _join_pending(tracker)


class _OccupiedBeyondX:
    resolution = 0.1

    def __init__(self, x):
        self.x = x

    def inflated_occupied(self, p):
        return p[0] >= self.x


def _moving_tracker_with_collision_ahead(tf, grid_x_ahead):
    tracker, collision = _async_tracker_with_scripted_collision(tf)
    t = 0.0
    for _ in range(40):
        t = _tick(tracker, tf, t)
    tracker._planner.obstacle_grid = _OccupiedBeyondX(tf.pos[0] + grid_x_ahead)
    gate = _gate_local_builds(tracker)
    collision["ahead_s"] = 1.0
    t = _tick(tracker, tf, t)
    return tracker, gate


def test_async_collision_stops_at_once_when_braking_after_the_expected_wait_would_collide():
    tf = _IdealTrackingTf()
    tracker, gate = _moving_tracker_with_collision_ahead(tf, 0.01)
    assert tracker.emergency_stops == 1
    assert tracker._pending_thread is not None
    gate.set()
    _join_pending(tracker)


def test_async_collision_keeps_flying_when_braking_after_the_expected_wait_is_clear():
    tf = _IdealTrackingTf()
    tracker, gate = _moving_tracker_with_collision_ahead(tf, 100.0)
    assert tracker.emergency_stops == 0
    assert tracker._pending_thread is not None
    gate.set()
    _join_pending(tracker)


def test_expected_replan_wait_is_a_margin_over_recent_async_waits():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=1.0, async_replan=True)
    assert tracker._expected_replan_wait_s() == pytest.approx(1.0)
    gate = _gate_local_builds(tracker)
    t = 0.0
    while tracker._pending_thread is None:
        t = _tick(tracker, tf, t)
    for _ in range(5):
        t = _tick(tracker, tf, t)
    gate.set()
    tracker._pending_thread.join()
    _tick(tracker, tf, t)
    assert tracker._expected_replan_wait_s() == pytest.approx(1.2 * 6 * DT)


def _braking_tracker(tf):
    tracker, _collision = _async_tracker_with_scripted_collision(tf)
    t = 0.0
    for _ in range(10):
        t = _tick(tracker, tf, t)
    gate = _gate_local_builds(tracker)
    tracker._start_emergency_stop(_HoldStop(tf.pos))
    t = _tick(tracker, tf, t)
    assert tracker._pending_is_brake_replan
    return tracker, gate, t


def test_braking_switches_to_a_collision_free_replan_once_the_brake_reaches_its_start():
    tf = _IdealTrackingTf()
    tracker, gate, t = _braking_tracker(tf)
    t_switch = tracker._brake_replan_t
    gate.set()
    tracker._pending_thread.join()
    ticks = 0
    while tracker._stop_profile is not None and ticks < 100:
        t = _tick(tracker, tf, t)
        ticks += 1
    assert tracker._stop_profile is None
    assert tracker.last_replan_occurred
    assert tracker.emergency_stops == 1
    assert ticks * DT >= t_switch - DT - 1e-9
    _join_pending(tracker)


def test_braking_drops_a_replan_that_lands_after_its_start_time():
    tf = _IdealTrackingTf()
    tracker, gate, t = _braking_tracker(tf)
    late = tracker._pending_thread
    for _ in range(int(tracker._brake_replan_t / DT) + 5):
        t = _tick(tracker, tf, t)
    gate.set()
    late.join()
    t = _tick(tracker, tf, t)
    assert tracker._stop_profile is not None
    assert tracker._brake_switch is None
    assert tracker._pending_thread is not None and tracker._pending_thread is not late
    _join_pending(tracker)


def test_braking_drops_a_replan_whose_local_collides():
    tf = _IdealTrackingTf()
    tracker, gate, t = _braking_tracker(tf)
    tracker._local_collides = lambda local, touches_goal: True
    gate.set()
    tracker._pending_thread.join()
    for _ in range(40):
        t = _tick(tracker, tf, t)
    assert tracker._stop_profile is not None
    assert tracker._brake_switch is None
    assert not tracker.last_replan_occurred
    _join_pending(tracker)


def test_stop_drops_a_solve_started_before_it():
    tf = _IdealTrackingTf()
    tracker, _collision = _async_tracker_with_scripted_collision(tf)
    gate = _gate_local_builds(tracker)
    tracker._start_background_replan()
    before_stop = tracker._pending_thread
    local_before = tracker.local_trajectory
    tracker._start_emergency_stop(_HoldStop(tf.pos))
    gate.set()
    before_stop.join()
    t = _tick(tracker, tf, 0.0)
    assert tracker.local_trajectory is local_before
    assert not tracker.last_replan_occurred
    assert tracker._pending_is_brake_replan
    _join_pending(tracker)


def test_sensor_stale_stops_and_holds_without_replanning_until_fresh_again():
    tf = _IdealTrackingTf()
    sensor = {"fresh": True}
    tracker = _make_tracker(tf, local_replan_period=0.5,
                            stop_profile_fn=lambda p, v, q, omega: _HoldStop(p),
                            sensor_fresh_fn=lambda: sensor["fresh"])
    t = 0.0
    for _ in range(10):
        t = _tick(tracker, tf, t)
    assert tracker.emergency_stops == 0

    sensor["fresh"] = False
    t = _tick(tracker, tf, t)
    assert tracker.emergency_stops == 1
    assert tracker.last_fallback_reason == "sensor_stale"
    for _ in range(int((_HoldStop.duration + 3.0) / DT)):
        t = _tick(tracker, tf, t)
    assert tracker._stop_profile is not None
    assert tracker.emergency_stops == 1

    sensor["fresh"] = True
    for _ in range(int(1.0 / DT)):
        t = _tick(tracker, tf, t)
    assert tracker._stop_profile is None
    assert not tracker.replanning_stopped


def test_rest_replan_into_an_obstacle_is_not_adopted():
    tf = _IdealTrackingTf()
    tracker = _make_tracker(tf, local_replan_period=0.5,
                            stop_profile_fn=lambda p, v, q, omega: _HoldStop(p))
    tracker._planner.obstacle_grid = _FreeGrid()
    tracker._start_emergency_stop(_HoldStop(tf.pos))
    tracker._local_collides = lambda local, touches_goal: True
    t = 0.0
    for _ in range(int((_HoldStop.duration + 2.0) / DT)):
        t = _tick(tracker, tf, t)
    assert tracker._stop_profile is not None
    assert tracker._rest_replan_failures > 0

    tracker._local_collides = lambda local, touches_goal: False
    for _ in range(int(1.0 / DT)):
        t = _tick(tracker, tf, t)
        if tracker._stop_profile is None:
            break
    assert tracker._stop_profile is None
    assert tracker.last_replan_source == "rest"
    assert tracker.last_replan_collides is False


def test_failed_local_retries_straight_then_random_seed(monkeypatch):
    """EGO-Planner v2 planFromLocalTraj (warm start, straight, widened random seed) plus our A*
    seed before the random one, all aiming at the same local target; the failure count resets."""
    import sobits_intball2_gnc_cpp as core

    tf = _IdealTrackingTf()
    tracker = _make_tracker(
        tf, face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL, local_piece_length_m=1.5,
        obstacle_grid=core.OccupancyGrid(0.1, 0.2))
    planner = tracker._planner
    state = tracker._reference_start_state()
    cursor = planner._global_search_t
    real = planner._build_local_seeded
    calls = []

    def scripted(*args):
        calls.append((args[-1], args[-2], planner._global_search_t))
        if args[-1] != "random":  # scripted: only the last tier solves
            raise MincoInfeasibleError("scripted")
        return real(*args)

    monkeypatch.setattr(planner, "_build_local_seeded", scripted)
    planner.build_local(*state)
    assert [c[0] for c in calls] == ["warm", "straight", "astar", "random"]
    assert calls[3][1] >= 1  # random seed widened
    assert all(c[2] == cursor for c in calls)
    assert planner._replan_failures == 0

    def always_fail(*_args):
        raise MincoInfeasibleError("scripted")

    monkeypatch.setattr(planner, "_build_local_seeded", always_fail)
    with pytest.raises(MincoInfeasibleError):
        planner.build_local(*state)
    assert planner._replan_failures == 1


def _reference_route_planner(route, **kwargs):
    import sobits_intball2_gnc_cpp as core
    from sobits_intball2_gnc.guidance.local_planner.minco_local_planner import MincoLocalPlanner
    return MincoLocalPlanner(
        route[0], np.zeros(3), route[-1], Q0, None, None, kwargs.pop("route_waypoints", None),
        4.0, 0.3, 1.0, None, True, (1.0, 0.0, 0.0), 0.15, 1.5,
        core.OccupancyGrid(0.05, 0.2), 0.2, None, **kwargs)


_L_ROUTE = [np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.8, 0.0]), np.array([3.0, 0.0, 0.0])]


def test_a_reference_route_makes_the_global_a_min_jerk_through_it():
    """docs/minco_astar_reference_global.md 2: the global follows the shared
    A* route instead of the chord, so the local target is already clear of the
    obstacles seen before departure."""
    from sobits_intball2_gnc.guidance.trajectory.reference_polynomial import MinJerkReference

    planner = _reference_route_planner(_L_ROUTE, reference_route=_L_ROUTE)
    assert isinstance(planner.global_trajectory, MinJerkReference)
    traj = planner.global_trajectory
    samples = np.array([traj.sample(t)[0]
                        for t in np.linspace(0.0, traj.global_total_duration, 400)])
    assert np.min(np.linalg.norm(samples - _L_ROUTE[1], axis=1)) < 0.02, "it rounds the corner"
    assert np.max(samples[:, 1]) > 0.5, "the chord would stay on y=0"


def test_the_local_target_lies_on_the_reference_route():
    planner = _reference_route_planner(_L_ROUTE, reference_route=_L_ROUTE)
    planner._planning_horizon_m = 1.0
    target = planner._get_local_target(_L_ROUTE[0])[0]
    traj = planner.global_trajectory
    samples = np.array([traj.sample(t)[0]
                        for t in np.linspace(0.0, traj.global_total_duration, 800)])
    assert np.min(np.linalg.norm(samples - target, axis=1)) < 1e-3  # sampling step
    assert target[1] > 0.1, "already heading around the corner, not down the chord"


def test_the_reference_global_speed_is_capped_at_the_local_max():
    planner = _reference_route_planner(_L_ROUTE, reference_route=_L_ROUTE)
    traj = planner.global_trajectory
    ts = np.linspace(0.0, traj.global_total_duration, 400)
    assert max(np.linalg.norm(traj.sample(t)[1]) for t in ts) <= 0.15 + 1e-9


def test_a_reference_route_and_via_waypoints_are_exclusive():
    with pytest.raises(ValueError):
        _reference_route_planner(_L_ROUTE, reference_route=_L_ROUTE,
                                 route_waypoints=[[1.0, 1.0, 1.0]])


def test_the_route_ends_are_taken_from_the_actual_start_and_goal():
    """The route is planned before the final TF read, so its own ends are stale."""
    moved = [_L_ROUTE[0] + 0.05, _L_ROUTE[1], _L_ROUTE[2]]
    planner = _reference_route_planner(moved, reference_route=_L_ROUTE)
    np.testing.assert_allclose(planner.global_trajectory.sample(0.0)[0], moved[0], atol=1e-6)


def test_without_a_reference_route_the_global_is_still_minco():
    from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

    planner = _reference_route_planner(_L_ROUTE)
    assert isinstance(planner.global_trajectory, MincoTrajectory)


def test_a_larger_hard_clearance_pushes_the_local_plan_further_from_the_box():
    """penalties.cpp's hard term is the enforced floor; the soft one saturates
    and only expresses a preference."""
    import sobits_intball2_gnc_cpp as core

    def closest(clearance):
        grid = core.OccupancyGrid(0.1, 0.2)
        grid.add_points(np.array(
            [[1.5, y, z] for y in np.arange(-0.4, 0.41, 0.05)
             for z in np.arange(-0.4, 0.41, 0.05)]).ravel().tolist())
        tracker = _make_tracker(
            _IdealTrackingTf(), p_target=np.array([3.0, 0.0, 0.0]), face_travel=True,
            local_max_vel=FACE_TRAVEL_MAX_VEL, local_piece_length_m=1.5,
            obstacle_grid=grid, obstacle_clearance_soft=0.3,
            obstacle_clearance=clearance)
        local = tracker.local_trajectory
        points = np.array([local.sample(t)[0] for t in
                           np.linspace(0.0, local.global_total_duration, 200)])
        return float(np.min(np.abs(points[:, 0] - 1.5)))

    assert closest(0.25) > closest(0.05)


@pytest.mark.parametrize("clearance", [0.3, 0.4, 0.0, -0.1])
def test_a_hard_clearance_not_below_the_soft_one_is_rejected(clearance):
    """At or above the soft clearance, penalties.cpp silently drops the soft term."""
    import sobits_intball2_gnc_cpp as core

    with pytest.raises(ValueError):
        _make_tracker(
            _IdealTrackingTf(), face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL,
            local_piece_length_m=1.5, obstacle_grid=core.OccupancyGrid(0.1, 0.2),
            obstacle_clearance_soft=0.3, obstacle_clearance=clearance)


def test_the_hard_clearance_defaults_to_the_binding_value():
    """None leaves bindings.cpp's 0.1 in place, so existing goals are unchanged."""
    import sobits_intball2_gnc_cpp as core

    tracker = _make_tracker(
        _IdealTrackingTf(), face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL,
        local_piece_length_m=1.5, obstacle_grid=core.OccupancyGrid(0.1, 0.2))
    assert tracker._planner._obstacle_clearance is None


def test_the_obstacle_lbfgs_delta_reaches_only_the_obstacle_solves(monkeypatch):
    """The global and obstacle-free solves keep the solver's 1e-8."""
    import sobits_intball2_gnc_cpp as core

    calls = []
    real = core.plan_minco

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "plan_minco", recording)
    _make_tracker(
        _IdealTrackingTf(), face_travel=True, local_max_vel=FACE_TRAVEL_MAX_VEL,
        local_piece_length_m=1.5, obstacle_grid=core.OccupancyGrid(0.1, 0.2),
        obstacle_lbfgs_delta=1e-3)
    with_grid = [c for c in calls if c.get("grid") is not None]
    assert with_grid and all(c["lbfgs_delta"] == 1e-3 for c in with_grid)
    assert all("lbfgs_delta" not in c for c in calls if c.get("grid") is None)


def test_the_wrench_use_is_computed_off_the_setpoint_loop(monkeypatch):
    tracker = _make_tracker(_IdealTrackingTf())
    tracker._solve_local_in_background(tracker._reference_start_state())
    result = tracker._pending_result
    assert result is not None and result[0].planned_wrench_use is None  # no envelope configured here
    result[0].planned_wrench_use = 0.42
    monkeypatch.setattr(tracker, "_planned_wrench_use",
                        lambda *a, **k: pytest.fail("wrench use computed in the setpoint loop"))
    tracker._adopt_local(result, 0.0, "async")
    assert tracker.last_replan_wrench_use == 0.42


def test_the_wrench_use_report_can_be_switched_off(monkeypatch):
    tracker = _make_tracker(_IdealTrackingTf(), report_wrench_use=False)
    monkeypatch.setattr(tracker, "_planned_wrench_use",
                        lambda *a, **k: pytest.fail("wrench use computed while switched off"))
    tracker._solve_local_in_background(tracker._reference_start_state())
    tracker._adopt_local(tracker._pending_result, 0.0, "async")
    assert tracker.last_replan_occurred and tracker.last_replan_wrench_use is None
