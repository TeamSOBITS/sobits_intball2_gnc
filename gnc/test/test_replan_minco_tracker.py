"""Unit tests for ReplanMincoTracker (docs/
2026-09-20_ego_v2_style_replan_migration_plan.md).
"""
import threading

import numpy as np
import pytest

pytest.importorskip("minco_native_py")

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


def test_async_collision_seen_during_an_older_solve_solves_again_before_deciding():
    tf = _IdealTrackingTf()
    tracker, collision = _async_tracker_with_scripted_collision(tf)
    gate = _gate_local_builds(tracker)
    tracker._start_background_replan()
    older = tracker._pending_thread

    collision["ahead_s"] = 1.0
    t = _tick(tracker, tf, 0.0)
    assert tracker._pending_thread is older

    gate.set()
    older.join()
    t = _tick(tracker, tf, t)
    assert tracker.emergency_stops == 0
    assert tracker._pending_thread is not None and tracker._pending_thread is not older

    collision["ahead_s"] = None
    tracker._pending_thread.join()
    t = _tick(tracker, tf, t)
    assert tracker.emergency_stops == 0
