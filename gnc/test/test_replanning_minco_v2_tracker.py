"""Unit tests for ReplanningMincoV2Tracker (docs/
2026-09-01_replanning_minco_v4_production_port_plan.md Phase 4).
"""
import numpy as np
import pytest

pytest.importorskip("minco_native_py")

from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v2_tracker import (
    ReplanningMincoV2Tracker,
)

P0 = [0.0, 0.0, 0.0]
P_TARGET = [3.0, 0.0, 0.0]
Q0 = [0.0, 0.0, 0.0, 1.0]
TARGET_SPEED = 0.5
MAX_ACCEL = 0.1 / 4.5


def _make_tracker(pose_fn, tf_fresh_fn=lambda stamp: True, **kwargs):
    kwargs.setdefault("global_replan_period", 2.0)
    kwargs.setdefault("t_local", 1.0)
    return ReplanningMincoV2Tracker(
        P0, P_TARGET, pose_fn, tf_fresh_fn, Q0,
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL, **kwargs
    )


def _still_pose_fn(pos=P0, quat=Q0):
    calls = {"n": 0}

    def pose_fn():
        calls["n"] += 1
        return list(pos), list(quat), float(calls["n"])

    return pose_fn


def test_t_local_must_be_strictly_between_zero_and_global_period():
    with pytest.raises(ValueError):
        _make_tracker(_still_pose_fn(), global_replan_period=1.0, t_local=1.0)
    with pytest.raises(ValueError):
        _make_tracker(_still_pose_fn(), global_replan_period=1.0, t_local=0.0)


def test_sample_produces_6dof_output():
    tracker = _make_tracker(_still_pose_fn())
    p, v, a, q = tracker.sample(0.1)

    assert p.shape == (3,)
    assert v.shape == (3,)
    assert a.shape == (3,)
    assert q.shape == (4,)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert tracker.last_fallback_reason is None
    assert tracker.last_local_fallback is False


def test_tf_stale_latches_and_freezes_output():
    tracker = _make_tracker(_still_pose_fn(), tf_fresh_fn=lambda stamp: False)
    p1, v1, a1, q1 = tracker.sample(0.1)
    assert tracker.last_fallback_reason == "tf_stale"

    p2, v2, a2, q2 = tracker.sample(0.2)
    assert tracker.last_fallback_reason == "tf_stale"
    assert np.allclose(p1, p2)
    assert np.allclose(v1, v2)
    assert np.allclose(q1, q2)


def test_pose_none_latches_tf_stale():
    tracker = _make_tracker(lambda: None)
    tracker.sample(0.1)
    assert tracker.last_fallback_reason == "tf_stale"


def test_distance_fallback_stops_global_replan_but_keeps_sampling():
    # Start already within distance_fallback_m of the target.
    near_target = [2.9, 0.0, 0.0]
    tracker = _make_tracker(
        _still_pose_fn(pos=near_target), distance_fallback_m=0.3,
        global_replan_period=0.2, t_local=0.1,
    )
    traj_before = tracker.trajectory

    for i in range(1, 6):
        tracker.sample(0.1 * i)

    # Global replan must never have fired again (latched stop).
    assert tracker.trajectory is traj_before
    assert tracker.last_fallback_reason is None  # not the same as tf_stale/minco_infeasible


def test_via_retirement_reduces_global_trajectory_waypoint_count():
    via = [1.0, 1.0, 0.0]  # dist(via, P_TARGET) = sqrt(2^2+1^2) ~= 2.236
    pending_pose_fn = _still_pose_fn(pos=[0.0, 0.0, 0.0])  # 3.0m from target, via pending
    tracker = _make_tracker(
        pending_pose_fn, route_waypoints=[via],
        global_replan_period=0.1, t_local=0.05,
    )
    tracker.sample(0.1)
    assert tracker.trajectory.num_waypoints == 3  # head, via, target

    past_via_pose_fn = _still_pose_fn(pos=[2.9, 0.0, 0.0])  # 0.1m from target, via retired
    tracker2 = _make_tracker(
        past_via_pose_fn, route_waypoints=[via],
        global_replan_period=0.1, t_local=0.05, distance_fallback_m=0.0,
    )
    tracker2.sample(0.1)
    assert tracker2.trajectory.num_waypoints == 2  # head, target (via retired)


def test_total_duration_and_repeated_t_do_not_corrupt_state():
    """GuidanceExecutor._run_trajectory clamps its `t` argument to
    `min(elapsed, tracker.total_duration)` and keeps calling sample() with
    that clamped (non-increasing) value while polling for real TF
    convergence -- this must be a safe no-op (hold last output), not force
    a bogus tiny dt through the KF/local segment every call."""
    tracker = _make_tracker(_still_pose_fn(), global_replan_period=0.5, t_local=0.2)

    tracker.sample(0.1)
    assert tracker.total_duration >= 0.1  # still comfortably ahead while replanning is active

    out1 = tracker.sample(0.2)
    clamped_t = tracker.total_duration  # what _run_trajectory would clamp `t` to
    out2 = tracker.sample(clamped_t)
    out3 = tracker.sample(clamped_t)  # repeated, as _run_trajectory would do while polling

    assert np.allclose(out2[0], out3[0])
    assert np.allclose(out2[1], out3[1])
    assert np.all(np.isfinite(out3[0]))
    assert np.all(np.isfinite(out3[1]))


def test_guard_falls_back_to_previous_velocity_on_large_jump():
    """A single-tick position glitch right at a global-replan tick must not
    feed a corrupted velocity estimate into the global replan (追試16)."""
    positions = iter([
        [0.1, 0.0, 0.0],
        [3.0, 5.0, 0.0],  # glitch: implausible jump -> large velocity estimate
        [0.3, 0.0, 0.0],
    ])

    def pose_fn():
        return list(next(positions)), list(Q0), 1.0

    tracker = _make_tracker(
        pose_fn, global_replan_period=0.2, t_local=0.1, guard_threshold=0.05,
    )
    tracker.sample(0.1)
    tracker.sample(0.2)  # global-replan tick coincides with the glitch
    # Must not raise/crash and must still produce a finite, sane output.
    p, v, a, q = tracker.sample(0.3)
    assert np.all(np.isfinite(p))
