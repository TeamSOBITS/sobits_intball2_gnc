"""Unit tests for ReplanningTrajectoryTracker (ROS-agnostic, no rclpy).

Covers the specific decisions this class implements (docs/
guidance_realtime_replanning_design.md, and the archived decision memos it
cites in its own module docstring): the replan-cadence counter, the
distance/staleness fallback latch (one-way), and the mandatory max_accel.
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_trajectory_tracker import (
    ReplanningTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.utils.trajectory import Trajectory

P0 = [0.0, 0.0, 0.0]
P_TARGET = [2.0, 0.0, 0.0]
TARGET_SPEED = 0.5
MAX_ACCEL = 0.1 / 4.5


class _Vel:
    def __init__(self, vel):
        self.vel = vel


def _flat_trajectory():
    """A trivial single-segment stub Trajectory long enough that none of
    these tests reach its end -- only replace_coeffs()'s call count/args are
    under test here, not the sampled values."""
    coeffs = np.zeros((1, 3, 8))
    return Trajectory([P0, P_TARGET], [100.0], coeffs)


def _make_tracker(pose_fn, tf_fresh_fn=lambda stamp: True,
                   velocity_fn=lambda: _Vel([0.0, 0.0, 0.0]),
                   distance_fallback_m=0.3, replan_every_n_ticks=5,
                   max_accel=MAX_ACCEL):
    return ReplanningTrajectoryTracker(
        _flat_trajectory(), P_TARGET, pose_fn, tf_fresh_fn, velocity_fn,
        target_speed=TARGET_SPEED, max_accel=max_accel,
        distance_fallback_m=distance_fallback_m,
        replan_every_n_ticks=replan_every_n_ticks,
    )


def test_max_accel_none_is_rejected():
    with pytest.raises(ValueError):
        _make_tracker(pose_fn=lambda: ([0.0, 0.0, 0.0], [0, 0, 0, 1], 1.0),
                      max_accel=None)


def test_replans_only_every_nth_tick():
    calls = {"n": 0}

    def pose_fn():
        calls["n"] += 1
        return [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(calls["n"])

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=5)
    for t in range(4):
        tracker.sample(float(t))
    assert calls["n"] == 0  # not yet the 5th tick
    tracker.sample(4.0)
    assert calls["n"] == 1  # 5th sample() call triggers exactly one replan attempt


def test_falls_back_once_within_distance_threshold_and_latches():
    pose_calls = {"n": 0}

    def pose_fn():
        pose_calls["n"] += 1
        # Already within distance_fallback_m of P_TARGET=[2,0,0].
        return [1.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(pose_calls["n"])

    tracker = _make_tracker(pose_fn, distance_fallback_m=0.3, replan_every_n_ticks=1)
    tracker.sample(0.0)
    assert pose_calls["n"] == 1
    # Fallback latched -- further ticks must never call pose_fn again, even
    # though replan_every_n_ticks would otherwise fire every tick.
    tracker.sample(1.0)
    tracker.sample(2.0)
    assert pose_calls["n"] == 1


def test_does_not_fall_back_while_outside_distance_threshold():
    pose_calls = {"n": 0}

    def pose_fn():
        pose_calls["n"] += 1
        # Far from P_TARGET=[2,0,0] -- outside distance_fallback_m.
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(pose_calls["n"])

    tracker = _make_tracker(pose_fn, distance_fallback_m=0.3, replan_every_n_ticks=1)
    tracker.sample(0.0)
    tracker.sample(1.0)
    tracker.sample(2.0)
    assert pose_calls["n"] == 3  # kept re-planning, no latch


def test_falls_back_and_latches_when_tf_pose_is_none():
    pose_calls = {"n": 0}

    def pose_fn():
        pose_calls["n"] += 1
        return None

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=1)
    tracker.sample(0.0)
    assert pose_calls["n"] == 1
    tracker.sample(1.0)
    assert pose_calls["n"] == 1  # latched, no further attempts


def test_falls_back_and_latches_when_tf_pose_is_stale():
    pose_calls = {"n": 0}

    def pose_fn():
        pose_calls["n"] += 1
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(pose_calls["n"])

    tracker = _make_tracker(
        pose_fn, tf_fresh_fn=lambda stamp: False, replan_every_n_ticks=1,
    )
    tracker.sample(0.0)
    assert pose_calls["n"] == 1
    tracker.sample(1.0)
    assert pose_calls["n"] == 1  # latched, no further attempts


def test_replan_updates_underlying_trajectory_global_total_duration():
    def pose_fn():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=1, distance_fallback_m=0.0)
    before = tracker.total_duration
    tracker.sample(5.0)
    after = tracker.total_duration
    assert after != before
