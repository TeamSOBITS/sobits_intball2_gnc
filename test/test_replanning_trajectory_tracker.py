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
from sobits_intball2_gnc.guidance.trajectory.trajectory import Trajectory

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
                   max_accel=MAX_ACCEL, via_waypoint=None):
    return ReplanningTrajectoryTracker(
        _flat_trajectory(), P_TARGET, pose_fn, tf_fresh_fn, velocity_fn,
        target_speed=TARGET_SPEED, max_accel=max_accel,
        distance_fallback_m=distance_fallback_m,
        replan_every_n_ticks=replan_every_n_ticks,
        via_waypoint=via_waypoint,
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
    assert tracker.last_fallback_reason == "distance"


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
    assert tracker.last_fallback_reason is None


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
    assert tracker.last_fallback_reason == "tf_stale"


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
    assert tracker.last_fallback_reason == "tf_stale"


def test_replan_updates_underlying_trajectory_global_total_duration():
    def pose_fn():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=1, distance_fallback_m=0.0)
    before = tracker.total_duration
    tracker.sample(5.0)
    after = tracker.total_duration
    assert after != before


def test_last_replan_occurred_true_only_on_a_replanning_tick():
    """A caller re-publishing an RViz preview on re-plan (docs/main_plan.md
    [G] "再計画軌道のRVizプレビュー更新") needs to distinguish a tick that
    actually re-planned from one that didn't -- ``last_replan_occurred``
    must be True only on the former, and reset back to False by the very
    next sample() call even if that next call doesn't re-plan either."""
    def pose_fn():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=5)
    for t in range(4):
        tracker.sample(float(t))
        assert tracker.last_replan_occurred is False
    tracker.sample(4.0)
    assert tracker.last_replan_occurred is True
    tracker.sample(5.0)
    assert tracker.last_replan_occurred is False  # not the 5th tick again yet


def test_last_replan_occurred_false_once_fallback_latches():
    def pose_fn():
        # Already within distance_fallback_m of P_TARGET=[2,0,0].
        return [1.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(pose_fn, distance_fallback_m=0.3, replan_every_n_ticks=1)
    tracker.sample(0.0)
    assert tracker.last_replan_occurred is True  # the fallback-triggering replan itself
    tracker.sample(1.0)
    assert tracker.last_replan_occurred is False  # latched, no further replans


def test_replan_tracks_large_disturbance_before_latching():
    """A collision that knocks the vehicle to a very different position
    (still outside distance_fallback_m of the target) must be picked up by
    the very next re-plan tick just like ordinary drift would be -- p_now
    comes straight from pose_fn each time, so an instantaneous jump is
    indistinguishable to this class from smooth motion between two ticks
    (docs/main_plan.md's outstanding "擬似衝突からの復帰再現" item)."""
    positions = iter([
        [1.0, 0.0, 0.0],   # tick 1: normal approach, 1.0m from P_TARGET=[2,0,0]
        [-3.0, 0.0, 0.0],  # tick 2: "collision" knock, 5.0m from target
    ])
    calls = {"n": 0}

    def pose_fn():
        calls["n"] += 1
        return list(next(positions)), [0.0, 0.0, 0.0, 1.0], float(calls["n"])

    tracker = _make_tracker(pose_fn, distance_fallback_m=0.3, replan_every_n_ticks=1)
    tracker.sample(0.0)
    assert np.allclose(tracker.trajectory.waypoints[0], [1.0, 0.0, 0.0])
    assert tracker.last_fallback_reason is None
    tracker.sample(1.0)
    assert np.allclose(tracker.trajectory.waypoints[0], [-3.0, 0.0, 0.0])
    assert tracker.last_replan_occurred is True
    assert tracker.last_fallback_reason is None  # still outside threshold -> kept replanning


def test_latched_fallback_does_not_recover_from_post_latch_disturbance():
    """Known one-way-latch tradeoff (module docstring): once the distance
    fallback trips near the target, sample() never calls pose_fn again for
    the rest of this goal. A collision that happens AFTER that point (e.g.
    bumped away right as it was arriving) is therefore invisible to this
    tracker and never triggers a corrective re-plan -- this is exactly the
    gap docs/main_plan.md flags as unverified: in the post-latch window,
    ReplanningTrajectoryTracker alone provides no protection against a
    pseudo-collision."""
    calls = {"n": 0}

    def pose_fn():
        calls["n"] += 1
        # Always reports right at P_TARGET=[2,0,0] -- latches on first sample().
        return [2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(calls["n"])

    tracker = _make_tracker(pose_fn, distance_fallback_m=0.3, replan_every_n_ticks=1)
    tracker.sample(0.0)
    assert tracker.last_fallback_reason == "distance"
    assert calls["n"] == 1
    # A "collision" here (if visible) would push the vehicle far from
    # P_TARGET and warrant a fresh re-plan -- but the latch means sample()
    # never reads pose_fn again to find out.
    tracker.sample(1.0)
    tracker.sample(2.0)
    assert calls["n"] == 1


# --- segment_time_infeasible fallback (condition 3) ---------------------
# docs/2026-08-25_v0_aware_time_allocation_lateral_velocity_fix.md 論点4:
# HeuristicSegmentTimeAllocator.allocate can raise SegmentTimeInfeasibleError
# once it accounts for v0's perpendicular component (T_min_perp) -- this
# tracker must catch it and latch the fallback like conditions 1/2, instead
# of letting it propagate out of sample().


def test_falls_back_and_latches_when_segment_time_is_infeasible():
    # d=1.0 (P0=[1,0,0] to P_TARGET=[2,0,0]), v0=[0.5, 0.5, 0]:
    # v_parallel=0.5 -> T_max=3*d/v_parallel=6.0; v_perp=0.5, a_max=0.1/4.5
    # -> T_min_perp=4*0.5/a_max=90.0, far above T_max -- infeasible.
    pose_calls = {"n": 0}

    def pose_fn():
        pose_calls["n"] += 1
        return [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], float(pose_calls["n"])

    tracker = _make_tracker(
        pose_fn, velocity_fn=lambda: _Vel([0.5, 0.5, 0.0]),
        distance_fallback_m=0.3, replan_every_n_ticks=1,
    )
    tracker.sample(0.0)
    assert pose_calls["n"] == 1
    assert tracker.last_fallback_reason == "segment_time_infeasible"
    assert tracker.last_replan_occurred is False  # the failed attempt never replaced the trajectory
    # Fallback latched -- further ticks must never call pose_fn again.
    tracker.sample(1.0)
    tracker.sample(2.0)
    assert pose_calls["n"] == 1


def test_trajectory_property_exposes_post_replan_state():
    def pose_fn():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(pose_fn, replan_every_n_ticks=1, distance_fallback_m=0.0)
    tracker.sample(5.0)
    assert np.allclose(tracker.trajectory.waypoints[0], [0.0, 0.0, 0.0])


# --- via_waypoint (docs/2026-08-25_guidance_waypoint_insertion_curve_verification.md) ---
# P_TARGET=[2,0,0], VIA=[1,1,0] -> dist(VIA, P_TARGET) = sqrt(2) ~= 1.4142.
# "Passed" fires once remaining distance-to-target undercuts that value.

VIA = [1.0, 1.0, 0.0]


def test_replan_routes_through_via_waypoint_while_pending():
    def pose_fn():
        # 2.0m from P_TARGET, still further than VIA's 1.4142m -> pending.
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(
        pose_fn, distance_fallback_m=0.0, replan_every_n_ticks=1, via_waypoint=VIA,
    )
    tracker.sample(0.0)
    assert np.allclose(tracker.trajectory.waypoints, [[0.0, 0.0, 0.0], VIA, P_TARGET])


def test_replan_drops_via_waypoint_once_passed():
    def pose_fn():
        # 0.1m from P_TARGET -- closer than VIA's own 1.4142m -> already passed.
        return [1.9, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(
        pose_fn, distance_fallback_m=0.0, replan_every_n_ticks=1, via_waypoint=VIA,
    )
    tracker.sample(0.0)
    assert np.allclose(tracker.trajectory.waypoints, [[1.9, 0.0, 0.0], P_TARGET])


def test_via_waypoint_passed_latch_does_not_re_arm_after_a_disturbance():
    """Once passed, a subsequent disturbance that pushes the vehicle back
    out beyond VIA's own distance-to-target must NOT re-insert the via leg
    -- this is a one-way latch, same shape as the distance/tf-stale fallback
    (module docstring): re-arming would risk flip-flopping the route."""
    positions = iter([
        [1.9, 0.0, 0.0],   # tick 1: already past VIA -> latches "passed"
        [-1.0, 0.0, 0.0],  # tick 2: knocked far away again (3.0m from target,
                           # further than VIA's 1.4142m) -- must stay 2-point.
    ])

    def pose_fn():
        return list(next(positions)), [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(
        pose_fn, distance_fallback_m=0.0, replan_every_n_ticks=1, via_waypoint=VIA,
    )
    tracker.sample(0.0)
    assert np.allclose(tracker.trajectory.waypoints, [[1.9, 0.0, 0.0], P_TARGET])
    tracker.sample(1.0)
    assert np.allclose(tracker.trajectory.waypoints, [[-1.0, 0.0, 0.0], P_TARGET])


def test_via_waypoint_none_reproduces_plain_two_point_route():
    def pose_fn():
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1.0

    tracker = _make_tracker(
        pose_fn, distance_fallback_m=0.0, replan_every_n_ticks=1, via_waypoint=None,
    )
    tracker.sample(0.0)
    assert np.allclose(tracker.trajectory.waypoints, [[0.0, 0.0, 0.0], P_TARGET])
