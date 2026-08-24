"""Contract test for guidance/trajectory_tracking/ implementations
(docs/architecture_guidelines.md 5 節: shared properties every
BaseTrajectoryTracker implementation must satisfy).

Both implementations wrap the same underlying Trajectory/generator/allocator
machinery (docs/guidance_realtime_replanning_design.md 4 節), so the property
every tracker must share is simply "eventually converges to p_target" --
StaticTrajectoryTracker trivially so (it just samples a pre-built Trajectory),
ReplanningTrajectoryTracker by construction (its distance-fallback (docs/
archive/achieved/2026-08-24_replanning_distance_fallback_decision.md) always
ends by handing off to a static leg aimed at p_target).
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_trajectory_tracker import (
    ReplanningTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.utils.trajectory import Trajectory

P0 = [0.0, 0.0, 0.0]
P_TARGET = [2.0, 0.0, 0.0]
TARGET_SPEED = 0.5
MAX_ACCEL = 0.1 / 4.5


class _Vel:
    def __init__(self, vel):
        self.vel = vel


def _build_trajectory(p0=P0, p_target=P_TARGET):
    waypoints = [p0, p_target]
    segment_times = HeuristicSegmentTimeAllocator(
        target_speed=TARGET_SPEED, max_accel=MAX_ACCEL,
    ).allocate(waypoints)
    coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)
    return Trajectory(waypoints, segment_times, coeffs)


class _StaticFixture:
    """No TF-clock stepping needed: sample(t) is a pure function of t."""

    def __init__(self):
        self.tracker = StaticTrajectoryTracker(_build_trajectory())

    def step(self, t):
        pass


class _ReplanningFixture:
    """Feeds the tracker's injected pose_fn/velocity_fn from the SAME
    underlying trajectory it re-plans against, so this exercises the real
    re-plan machinery (allocator + generator + replace_coeffs) each tick
    without needing a real dynamics simulation -- the "TF" here just reports
    exactly what the tracker's own last-planned trajectory predicts."""

    def __init__(self):
        self._traj = _build_trajectory()
        self._t = 0.0
        self.tracker = ReplanningTrajectoryTracker(
            self._traj, P_TARGET, self._pose_fn, tf_fresh_fn=lambda stamp: True,
            velocity_fn=self._velocity_fn, target_speed=TARGET_SPEED,
            max_accel=MAX_ACCEL, replan_every_n_ticks=1,
        )

    def _pose_fn(self):
        p, _v, _a, _q = self._traj.sample(self._t)
        return list(p), [0.0, 0.0, 0.0, 1.0], self._t

    def _velocity_fn(self):
        _p, v, _a, _q = self._traj.sample(self._t)
        return _Vel(list(v))

    def step(self, t):
        self._t = t


FIXTURE_FACTORIES = [_StaticFixture, _ReplanningFixture]


@pytest.mark.parametrize("make_fixture", FIXTURE_FACTORIES)
def test_reaches_target_by_total_duration(make_fixture):
    fixture = make_fixture()
    dt = 0.05
    t = 0.0
    p, v = None, None
    while t < fixture.tracker.total_duration + 1.0:
        fixture.step(t)
        p, v, _a, _q = fixture.tracker.sample(t)
        t += dt
    assert np.allclose(p, P_TARGET, atol=1e-3)
    assert np.allclose(v, [0.0, 0.0, 0.0], atol=1e-3)


@pytest.mark.parametrize("make_fixture", FIXTURE_FACTORIES)
def test_sample_returns_p0_at_t_zero(make_fixture):
    fixture = make_fixture()
    fixture.step(0.0)
    p, _v, _a, _q = fixture.tracker.sample(0.0)
    assert np.allclose(p, P0, atol=1e-6)
