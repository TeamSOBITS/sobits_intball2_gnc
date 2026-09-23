"""Contract test for guidance/trajectory_tracking/ implementations
(docs/architecture_guidelines.md 5 節: shared properties every
BaseTrajectoryTracker implementation must satisfy).

The property every tracker must share is simply "eventually converges to
p_target" -- StaticTrajectoryTracker trivially so (it just samples a
pre-built Trajectory), ReplanningMincoV3Tracker by construction (its last
local trajectory touches the goal and ends at rest there).
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import (
    ReplanningMincoV3Tracker,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.static_trajectory_tracker import (
    StaticTrajectoryTracker,
)
from sobits_intball2_gnc.guidance.trajectory.trajectory import Trajectory

P0 = [0.0, 0.0, 0.0]
P_TARGET = [2.0, 0.0, 0.0]
TARGET_SPEED = 0.5
MAX_ACCEL = 0.1 / 4.5


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


class _MincoV3Fixture:
    """Ideal tracking: the "TF" reports the tracker's own last setpoint,
    since v3 replans from its reference state rather than a trajectory the
    fixture could sample independently."""

    def __init__(self):
        pytest.importorskip("minco_native_py")
        self._t = 0.0
        self.tracker = ReplanningMincoV3Tracker(
            P0, P_TARGET, self._pose_fn, tf_fresh_fn=lambda stamp: True,
            q0=[0.0, 0.0, 0.0, 1.0], target_speed=TARGET_SPEED,
            max_accel=MAX_ACCEL,
        )

    def _pose_fn(self):
        p, _v, _a, _q = self.tracker._last_output
        return list(p), [0.0, 0.0, 0.0, 1.0], self._t

    def step(self, t):
        self._t = t


FIXTURE_FACTORIES = [_StaticFixture, _MincoV3Fixture]


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
