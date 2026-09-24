"""Unit tests for the JAXA-ported stopping profile (plain-value, no ROS)."""
import math

import numpy as np
import pytest

from sobits_intball2_gnc.common.utils.stopping_profile import (
    StoppingProfile,
    effective_fmax,
    effective_tmax,
)
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)

MASS = 3.216
INERTIA = 0.0136
ETA = 0.7
IDENTITY = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def allocator():
    return ThrustAllocator(minimax_objective=True)


def _profile(allocator, v0, w0=(0.0, 0.0, 0.0), q0=IDENTITY, r0=(1.0, 2.0, 3.0)):
    return StoppingProfile(r0, v0, q0, w0, allocator, MASS, INERTIA, ETA)


def _envelope_max_force(allocator, d, margin):
    F, g = wrench_envelope_halfspaces(allocator.A, allocator.fj_max, safety_margin=margin)
    s = F @ np.concatenate([d, np.zeros(3)])
    return float(np.min(g[s > 1e-12] / s[s > 1e-12]))


@pytest.mark.parametrize("d", [
    [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],
    list(np.ones(3) / math.sqrt(3.0)),
])
def test_effective_fmax_matches_envelope(allocator, d):
    d = np.asarray(d)
    assert math.isclose(
        effective_fmax(allocator, d, 0.181, ETA),
        _envelope_max_force(allocator, d, ETA), rel_tol=1e-3,
    )


def test_effective_fmax_is_independent_of_reference_fmax(allocator):
    d = np.array([0.0, 1.0, 0.0])
    assert math.isclose(effective_fmax(allocator, d, 0.181, ETA),
                        effective_fmax(allocator, d, 0.1, ETA), rel_tol=1e-4)


def test_effective_rejects_eta_at_or_above_one(allocator):
    with pytest.raises(ValueError):
        effective_fmax(allocator, [1.0, 0.0, 0.0], 0.181, 1.0)


def test_effective_tmax_positive(allocator):
    assert effective_tmax(allocator, [0.0, 0.0, 1.0], 8.1904e-3, ETA) > 0.0


def test_at_rest_holds_start_pose(allocator):
    prof = _profile(allocator, [0.0, 0.0, 0.0])
    assert prof.duration == pytest.approx(0.0)
    p, v, a, q, w, _ = prof.sample(0.0)
    np.testing.assert_allclose(p, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(v, 0.0)
    np.testing.assert_allclose(prof.r1, [1.0, 2.0, 3.0])


def test_translation_stop_distance_and_continuity(allocator):
    v0 = np.array([0.2, 0.0, 0.0])
    prof = _profile(allocator, v0)
    a_max = prof.a_max
    assert np.linalg.norm(prof.r1 - [1.0, 2.0, 3.0]) == pytest.approx(0.2 ** 2 / (2 * a_max))
    assert prof.duration == pytest.approx(0.2 / a_max)

    # JAXA posProfile's `t <= qtt` coast branch owns t == 0 exactly.
    _, vs, a0, *_ = prof.sample(1e-9)
    np.testing.assert_allclose(vs, v0, atol=1e-9)
    np.testing.assert_allclose(a0, -a_max * v0 / 0.2)

    p_end, v_end, a_end, *_ = prof.sample(prof.duration + 1.0)
    np.testing.assert_allclose(p_end, prof.r1)
    np.testing.assert_allclose(v_end, 0.0)
    np.testing.assert_allclose(a_end, 0.0)

    ts = np.linspace(0.0, prof.duration, 200)
    speeds = [np.linalg.norm(prof.sample(t)[1]) for t in ts]
    assert all(s1 <= s0 + 1e-12 for s0, s1 in zip(speeds, speeds[1:]))


def test_small_stop_below_threshold_jumps_to_stop_point(allocator):
    prof = _profile(allocator, [0.01, 0.0, 0.0])
    p, v, _, *_ = prof.sample(1e-9)
    np.testing.assert_allclose(p, prof.r1, atol=1e-9)
    np.testing.assert_allclose(v, 0.0)


def test_att_pos_coasts_translation_while_rotation_stops(allocator):
    v0 = np.array([0.0, 0.2, 0.0])
    w0 = np.array([0.0, 0.0, 0.3])
    prof = _profile(allocator, v0, w0=w0)
    t_rot = 0.3 / prof.wd_max
    assert prof.duration == pytest.approx(t_rot + 0.2 / prof.a_max)

    p, v, a, q, w, _ = prof.sample(0.5 * t_rot)
    np.testing.assert_allclose(p, np.array([1.0, 2.0, 3.0]) + v0 * 0.5 * t_rot)
    np.testing.assert_allclose(v, v0)
    np.testing.assert_allclose(a, 0.0)
    np.testing.assert_allclose(w, [0.0, 0.0, 0.15], atol=1e-9)

    _, _, _, q_end, w_end, _ = prof.sample(t_rot + 1e-9)
    assert geodesic_angle(q_end, prof.q1) < 1e-6
    np.testing.assert_allclose(w_end, 0.0, atol=1e-6)
    assert geodesic_angle(prof.q1, IDENTITY) == pytest.approx(
        0.5 * prof.wd_max * t_rot ** 2, rel=1e-6)

    stop_extra = np.linalg.norm(prof.r1 - [1.0, 2.0, 3.0]) - 0.2 ** 2 / (2 * prof.a_max)
    assert stop_extra == pytest.approx(0.0, abs=1e-9)
    p_end = prof.sample(prof.duration + 1.0)[0]
    np.testing.assert_allclose(p_end, prof.r1 + v0 * t_rot)


def test_decel_direction_uses_final_attitude(allocator):
    # Body rotated 90deg about z: world +x motion is body -y, the weak axis.
    q_yaw90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    rotated = _profile(allocator, [0.2, 0.0, 0.0], q0=q_yaw90)
    aligned = _profile(allocator, [0.0, 0.2, 0.0])
    assert rotated.a_max == pytest.approx(aligned.a_max, rel=1e-3)


def test_profile_does_not_alias_caller_state(allocator):
    r0 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.2, 0.0, 0.0])
    prof = _profile(allocator, v0, r0=r0)
    end_before = prof.sample(prof.duration)[0].copy()
    r0 += 5.0
    v0 *= 3.0
    np.testing.assert_allclose(prof.sample(prof.duration)[0], end_before)


def test_max_axis_force_caps_decel_along_that_axis(allocator):
    capped = StoppingProfile([0, 0, 0], [0.2, 0, 0], IDENTITY, [0, 0, 0],
                             allocator, MASS, INERTIA, ETA, max_axis_force=0.1)
    assert capped.a_max == pytest.approx(0.1 / MASS)
    uncapped = _profile(allocator, [0.2, 0.0, 0.0])
    assert uncapped.a_max * MASS > 0.1


def test_max_axis_force_leaves_weaker_direction_unchanged(allocator):
    capped = StoppingProfile([0, 0, 0], [0, 0.2, 0], IDENTITY, [0, 0, 0],
                             allocator, MASS, INERTIA, ETA, max_axis_force=0.1)
    assert capped.a_max == pytest.approx(_profile(allocator, [0.0, 0.2, 0.0]).a_max)
