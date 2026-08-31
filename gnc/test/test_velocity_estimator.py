"""Unit tests for guidance/utils/velocity_estimator.py (ROS-agnostic, pure
finite-difference + EMA velocity estimate -- see
docs/guidance_velocity_estimator_design.md for the design this implements).
"""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.utils.velocity_estimator import VelocityEstimator


def test_initial_state_is_zero_velocity_before_any_update():
    est = VelocityEstimator()
    snap = est.get()
    assert snap.pos is None
    assert snap.stamp is None
    assert np.allclose(snap.vel, [0.0, 0.0, 0.0])


def test_first_update_reports_zero_velocity_no_prior_sample():
    est = VelocityEstimator()
    est.update([1.0, 2.0, 3.0], stamp=10.0)
    snap = est.get()
    assert snap.pos == [1.0, 2.0, 3.0]
    assert snap.stamp == 10.0
    assert np.allclose(snap.vel, [0.0, 0.0, 0.0])


def test_two_updates_yield_finite_difference_velocity_with_alpha_1():
    # alpha=1.0 disables EMA filtering, so the estimate should match the raw
    # finite difference exactly.
    est = VelocityEstimator(alpha=1.0)
    est.update([0.0, 0.0, 0.0], stamp=0.0)
    est.update([1.0, 0.0, 0.0], stamp=0.5)
    snap = est.get()
    assert np.allclose(snap.vel, [2.0, 0.0, 0.0])


def test_ema_blends_toward_new_sample_not_fully():
    est = VelocityEstimator(alpha=0.3)
    est.update([0.0, 0.0, 0.0], stamp=0.0)
    est.update([1.0, 0.0, 0.0], stamp=1.0)  # raw vel = [1, 0, 0]
    first = est.get().vel[0]
    assert 0.0 < first < 1.0  # blended, not the raw value
    assert np.isclose(first, 0.3 * 1.0 + 0.7 * 0.0)

    est.update([3.0, 0.0, 0.0], stamp=2.0)  # raw vel = [2, 0, 0]
    second = est.get().vel[0]
    assert np.isclose(second, 0.3 * 2.0 + 0.7 * first)


def test_stale_gap_beyond_max_dt_does_not_produce_velocity_spike():
    est = VelocityEstimator(alpha=1.0, max_dt=1.0)
    est.update([0.0, 0.0, 0.0], stamp=0.0)
    est.update([1.0, 0.0, 0.0], stamp=1.0)
    steady_vel = est.get().vel

    # TF stalls for 5s then bursts back with a large position jump -- dt
    # exceeds max_dt, so this must NOT be folded into the filtered estimate
    # as a huge finite-difference spike.
    est.update([50.0, 0.0, 0.0], stamp=6.0)
    snap = est.get()
    assert np.allclose(snap.vel, steady_vel)
    # the sample itself is still adopted as the new baseline
    assert snap.pos == [50.0, 0.0, 0.0]
    assert snap.stamp == 6.0

    # the next normal-cadence update resumes finite-differencing from this
    # new baseline, not from the pre-gap one.
    est.update([51.0, 0.0, 0.0], stamp=7.0)
    assert np.allclose(est.get().vel, [1.0, 0.0, 0.0])


def test_backwards_stamp_does_not_produce_velocity_spike():
    est = VelocityEstimator(alpha=1.0)
    est.update([0.0, 0.0, 0.0], stamp=10.0)
    est.update([1.0, 0.0, 0.0], stamp=11.0)
    steady_vel = est.get().vel

    # simulator restart: stamp jumps backwards.
    est.update([0.0, 0.0, 0.0], stamp=2.0)
    snap = est.get()
    assert np.allclose(snap.vel, steady_vel)
    assert snap.stamp == 2.0

    est.update([0.5, 0.0, 0.0], stamp=3.0)
    assert np.allclose(est.get().vel, [0.5, 0.0, 0.0])


def test_zero_dt_is_ignored_like_a_stale_gap():
    est = VelocityEstimator(alpha=1.0)
    est.update([0.0, 0.0, 0.0], stamp=5.0)
    est.update([0.0, 0.0, 0.0], stamp=5.0)  # duplicate stamp, dt == 0
    assert np.allclose(est.get().vel, [0.0, 0.0, 0.0])


def test_set_gains_updates_alpha_and_max_dt():
    est = VelocityEstimator(alpha=0.3, max_dt=1.0)
    est.set_gains(alpha=1.0, max_dt=2.0)
    assert est.alpha == pytest.approx(1.0)
    assert est.max_dt == pytest.approx(2.0)

    est.update([0.0, 0.0, 0.0], stamp=0.0)
    est.update([1.0, 0.0, 0.0], stamp=1.0)
    assert np.allclose(est.get().vel, [1.0, 0.0, 0.0])  # alpha=1.0 took effect


def test_set_gains_with_no_args_is_a_no_op():
    est = VelocityEstimator(alpha=0.3, max_dt=1.0)
    est.set_gains()
    assert est.alpha == pytest.approx(0.3)
    assert est.max_dt == pytest.approx(1.0)
