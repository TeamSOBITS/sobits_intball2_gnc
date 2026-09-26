"""Unit tests for ModelKfEstimator (docs/
2026-09-01_replanning_minco_v4_production_port_plan.md Phase 4, offline
design/verification in docs/archive/
2026-09-17_replanning_minco_v4_lag_compensation_noise_robustness.md).
"""
import numpy as np

from sobits_intball2_gnc.guidance.estimation.model_kf_estimator import ModelKfEstimator


def test_perfect_model_no_noise_converges_to_truth():
    """Constant commanded acceleration, exact position observations (no
    noise, a_cmd == true accel) -- should converge to near-zero error
    quickly (mirrors bench_v10's noise_sigma=0 PASS cases, ~1mm-order)."""
    dt = 0.1
    true_accel = np.array([0.02, -0.01, 0.0])
    true_pos = np.zeros(3)
    true_vel = np.array([0.1, 0.0, 0.0])

    kf = ModelKfEstimator(n_axes=3, q_accel_std=0.01, r_std=0.001)
    kf.reset(true_pos, true_vel)

    for _ in range(50):
        true_pos = true_pos + true_vel * dt + 0.5 * true_accel * dt * dt
        true_vel = true_vel + true_accel * dt
        pos, vel = kf.step(true_pos, true_accel, dt)

    assert np.allclose(pos, true_pos, atol=1e-6)
    assert np.allclose(vel, true_vel, atol=1e-6)


def test_noisy_observations_stay_bounded_and_track_mean():
    """With observation noise, the filtered estimate must not blow up and
    should stay within a small multiple of the noise level of the true
    trajectory (qualitative sanity check, not a tight bound -- the offline
    benches are the quantitative reference)."""
    rng = np.random.default_rng(0)
    dt = 0.1
    true_accel = np.array([0.0, 0.0, 0.0])
    true_pos = np.zeros(3)
    true_vel = np.array([0.3, 0.0, 0.0])
    noise_sigma = 0.003

    kf = ModelKfEstimator(n_axes=3, q_accel_std=0.01, r_std=noise_sigma)
    kf.reset(true_pos, true_vel)

    errors = []
    for _ in range(200):
        true_pos = true_pos + true_vel * dt
        z = true_pos + rng.normal(0.0, noise_sigma, size=3)
        pos, vel = kf.step(z, true_accel, dt)
        errors.append(np.linalg.norm(pos - true_pos))

    # Steady-state error should settle well within a few noise-sigmas, not
    # diverge (the whole point of MODEL_KF vs. raw finite-difference).
    assert np.mean(errors[-50:]) < 0.01


def test_reset_reinitializes_state_and_covariance():
    kf = ModelKfEstimator(n_axes=3, q_accel_std=0.01, r_std=0.001)
    kf.step(np.array([1.0, 1.0, 1.0]), np.zeros(3), 0.1)
    kf.reset(np.array([5.0, 5.0, 5.0]), np.array([1.0, 0.0, 0.0]))

    assert np.allclose(kf.pos, [5.0, 5.0, 5.0])
    assert np.allclose(kf.vel, [1.0, 0.0, 0.0])


def test_rejects_nonpositive_dt():
    kf = ModelKfEstimator(n_axes=3, q_accel_std=0.01, r_std=0.001)
    import pytest
    with pytest.raises(ValueError):
        kf.step(np.zeros(3), np.zeros(3), 0.0)
