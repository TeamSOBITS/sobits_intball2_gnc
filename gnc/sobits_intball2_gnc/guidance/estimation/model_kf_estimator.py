#!/usr/bin/env python3
"""Model-based (constant-acceleration) Kalman filter state estimator
(ROS-agnostic, pure).

This is "対策案4" from the replanning_minco v4 lag/noise investigation
(``docs/archive/2026-09-17_replanning_minco_v4_lag_compensation_noise_robustness.md``):
unlike :class:`~sobits_intball2_gnc.guidance.estimation.velocity_estimator.
VelocityEstimator` (finite-difference + EMA on position alone, which has an
inherent lag that compounds badly with replanning-induced feedback loops),
this predicts forward using the *commanded* (feedforward) acceleration --
already computed every tick as a side effect of trajectory sampling
(``ReplanningTrajectoryTracker.sample()``'s ``a``) -- and corrects with the
noisy position (or rotation-vector) observation via a per-axis
constant-acceleration Kalman filter. Prediction is a pure integration step
(does not amplify observation noise); correction is a weighted fusion (does
not reintroduce raw finite-difference noise). Verified offline (32-48 case
sweeps each) against measurement noise, model mismatch (systematic gain
error, stochastic disturbance), R mistuning, attitude/multi-via
generalization, cruise-speed variation, and single-tick TF glitches
(``gnc/test/experiment_minco_native/bench_v10_lag_compensation_noise.cpp``
through ``bench_v18``-equivalent rotational extension,
``bench_v14_attitude_noise_kf.cpp``) -- this is the direct Python port for
``docs/2026-09-01_replanning_minco_v4_production_port_plan.md`` Phase 4.

Each axis is filtered independently (no cross-axis coupling), matching the
offline benches exactly -- used for both the 3 position axes and the 3
rotation-vector axes (two separate instances, one per state).
"""
import numpy as np


class ModelKfEstimator:
    """Per-axis constant-acceleration KF: state ``[pos, vel]``.

    Args:
        n_axes: number of independent per-axis filters (3 for position, 3
            for a rotation vector).
        q_accel_std: process noise standard deviation, modeling
            "commanded acceleration vs. actual acceleration" mismatch as a
            white-acceleration-noise constant-acceleration model (standard
            discretized CA-model Q). Verified robust across ~3-4 orders of
            magnitude around the offline-tuned value (bench_v10's
            q_accel_std sensitivity table) -- not a knife-edge tuning.
        r_std: assumed observation noise standard deviation (how much the
            filter trusts each new position/rotvec observation). Also
            verified robust across ~4 orders of magnitude of mistuning
            (bench_v11's R-mistuning sweep).
        initial_pos, initial_vel: shape ``(n_axes,)``, or ``None`` for
            zeros. Set from the first real observation via :meth:`reset`
            once available -- a filter that starts at an arbitrary zero
            pos/vel will simply correct itself over the next few ticks, no
            special first-sample handling needed (the KF's own correction
            step does this).
    """

    def __init__(self, n_axes, q_accel_std, r_std, initial_pos=None, initial_vel=None):
        self.n_axes = int(n_axes)
        self.q_accel_std = float(q_accel_std)
        self.r = float(r_std) ** 2
        self._pos = (
            np.zeros(self.n_axes) if initial_pos is None
            else np.array(initial_pos, dtype=float)
        )
        self._vel = (
            np.zeros(self.n_axes) if initial_vel is None
            else np.array(initial_vel, dtype=float)
        )
        # Per-axis 2x2 covariance, generously uninformative at construction
        # (the first correct() quickly pulls it down -- matches the offline
        # benches' cold-start behavior, never a source of the divergences
        # they found).
        self._P = np.tile(np.eye(2) * 1.0, (self.n_axes, 1, 1))

    def reset(self, pos, vel=None):
        """Re-baseline the filter on a known-good observation (e.g. the
        first real pose at goal start, or after a TF-stale recovery).
        Also resets covariance to the same uninformative prior used at
        construction, so a stale post-recovery covariance can't bias the
        next few corrections."""
        self._pos = np.array(pos, dtype=float)
        self._vel = np.zeros(self.n_axes) if vel is None else np.array(vel, dtype=float)
        self._P = np.tile(np.eye(2) * 1.0, (self.n_axes, 1, 1))

    def step(self, z, a_cmd, dt):
        """Predict forward by ``dt`` using ``a_cmd``, then correct with the
        noisy observation ``z``. Returns the corrected ``(pos, vel)``
        estimate, each shape ``(n_axes,)``.

        Args:
            z: noisy position/rotvec observation, shape ``(n_axes,)``.
            a_cmd: commanded (feedforward) acceleration used for the
                predict step, shape ``(n_axes,)`` -- the same quantity
                ``ReplanningTrajectoryTracker.sample()`` already computes
                as ``a`` (docs/
                2026-09-01_replanning_minco_v4_production_port_plan.md,
                "指令加速度の取得元は解決"節).
            dt: elapsed time since the previous ``step()`` call, seconds.
                Must be > 0.
        """
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        z = np.asarray(z, dtype=float)
        a_cmd = np.asarray(a_cmd, dtype=float)

        # Predict (per axis): [pos,vel] <- F@[pos,vel] + B*a_cmd, a pure
        # integration step -- doesn't amplify observation noise (this is
        # the structural reason MODEL_KF avoids the raw-finite-difference
        # noise-vs-lag tradeoff bench_v6/v8 found for EMA-on-position).
        pos_pred = self._pos + self._vel * dt + 0.5 * a_cmd * dt * dt
        vel_pred = self._vel + a_cmd * dt

        # Standard discretized constant-acceleration-model process noise
        # (continuous white acceleration noise intensity q_accel_std**2).
        q = self.q_accel_std ** 2
        Q = q * np.array([
            [dt ** 4 / 4.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, dt ** 2],
        ])
        F = np.array([[1.0, dt], [0.0, 1.0]])

        pos = np.empty(self.n_axes)
        vel = np.empty(self.n_axes)
        for i in range(self.n_axes):
            P_pred = F @ self._P[i] @ F.T + Q

            # Correct: H = [1, 0] (position-only observation).
            innovation = z[i] - pos_pred[i]
            S = P_pred[0, 0] + self.r
            K = P_pred[:, 0] / S  # (2,) Kalman gain
            state = np.array([pos_pred[i], vel_pred[i]]) + K * innovation
            self._P[i] = P_pred - np.outer(K, P_pred[0, :])
            pos[i], vel[i] = state

        self._pos, self._vel = pos, vel
        return self._pos.copy(), self._vel.copy()

    @property
    def pos(self):
        return self._pos.copy()

    @property
    def vel(self):
        return self._vel.copy()
