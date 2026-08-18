#!/usr/bin/env python3
"""Pure pose-tracking control laws: (target, current) -> (force, torque).

Extracted out of :class:`PoseCorrector` so the same error-to-wrench math can
be reused by a future moving-target controller (e.g. Phase 3a/3b's
trajectory controller, see ``docs/main_plan.md``) without duplicating it.
This module only computes wrenches from poses/velocities already expressed
in a common reference frame -- it knows nothing about TF, liveness, or how a
target is chosen (that responsibility stays with the caller, e.g.
:class:`PoseCorrector`).
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul, quat_rotate


def position_error_to_force(kp_pos, kd_pos, target_pos, pos, vel, quat, max_force):
    """P+D position error (reference frame) rotated into the body frame.

    ``target_pos``/``pos``/``vel`` are reference-frame vectors; ``quat`` is
    the current orientation (reference frame -> body), used to express the
    result in body frame. Returns a clamped body-frame force (numpy array).
    """
    f_ref = kp_pos * (target_pos - pos) - kd_pos * vel
    f_body = quat_rotate(quat_conj(quat), f_ref)
    return np.clip(f_body, -max_force, max_force)


def attitude_error_to_torque(kp_att, kd_att, target_quat, quat, omega_err, max_torque):
    """P+D control on the quaternion error, expressed in the body frame.

    ``omega_err`` is the *relative* angular rate of the error (e.g. a finite
    difference of the quaternion error's vector part over time, see
    :class:`~sobits_intball2_gnc.control.utils.pose_corrector.PoseCorrector`)
    -- NOT the IMU's absolute angular rate. When the reference frame itself
    rotates (e.g. ``iss_body`` turning in the world, see
    docs/phase0_findings.md observations 8/11), holding a fixed relative
    attitude requires a non-zero *absolute* rate, so damping against the
    absolute rate fights the tracking motion and inflates the steady-state
    error (docs/phase0_5_findings.md). Damping the relative rate instead
    vanishes once tracking is locked, regardless of how fast the reference
    frame itself is turning.

    Returns a clamped body-frame torque (numpy array).
    """
    qe = quat_mul(quat_conj(target_quat), quat)
    sign = np.sign(qe[3] if qe[3] != 0.0 else 1.0)
    torque = -kp_att * sign * qe[:3] - kd_att * omega_err
    return np.clip(torque, -max_torque, max_torque)
