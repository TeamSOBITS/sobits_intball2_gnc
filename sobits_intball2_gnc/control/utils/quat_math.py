#!/usr/bin/env python3
"""Pure quaternion/vector helpers shared across the control-law modules.

All quaternions are ``[x, y, z, w]`` (Hamilton convention). No ROS or numpy
state is held here -- these are plain-value functions only.
"""
import numpy as np


def deadband(vec, threshold):
    """Zero out per-axis components whose magnitude is below ``threshold``."""
    return np.where(np.abs(vec) < threshold, 0.0, vec)


def quat_conj(q):
    """Conjugate of quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    """Hamilton product a ⊗ b for quaternions [x, y, z, w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_rotate(q, v):
    """Rotate vector ``v`` by quaternion ``q`` [x, y, z, w]."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def geodesic_angle(q_a, q_b):
    """Angle [rad] between two orientations, robust to the double cover."""
    dot = np.clip(abs(np.dot(np.asarray(q_a, dtype=float),
                             np.asarray(q_b, dtype=float))), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_log(q):
    """Rotation-vector (axis*angle, [x,y,z]) logarithm of unit quaternion ``q``.

    Inverse of :func:`quat_exp`. Used to express a small orientation offset
    (e.g. relative to a reference attitude) as a 3-vector coordinate, for
    contexts (like feeding a spline/path library) that need attitude as a
    plain vector rather than a quaternion. Only meaningful as a coordinate
    for offsets that stay well away from a full-turn (2*pi) rotation -- see
    ``docs/2026-08-28_constrained_trajectory_generation_research.md``'s note
    on the right-Jacobian mismatch between rotation-vector derivatives and
    true angular velocity for large/varying-axis rotations.
    """
    q = np.asarray(q, dtype=float)
    v, w = q[:3], q[3]
    if w < 0.0:
        v, w = -v, -w
    sin_half = np.linalg.norm(v)
    angle = 2.0 * np.arctan2(sin_half, w)
    if sin_half < 1e-9:
        return np.zeros(3)
    return (v / sin_half) * angle


def quat_exp(rotvec):
    """Unit quaternion ``[x,y,z,w]`` exponential of rotation vector ``rotvec``.

    Inverse of :func:`quat_log`."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([axis[0] * np.sin(half), axis[1] * np.sin(half),
                      axis[2] * np.sin(half), np.cos(half)])


def slerp(q0, q1, t):
    """SLERP between unit quaternions ``q0``/``q1`` (both [x, y, z, w]) at
    ``t`` in [0, 1], resolving the double-cover sign so the interpolation
    always takes the short arc (unlike ``guidance.utils.attitude_reference
    ._slerp``, which assumes the caller already resolved that sign)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1
