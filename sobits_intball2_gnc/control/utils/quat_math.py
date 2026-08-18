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
