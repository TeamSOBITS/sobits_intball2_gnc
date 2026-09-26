"""Unit tests for control/utils/pose_control_law.py (plain-value, no ROS)."""
import numpy as np
from scipy.spatial.transform import Rotation

from sobits_intball2_gnc.control.utils.pose_control_law import (
    attitude_error_to_torque,
    position_error_to_force,
)

# Symmetric attitudes (identity, pure 90deg yaw, 180deg) make q and conj(q),
# or q1*q2 and q2*q1, coincide -- these don't.
BODY = Rotation.from_euler("ZYX", [90.0, 0.0, 25.0], degrees=True)
DESIRED = Rotation.from_euler("ZYX", [30.0, -15.0, 10.0], degrees=True)


def test_position_error_force_is_expressed_in_body_frame():
    kp, kd = 2.0, 0.5
    target, pos, vel = np.array([1.0, -0.5, 0.3]), np.zeros(3), np.array([0.1, 0.2, -0.1])
    force = position_error_to_force(kp, kd, target, pos, vel, BODY.as_quat(),
                                    max_force=np.inf)
    expected = BODY.inv().apply(kp * (target - pos) - kd * vel)
    assert np.allclose(force, expected, atol=1e-12)


def test_attitude_torque_opposes_a_body_frame_offset_from_non_identity_target():
    kp = 3.0
    offset_rotvec_body = np.radians(20.0) * np.array([0.6, -0.48, 0.64])
    current = DESIRED * Rotation.from_rotvec(offset_rotvec_body)
    torque = attitude_error_to_torque(kp, 0.0, DESIRED.as_quat(), current.as_quat(),
                                      np.zeros(3), max_torque=np.inf)
    angle = np.linalg.norm(offset_rotvec_body)
    expected = -kp * np.sin(0.5 * angle) * offset_rotvec_body / angle
    assert np.allclose(torque, expected, atol=1e-12)
