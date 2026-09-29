"""Unit tests for guidance/utils/relative_target.py (plain-value, no ROS)."""
import math

import numpy as np

from sobits_intball2_gnc.guidance.utils.relative_target import compose_relative_target

IDENTITY = [0.0, 0.0, 0.0, 1.0]
YAW_90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]


def test_zero_relative_keeps_pose():
    p, q = compose_relative_target([1.0, 2.0, 3.0], YAW_90, [0.0, 0.0, 0.0], IDENTITY)
    assert np.allclose(p, [1.0, 2.0, 3.0])
    assert np.allclose(q, YAW_90)


def test_translation_with_identity_attitude_is_plain_offset():
    p, q = compose_relative_target([1.0, 2.0, 3.0], IDENTITY, [0.3, -0.2, 0.5], IDENTITY)
    assert np.allclose(p, [1.3, 1.8, 3.5])
    assert np.allclose(q, IDENTITY)


def test_translation_is_in_body_frame():
    p, _ = compose_relative_target([0.0, 0.0, 0.0], YAW_90, [1.0, 0.0, 0.0], IDENTITY)
    assert np.allclose(p, [0.0, 1.0, 0.0])


def test_rotation_composes_on_the_right():
    roll_90 = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
    _, q = compose_relative_target([0.0, 0.0, 0.0], YAW_90, [0.0, 0.0, 0.0], roll_90)
    # yaw then body-x roll: body x stays world +y, body z goes to world +x.
    expected = [0.5, 0.5, 0.5, 0.5]
    assert np.allclose(q, expected) or np.allclose(q, -np.array(expected))
