#!/usr/bin/env python3
"""Body-frame relative goal -> absolute goal, JAXA ctl_only's convention.

``pos_att_profiler.cpp`` (``setProfilePos``/``setProfileAtt``) composes
``r1 = r0 + q0 * dr`` and ``q1 = q0 * dq`` from the pose at goal receipt
(``docs/archive/achieved/2026-09-29_move_relative_design.md``).
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_mul, quat_rotate


def compose_relative_target(p0, q0, dp, dq):
    """Return ``(p, q)``: ``dp``/``dq`` (body frame of ``q0``) applied to ``(p0, q0)``."""
    q0 = np.asarray(q0, dtype=float)
    p = np.asarray(p0, dtype=float) + quat_rotate(q0, np.asarray(dp, dtype=float))
    q = quat_mul(q0, np.asarray(dq, dtype=float))
    return p, q / np.linalg.norm(q)
