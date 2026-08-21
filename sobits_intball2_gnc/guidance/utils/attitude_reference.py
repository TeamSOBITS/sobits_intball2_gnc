#!/usr/bin/env python3
"""Velocity-direction attitude reference (ROS-agnostic).

Computes a desired orientation quaternion ``q_des`` that faces a fixed body
axis (default: body +X) toward the direction of travel, given ``v_des(t)``.
Below a low-speed threshold the previous ``q_des`` is held unchanged (see
Aerostack2's ``yaw_threshold``) so attitude doesn't chatter at rest/low speed.

Kept independent from :mod:`sobits_intball2_gnc.guidance.utils.trajectory` so
this "face direction of travel" policy can later be swapped for another one
(face a fixed camera offset, look at a target point, etc. -- see
``docs/future_design_notes.md`` 2-2) without touching trajectory sampling.
The other two policies are not implemented here yet: their input contracts
(camera-offset config key, look-at target source) are still undecided, and
``docs/architecture_guidelines.md`` 3節 calls for fixing a shared interface
only once at least two concrete policies exist -- committing to one now would
just bake in this policy's assumptions.

Quaternions are ``[x, y, z, w]`` and represent the body's orientation in the
reference frame (i.e. ``quat_rotate(q, v_body) == v_ref``), matching
:mod:`sobits_intball2_gnc.control.utils.quat_math`.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def compute_q_des(v_des, prev_q_des, speed_threshold, forward_axis=(1.0, 0.0, 0.0),
                   dt=None, max_angular_rate=None):
    """Return ``q_des`` that points ``forward_axis`` (body frame) along ``v_des``.

    ``v_des`` is a reference-frame velocity vector. If its norm is below
    ``speed_threshold``, ``prev_q_des`` is returned unchanged (``None`` falls
    back to the identity quaternion, e.g. for the very first sample).

    The rotation about ``forward_axis`` itself (roll) is not constrained by
    the pointing task -- only the pointing direction is specified, so any
    rotation about ``forward_axis`` satisfies it equally. When ``prev_q_des``
    is given, that free roll is filled in to match ``prev_q_des``'s own
    rotation about ``forward_axis`` (a twist/swing decomposition of the
    "shortest arc" solution's deviation from ``prev_q_des``, see
    ``_preserve_roll``), rather than the "shortest arc" convention's
    incidental value -- which is fixed purely from ``forward_axis`` and
    ``v_des`` and ignores the vehicle's actual current attitude entirely, so
    it can differ arbitrarily from ``prev_q_des``'s roll and force a large,
    functionally unnecessary roll rotation to chase it (observed for
    ``pre_align``: a spurious ~45 degree roll excursion during what should
    have been a pure heading change, see
    docs/2026-08-21_tf_correction_align_optimization.md 8節). When
    ``prev_q_des`` is ``None`` there is nothing to preserve roll relative to,
    so the plain shortest-arc solution is returned as before.

    ``dt``/``max_angular_rate`` (both optional, rad/s for the latter): when
    given (and ``prev_q_des`` is not ``None``), the raw shortest-arc target is
    rate-limited -- stepped from ``prev_q_des`` toward it by at most
    ``max_angular_rate * dt`` -- instead of being returned directly. Without
    this, a direction reversal in ``v_des`` (e.g. near a curve's start, where
    the tangent can differ sharply from the vehicle's current heading) makes
    ``q_des`` jump by up to 180 degrees in a single tick; the P+D attitude
    controller's restoring torque is weakest exactly there (proportional to
    the quaternion error's vector part, i.e. roughly sin(error/2), which is
    smallest near a fully antipodal error), so a large instantaneous jump
    produces a large, slow-to-clear tracking error rather than a quick
    correction (confirmed in sim: 178 degree peak error, ~30s to converge --
    see docs/trajectory_force_duration_investigation.md 6-3). Limiting the
    *reference*'s rate of change keeps the commanded error small enough that
    the controller's (comparatively weak, still-P+D) torque stays effective
    throughout, rather than relying on ever-larger gains to compensate for a
    reference that can jump arbitrarily far in one step.
    """
    v_des = np.asarray(v_des, dtype=float)
    speed = np.linalg.norm(v_des)

    if speed < speed_threshold:
        return IDENTITY_QUAT.copy() if prev_q_des is None else np.asarray(prev_q_des, dtype=float)

    forward_axis = np.asarray(forward_axis, dtype=float)
    target = _shortest_arc_quat(forward_axis, v_des / speed)

    if prev_q_des is None:
        return target

    prev_q_des = np.asarray(prev_q_des, dtype=float)
    target = _preserve_roll(target, forward_axis, prev_q_des)

    if dt is None or max_angular_rate is None:
        return target

    return _rate_limited_step(prev_q_des, target, max_angular_rate * dt)


def compute_look_at_quat(own_pos, target_pos, forward_axis=(1.0, 0.0, 0.0)):
    """Return the quaternion that points ``forward_axis`` (body frame) at
    ``target_pos`` from ``own_pos`` (both in the same reference frame).

    A one-shot version of the "look at a point" policy (docs/
    movement_mode_design.md): unlike :func:`compute_q_des`, there is no
    ``prev_q_des``/rate-limiting here -- callers that need continuous
    tracking of a moving target must call this repeatedly themselves.

    Raises ``ValueError`` if ``own_pos`` and ``target_pos`` coincide (no
    direction to look at).
    """
    own_pos = np.asarray(own_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    direction = target_pos - own_pos
    distance = np.linalg.norm(direction)
    if distance < 1e-9:
        raise ValueError("own_pos and target_pos coincide; no look-at direction")
    return _shortest_arc_quat(np.asarray(forward_axis, dtype=float), direction / distance)


def compute_camera_relative_quat(q_target, from_axis, to_axis):
    """Return ``q`` such that ``to_axis`` (body frame) ends up pointing where
    ``from_axis`` would have pointed under ``q_target``.

    Use case: ``align_at_arrival`` wants "whatever the main camera would
    have seen facing ``q_target``, but through a different camera" (docs/
    movement_mode_design.md). ``from_axis``/``to_axis`` are two body-frame
    unit vectors (e.g. ``camera_forward_axis["main"]``/``["stereo"]``).
    When they're equal this returns ``q_target`` unchanged.
    """
    offset = _shortest_arc_quat(
        np.asarray(from_axis, dtype=float), np.asarray(to_axis, dtype=float)
    )
    return quat_mul(np.asarray(q_target, dtype=float), quat_conj(offset))


def _rate_limited_step(q_from, q_to, max_angle):
    """Step from ``q_from`` toward ``q_to`` by at most ``max_angle`` radians.

    ``max_angle`` is measured as the geodesic angle between the two
    orientations (``2*arccos(|dot(q_from, q_to)|)``, matching the metric used
    for tracking-error measurement elsewhere in this package). Returns
    ``q_to`` unchanged if it is already within ``max_angle`` of ``q_from``.
    """
    dot = np.dot(q_from, q_to)
    if dot < 0.0:
        # Double cover: q_to and -q_to represent the same orientation: pick
        # whichever sign is the shorter arc from q_from.
        q_to = -q_to
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    if angle <= max_angle or angle < 1e-9:
        return q_to
    return _slerp(q_from, q_to, max_angle / angle)


def _slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions ``q0``/``q1``.

    Assumes the caller has already resolved the double-cover sign ambiguity
    (``dot(q0, q1) >= 0``), so this always takes the shorter arc.
    """
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        # Nearly identical: linear interpolation avoids a near-zero sin(theta_0)
        # division below.
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def _preserve_roll(q_ref, axis, reference_q):
    """Re-express ``q_ref``'s free rotation about ``axis`` to match ``reference_q``.

    ``q_ref`` points ``axis`` (body frame) somewhere; any additional rotation
    about ``axis`` applied to the body *before* ``q_ref`` leaves that pointing
    unchanged (rotating a vector about itself is a no-op), so the family of
    orientations satisfying the same pointing constraint as ``q_ref`` is
    ``q_ref ⊗ twist(phi)`` for a free angle ``phi`` about ``axis``.

    ``q_diff = q_ref⁻¹ ⊗ reference_q`` is the body-frame rotation that would
    turn ``q_ref`` into ``reference_q``. Its component about ``axis`` (the
    "twist", via the standard twist/swing decomposition -- project the vector
    part onto ``axis``, keep the scalar part, renormalize) is exactly the
    ``phi`` that makes ``q_ref ⊗ twist`` match ``reference_q``'s own rotation
    about ``axis``, while the remaining "swing" component (unavoidable to
    satisfy the pointing constraint) is dropped. Degenerate cases (``q_diff``
    has no measurable twist about ``axis``, e.g. it is a pure swing) fall
    back to ``q_ref`` unchanged -- there is nothing to align.
    """
    q_diff = quat_mul(quat_conj(q_ref), reference_q)
    proj = np.dot(q_diff[:3], axis) * axis
    twist = np.array([proj[0], proj[1], proj[2], q_diff[3]])
    norm = np.linalg.norm(twist)
    if norm < 1e-9:
        return q_ref
    twist = twist / norm
    return quat_mul(q_ref, twist)


def _shortest_arc_quat(a, b):
    """Quaternion ``q`` with minimal rotation such that ``quat_rotate(q, a) == b``.

    ``a`` and ``b`` must be unit vectors.
    """
    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if dot < -1.0 + 1e-9:
        # a and b point in opposite directions: cross/dot degenerate to zero,
        # so pick an arbitrary axis orthogonal to a for the 180-degree turn.
        ortho = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(a, [0.0, 1.0, 0.0])
        ortho = ortho / np.linalg.norm(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0])

    q = np.array([cross[0], cross[1], cross[2], dot + 1.0])
    return q / np.linalg.norm(q)
