#!/usr/bin/env python3
"""Force/torque-aware combined position+attitude trajectory (ROS-agnostic).

Builds a single 6-DOF geometric path (3 position + 3 attitude, the latter a
rotation-vector offset from a reference orientation ``q0``, see
:func:`sobits_intball2_gnc.control.utils.quat_math.quat_log`/``quat_exp``)
through ``toppra`` (Pham & Pham, arXiv:1707.07239) and time-parameterizes it
subject to the vehicle's actual combined force+torque actuation envelope
(:mod:`~sobits_intball2_gnc.guidance.utils.actuation_envelope`, see below),
plus per-axis velocity limits -- see
``docs/2026-08-28_constrained_trajectory_generation_research.md`` for the
design rationale and the open questions this implementation resolves:

- Acceleration is constrained via the real actuator's achievable wrench
  region, not independent per-axis force/torque boxes (the original design
  here). :class:`~sobits_intball2_gnc.control.utils.thrust_allocator.
  ThrustAllocator`'s 8 fans each have a fixed thrust direction contributing
  to force AND torque simultaneously, so the two channels share one physical
  budget -- confirmed empirically (docs/
  2026-08-28_toppra_static_path_attitude_overshoot_incident.md "追記
  （2026-08-28 その2）"): even a planned path's feedforward-only wrench
  (zero tracking error) exceeded the true achievable region at ~92% of
  sampled points despite respecting each axis's own independent max. A
  ``toppra.constraint.SecondOrderConstraint`` wired to ``inv_dyn(q, qd,
  qdd) = M @ qdd`` (``M = diag(mass, mass, mass, inertia, inertia,
  inertia)``) plus the exact half-space representation of that achievable
  region (:func:`~sobits_intball2_gnc.guidance.utils.actuation_envelope.
  wrench_envelope_halfspaces`) replaces the old
  ``JointAccelerationConstraint`` box.

- Attitude is expressed as a rotation vector relative to ``q0`` (not an
  independent SO(3) path/TOPP-SO3), which is only an exact stand-in for
  angular velocity/acceleration in the small-rotation-between-waypoints
  regime (that doc's "回転ベクトルを独立joint座標として扱うことの妥当性"
  section). Large single-turn reorientations (up to ~144 deg between
  consecutive waypoints) have since been sim-validated with no overshoot
  or convergence issue (``docs/main_plan.md`` "90°超waypointでの分離型機動",
  ``docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md``
  "その11") -- this approximation is not the limiting factor it was
  originally thought to be.
- Only used for ``trajectory_tracking_mode="static"``: ``toppra``'s
  ``compute_trajectory(sd_start, ...)`` only accepts a *scalar* path-tangent
  start speed, which cannot express a velocity residual perpendicular to the
  path (the case ``ReplanningTrajectoryTracker``'s exact v0-aware bound
  derivation exists specifically to handle) -- replanning keeps using
  :class:`~sobits_intball2_gnc.guidance.trajectory.trajectory.Trajectory` /
  ``HeuristicSegmentTimeAllocator`` unchanged.
- The position path shape (before TOPP-RA re-times it) comes from
  :class:`~sobits_intball2_gnc.guidance.trajectory_generation.
  hermite_spline_trajectory_generator.HermiteSplineTrajectoryGenerator`,
  **not** ``toppra.SplineInterpolator``'s own from-waypoints fit. A
  from-waypoints fit was tried first and rejected: feeding
  ``toppra.SplineInterpolator`` only the sparse waypoints directly produces
  a natural ("not-a-knot") cubic spline that bulges/overshoots near a sharp
  corner (confirmed empirically -- a 90 deg corner produced ~0.2m of
  off-line deviation even with ``bc_type="clamped"``), because it has no
  concept of "face the corner's own tangent direction" the way Hermite's
  interior-waypoint tangent estimate does. Each Hermite segment is instead
  densely resampled (``_SAMPLES_PER_SEGMENT`` points) and *those* dense
  samples are what's handed to ``toppra.SplineInterpolator`` -- with enough
  samples its fit hugs the already-correct Hermite shape rather than
  re-inventing (and bulging) its own path through the sparse waypoints.
- Attitude is *not* pre-generated per waypoint and fit with its own Hermite
  spline (the original design here, see
  ``docs/2026-08-28_attitude_waypoint_premature_rotation_root_cause.md``).
  That waypoint-level scheme picked each interior waypoint's target facing
  as a Catmull-Rom-style average of its incoming/outgoing leg directions,
  then interpolated continuously in attitude-space from the very start of
  travel -- so the vehicle began rotating toward a corner's *average*
  direction long before it actually reached the corner, even while still
  moving in a dead-straight line. Instead, attitude is derived per dense
  position sample directly from that sample's local path tangent (see
  ``_dense_travel_rotvecs`` below): the facing direction only starts
  changing once the position path's own tangent starts changing, i.e. once
  the vehicle is actually in the curve.
"""
import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

_WRENCH_DOF = 6

# Dimensionless: samples are spaced along the path's internal arc-length-ish
# parameter (module docstring's "segment_times"), not real time, so this only
# needs to reject genuinely-degenerate (near-zero) tangent vectors -- not act
# as a physically meaningful low-speed threshold.
_DEGENERATE_TANGENT_THRESHOLD = 1e-9

_SAMPLES_PER_SEGMENT = 20


class TrajectoryInfeasibleError(ValueError):
    """No time-parameterization exists that respects the given velocity/
    acceleration limits for this path (``toppra``'s ``compute_trajectory``
    returned ``None``) -- e.g. the path's curvature demands more
    acceleration than the vehicle's force/torque budget allows at any
    speed. Callers should treat this the same way
    ``SegmentTimeInfeasibleError`` is treated elsewhere in this package: a
    genuine kinematic dead end, not a bug."""


class ToppraTrajectory:
    """Time-parameterized combined position+attitude trajectory.

    Satisfies the same minimal duck-typed interface
    :class:`~sobits_intball2_gnc.guidance.trajectory.trajectory.Trajectory` gives
    :class:`~sobits_intball2_gnc.guidance.trajectory_tracking.
    static_trajectory_tracker.StaticTrajectoryTracker` (``sample(t) ->
    (p, v, a, q)`` and a ``global_total_duration`` property) -- ``Trajectory``
    itself is untouched; this is a separate class for the ``static`` path
    only (see module docstring).

    Args:
        position_waypoints: ``(n, 3)`` reference-frame positions.
        q0: the vehicle's actual departure attitude (shape ``(4,)``) --
            both the rotation-vector coordinate's reference orientation and
            (when ``face_travel``) the fixed starting facing (``pre_align``
            is responsible for making that already face the right way, same
            convention as ``Trajectory``'s ``initial_q_des``).
        max_vel: linear velocity limit [m/s], scalar (applied to all 3
            translational axes).
        mass: vehicle mass [kg] (``trajectory_controller.mass``).
        inertia: vehicle inertia, isotropic [kg*m^2]
            (``trajectory_controller.inertia``).
        wrench_envelope: ``(F, g)`` half-space representation (see
            :func:`~sobits_intball2_gnc.guidance.utils.actuation_envelope.
            wrench_envelope_halfspaces`) of the vehicle's actual achievable
            combined force+torque region, i.e. ``{[F;T] : F @ [F;T] <= g}``
            -- computed once from the real fan geometry/``fj_max``, static
            across ``move_to`` calls (module docstring).
        max_angular_rate: angular velocity limit [rad/s], scalar (applied to
            all 3 rotational axes; same value
            ``guidance.max_angular_rate_deg`` already feeds to
            ``compute_q_des``'s rate limiter for the replanning path).
        forward_axis: body-frame axis (unit vector, shape ``(3,)``) that
            ``face_travel`` points along the position path's local tangent.
            Ignored when ``face_travel`` is ``False``.
        face_travel: when ``True`` (default), the attitude path faces
            ``forward_axis`` along the position path's own local direction
            of travel throughout (see module docstring); when ``False``,
            attitude is held at ``q0`` for the whole path.

    Raises:
        TrajectoryInfeasibleError: if no feasible time-parameterization
            exists for this path under the given limits.
    """

    def __init__(self, position_waypoints, q0,
                 max_vel, mass, inertia, wrench_envelope, max_angular_rate,
                 forward_axis=(1.0, 0.0, 0.0), face_travel=True):
        position_waypoints = np.asarray(position_waypoints, dtype=float)
        self._q0 = np.asarray(q0, dtype=float)

        # Internal-only segment "times" (module docstring): any positive,
        # monotonic-ish spacing works here -- TOPP-RA re-times the whole
        # path from scratch below, this only parameterizes the Hermite fit
        # and the dense-resample grid. Arc length is a natural, simple
        # choice (falls back to unit spacing for a degenerate zero-length
        # segment, e.g. a pure in-place reorientation).
        distances = np.linalg.norm(np.diff(position_waypoints, axis=0), axis=1)
        segment_times = np.where(distances < 1e-9, 1.0, distances)

        pos_coeffs = HermiteSplineTrajectoryGenerator().generate(
            position_waypoints, segment_times
        )

        p_list = []
        v_list = []
        n_segments = len(segment_times)
        for seg in range(n_segments):
            taus = np.linspace(
                0.0, segment_times[seg], _SAMPLES_PER_SEGMENT,
                endpoint=(seg == n_segments - 1),
            )
            for tau in taus:
                p_list.append(evaluate_vector(pos_coeffs[seg], tau, order=0))
                v_list.append(evaluate_vector(pos_coeffs[seg], tau, order=1))
        p_arr = np.array(p_list)
        # True cumulative Euclidean arc length of the dense position samples,
        # not the Hermite tau/waypoint-distance parameter above: that tau
        # only equals real path distance for a perfectly straight segment,
        # so reusing it as `ss` makes toppra's `sd`/`sd_start` (rad or m per
        # unit of `ss`) not equal real m/s (test/experiment_toppra_v0_start_tangent.py).
        seglens = np.linalg.norm(np.diff(p_arr, axis=0), axis=1)
        ss = np.concatenate([[0.0], np.cumsum(seglens)])

        rotvecs = _dense_travel_rotvecs(
            v_list, self._q0, forward_axis, face_travel
        )
        combined = np.concatenate([p_arr, rotvecs], axis=1)

        path = ta.SplineInterpolator(ss, combined)

        vel_max = np.array([max_vel] * 3 + [float(max_angular_rate)] * 3)
        pc_vel = constraint.JointVelocityConstraint(
            np.vstack([-vel_max, vel_max]).T
        )

        # w = M @ qdd (module docstring): no q/qd dependence (no gravity, no
        # Coriolis-like coupling modeled for the rotvec coordinate -- same
        # simplification _dense_travel_rotvecs already makes elsewhere), so
        # inv_dyn ignores its first two arguments.
        inertia_matrix = np.diag([mass, mass, mass, inertia, inertia, inertia])
        wrench_F, wrench_g = wrench_envelope

        def inv_dyn(_q, _qd, qdd):
            return inertia_matrix @ qdd

        pc_wrench = constraint.SecondOrderConstraint(
            inv_dyn, lambda _q: wrench_F, lambda _q: wrench_g, dof=_WRENCH_DOF
        )

        instance = algo.TOPPRA([pc_vel, pc_wrench], path,
                                parametrizer="ParametrizeConstAccel")
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            raise TrajectoryInfeasibleError(
                "no feasible time-parameterization for this path under the "
                "given velocity/acceleration limits"
            )
        self._jnt_traj = jnt_traj

    @property
    def global_total_duration(self):
        return float(self._jnt_traj.duration)

    def sample(self, t):
        """Return ``(p, v, a, q)`` at time ``t`` (clamped to the
        trajectory's duration, holding the final state past it)."""
        t = min(max(float(t), 0.0), self._jnt_traj.duration)
        state = self._jnt_traj(t)
        vel = self._jnt_traj(t, 1)
        acc = self._jnt_traj(t, 2)
        p = state[:3]
        v = vel[:3]
        a = acc[:3]
        q = quat_mul(self._q0, quat_exp(state[3:]))
        return p, v, a, q


def _dense_travel_rotvecs(v_list, q0, forward_axis, face_travel):
    """Return ``(len(v_list), 3)`` rotation vectors (relative to ``q0``) for
    each dense position-path tangent sample in ``v_list``.

    Sample 0 is always ``q0`` unchanged (matches ``Trajectory``'s
    ``initial_q_des`` convention). When ``face_travel``, every later sample
    faces ``forward_axis`` along that sample's own tangent direction
    (:func:`~sobits_intball2_gnc.guidance.utils.attitude_reference.compute_q_des`,
    the same "instantaneous velocity direction" policy the replanning path
    uses -- module docstring explains why this replaced a waypoint-level
    precomputed attitude path), with the free roll DOF carried over
    continuously from the previous sample. When not ``face_travel``, every
    sample stays at ``q0``.
    """
    n = len(v_list)
    q_prev = np.asarray(q0, dtype=float)
    rotvecs = np.zeros((n, 3))
    for i in range(1, n):
        if face_travel:
            q_prev = compute_q_des(
                v_list[i], q_prev, _DEGENERATE_TANGENT_THRESHOLD, forward_axis
            )
        rotvecs[i] = quat_log(quat_mul(quat_conj(np.asarray(q0, dtype=float)), q_prev))
    return rotvecs
