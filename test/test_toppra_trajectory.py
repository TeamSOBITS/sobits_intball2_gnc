"""Unit tests for guidance/utils/toppra_trajectory.py (plain-value, no ROS)."""
import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_rotate
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import IDENTITY_QUAT
from sobits_intball2_gnc.guidance.utils.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
)

FORWARD = (1.0, 0.0, 0.0)
MASS = 3.216
INERTIA = 0.0136
MAX_VEL = 0.5
MAX_ANGULAR_RATE = np.radians(90.0)

# Real fan geometry (same defaults ThrustAllocator/config/gnc_params.yaml
# use) -- static given the geometry, computed once and reused, mirroring how
# guidance.py wires this in production (see toppra_trajectory.py's module
# docstring).
_ALLOC = ThrustAllocator()
WRENCH_ENVELOPE = wrench_envelope_halfspaces(_ALLOC.A, _ALLOC.fj_max)


def _build(waypoints, q0=None, max_vel=MAX_VEL, mass=MASS, inertia=INERTIA,
           wrench_envelope=WRENCH_ENVELOPE, max_angular_rate=MAX_ANGULAR_RATE,
           face_travel=True):
    q0 = IDENTITY_QUAT.copy() if q0 is None else np.asarray(q0, dtype=float)
    return ToppraTrajectory(
        waypoints, q0,
        max_vel=max_vel, mass=mass, inertia=inertia,
        wrench_envelope=wrench_envelope, max_angular_rate=max_angular_rate,
        forward_axis=FORWARD, face_travel=face_travel,
    ), q0


def test_starts_at_p0_and_q0():
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    traj, q0 = _build(waypoints)
    p, v, _a, q = traj.sample(0.0)
    assert np.allclose(p, waypoints[0], atol=1e-6)
    assert np.allclose(v, [0.0, 0.0, 0.0], atol=1e-6)
    assert geodesic_angle(q, q0) < 1e-6


def test_reaches_target_at_total_duration_at_rest():
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    p, v, _a, _q = traj.sample(traj.global_total_duration)
    assert np.allclose(p, waypoints[-1], atol=1e-2)
    assert np.allclose(v, [0.0, 0.0, 0.0], atol=1e-2)


def test_holds_final_state_past_total_duration():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    p_end, v_end, _a, q_end = traj.sample(traj.global_total_duration)
    p_after, v_after, _a2, q_after = traj.sample(traj.global_total_duration + 5.0)
    assert np.allclose(p_end, p_after, atol=1e-9)
    assert np.allclose(v_end, v_after, atol=1e-9)
    assert np.allclose(q_end, q_after, atol=1e-9)


def test_clamps_negative_time_to_start():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    p_neg, _v, _a, _q = traj.sample(-5.0)
    p_zero, _v2, _a2, _q2 = traj.sample(0.0)
    assert np.allclose(p_neg, p_zero)


def test_respects_wrench_envelope():
    # Regression test for docs/
    # 2026-08-28_toppra_static_path_attitude_overshoot_incident.md "追記
    # （2026-08-28 その2）": the old per-axis-independent max_accel/
    # max_angular_accel boxes let TOPP-RA plan combined force+torque demand
    # the real 8-fan allocator can't jointly deliver (a 2-axis combined
    # force request at each axis's own individual max was only 68%
    # achievable). The combined (linear, angular) acceleration at every
    # sampled point, converted to a wrench via M @ qdd, must now satisfy the
    # real allocator's achievable half-space region directly (this is what
    # the SecondOrderConstraint enforces during planning).
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints)
    F, g = WRENCH_ENVELOPE
    inertia_matrix = np.diag([MASS] * 3 + [INERTIA] * 3)
    ts = np.linspace(0.0, traj.global_total_duration, 200)
    for t in ts:
        acc = traj._jnt_traj(min(t, traj.global_total_duration), 2)
        w = inertia_matrix @ acc
        # Small absolute tolerance (1e-4 N / N*m): TOPP-RA's own numerical
        # solve tolerance, not a physically meaningful margin -- several
        # orders of magnitude below the ~50%+ shortfalls the old
        # independent-per-axis-box design let through.
        assert np.all(F @ w <= g + 1e-4)


def test_feedforward_wrench_is_achievable_by_the_real_allocator():
    # Stronger, end-to-end version of test_respects_wrench_envelope: feed
    # the planned feedforward wrench straight into the real
    # ThrustAllocator.allocate()/achieved_wrench() (not just the half-space
    # check) and confirm it comes back essentially unchanged -- this is the
    # exact check that caught the original bug (docs/
    # 2026-08-28_toppra_static_path_attitude_overshoot_incident.md), applied
    # here as a permanent regression guard.
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints)
    inertia_matrix = np.diag([MASS] * 3 + [INERTIA] * 3)
    ts = np.linspace(0.0, traj.global_total_duration, 100)
    for t in ts:
        acc = traj._jnt_traj(min(t, traj.global_total_duration), 2)
        w = inertia_matrix @ acc
        f_ff, t_ff = w[:3], w[3:]
        duties = _ALLOC.allocate(f_ff, t_ff)
        f_ach, t_ach = _ALLOC.achieved_wrench(duties)
        assert np.allclose(f_ff, f_ach, atol=1e-4)
        assert np.allclose(t_ff, t_ach, atol=1e-4)


def test_respects_translational_velocity_limit():
    waypoints = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    ts = np.linspace(0.0, traj.global_total_duration, 200)
    vels = np.array([traj.sample(t)[1] for t in ts])
    speeds = np.linalg.norm(vels, axis=1)
    assert np.all(speeds <= MAX_VEL * 1.05)


def test_returns_unit_quaternions_throughout():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints)
    ts = np.linspace(0.0, traj.global_total_duration, 50)
    for t in ts:
        _p, _v, _a, q = traj.sample(t)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)


def test_faces_direction_of_travel_partway_through_a_straight_leg():
    # q0 already faces the travel direction (+Y) here, matching what
    # pre_align guarantees in the real system (ToppraTrajectory itself
    # doesn't know about pre_align, so the test must set up its
    # precondition) -- this isolates "does it keep facing +Y mid-transit"
    # from "does it rotate away from a mismatched q0 by the end".
    q0 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])  # +X -> +Y, 90 deg about +Z
    waypoints = [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    traj, _q0 = _build(waypoints, q0=q0)
    mid_t = traj.global_total_duration / 2.0
    _p, _v, _a, q = traj.sample(mid_t)
    rotated = quat_rotate(q, np.array(FORWARD))
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=0.05)


def test_does_not_prematurely_rotate_before_a_corner():
    # Regression test for docs/
    # 2026-08-28_attitude_waypoint_premature_rotation_root_cause.md: with the
    # old waypoint-level Catmull-Rom-averaged attitude path, the facing
    # direction started rotating toward the corner's *average* direction
    # from t=0, well ahead of the position path's own tangent actually
    # turning (decoupled from it). The fixed per-sample-tangent design must
    # keep facing tightly locked to the position path's own instantaneous
    # direction of travel throughout -- including through the corner's
    # gradual Hermite-smoothed curvature, not just the dead-straight
    # middle -- rather than a separately-interpolated, decoupled target.
    q0 = IDENTITY_QUAT.copy()  # already faces +X, matching leg 1's direction
    waypoints = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0]]
    traj, _q0 = _build(waypoints, q0=q0)
    ts = np.linspace(0.0, traj.global_total_duration, 30)
    for t in ts:
        _p, v, _a, q = traj.sample(t)
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            continue
        v_dir = v / speed
        facing = quat_rotate(q, np.array(FORWARD))
        # atol, not an exact match: toppra.SplineInterpolator refits a cubic
        # spline through the dense (position, rotvec) samples independently
        # per joint, so its resampled v/q at a given t deviate slightly from
        # the exact per-sample values this class computed them from.
        assert np.allclose(facing, v_dir, atol=0.02)


def test_raises_when_toppra_reports_infeasible(monkeypatch):
    # Exercise the wrapper's error-handling path directly (toppra.compute_
    # trajectory() returning None) rather than hunting for real input values
    # that trigger it -- TOPP-RA can generally always find *some* feasible
    # (if very slow) time-parameterization for a well-posed path, so genuine
    # infeasibility is not a reliable thing to provoke with plain inputs.
    import sobits_intball2_gnc.guidance.utils.toppra_trajectory as mod

    monkeypatch.setattr(
        mod.algo.TOPPRA, "compute_trajectory", lambda self, *a, **kw: None
    )
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    q0 = IDENTITY_QUAT.copy()
    with pytest.raises(TrajectoryInfeasibleError):
        ToppraTrajectory(
            waypoints, q0,
            max_vel=MAX_VEL, mass=MASS, inertia=INERTIA,
            wrench_envelope=WRENCH_ENVELOPE, max_angular_rate=MAX_ANGULAR_RATE,
            forward_axis=FORWARD,
        )


def test_three_waypoint_path_reaches_final_target():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints)
    p, v, _a, _q = traj.sample(traj.global_total_duration)
    assert np.allclose(p, waypoints[-1], atol=1e-2)
    assert np.allclose(v, [0.0, 0.0, 0.0], atol=1e-2)


def test_face_travel_false_holds_q0_throughout():
    q0 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])  # 90 deg about +Z
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints, q0=q0, face_travel=False)
    ts = np.linspace(0.0, traj.global_total_duration, 20)
    for t in ts:
        _p, _v, _a, q = traj.sample(t)
        assert geodesic_angle(q, q0) < 1e-6


def test_non_identity_q0_is_respected():
    q0 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])  # 90 deg about +Z
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints, q0=q0)
    p, _v, _a, q = traj.sample(0.0)
    assert np.allclose(p, waypoints[0], atol=1e-6)
    assert geodesic_angle(q, q0) < 1e-6
