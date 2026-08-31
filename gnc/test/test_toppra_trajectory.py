"""Unit tests for guidance/utils/toppra_trajectory.py (plain-value, no ROS)."""
import numpy as np
import pytest
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_rotate
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.utils.actuation_envelope import (
    wrench_envelope_halfspaces,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    IDENTITY_QUAT,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.trajectory.toppra_trajectory import (
    ToppraTrajectory,
    TrajectoryInfeasibleError,
    _SAMPLES_PER_SEGMENT,
    _WRENCH_DOF,
    _dense_travel_rotvecs,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

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
    import sobits_intball2_gnc.guidance.trajectory.toppra_trajectory as mod

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


def test_retime_reproduces_construction_with_sd_start_zero():
    # retime(sd_start=0.0, sd_end=0.0) must reproduce exactly what __init__
    # already computed (v0=0 fixed, same fallback __init__ uses) -- this is
    # the baseline sanity check before trusting retime() with nonzero
    # sd_start below.
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    traj, _q0 = _build(waypoints)
    original_duration = traj.global_total_duration
    traj.retime(sd_start=0.0, sd_end=0.0)
    assert traj.global_total_duration == pytest.approx(original_duration, abs=1e-9)


def test_retime_with_nonzero_sd_start_changes_initial_velocity():
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    sd_start = MAX_VEL * 0.5
    traj.retime(sd_start=sd_start, sd_end=0.0)
    _p, v, _a, _q = traj.sample(0.0)
    assert np.isclose(np.linalg.norm(v), sd_start, atol=1e-6)


def test_retime_still_reaches_target_at_rest():
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    traj.retime(sd_start=MAX_VEL * 0.3, sd_end=0.0)
    p, v, _a, _q = traj.sample(traj.global_total_duration)
    assert np.allclose(p, waypoints[-1], atol=1e-2)
    assert np.allclose(v, [0.0, 0.0, 0.0], atol=1e-2)


def test_retime_raises_when_toppra_reports_infeasible(monkeypatch):
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    traj, _q0 = _build(waypoints)
    monkeypatch.setattr(
        traj._instance, "compute_trajectory", lambda *a, **kw: None
    )
    with pytest.raises(TrajectoryInfeasibleError):
        traj.retime(sd_start=0.1, sd_end=0.0)


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


def _dense_rotvecs_for(waypoints, q0):
    waypoints = np.asarray(waypoints, dtype=float)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.where(distances < 1e-9, 1.0, distances)
    pos_coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)
    v_list = []
    n_segments = len(segment_times)
    for seg in range(n_segments):
        taus = np.linspace(
            0.0, segment_times[seg], _SAMPLES_PER_SEGMENT,
            endpoint=(seg == n_segments - 1),
        )
        for tau in taus:
            v_list.append(evaluate_vector(pos_coeffs[seg], tau, order=1))
    return _dense_travel_rotvecs(v_list, q0, FORWARD, True)


def test_dense_travel_rotvecs_stays_continuous_past_a_180_degree_crossing():
    """Regression test for docs/
    2026-08-31_multi_via_waypoints_static_test_near_dock_anomaly.md: a
    multi-via-waypoint route (real coordinates from that incident,
    maps/iss_location.yaml's above_dock_2/nav_entry/near_dock) whose
    cumulative face-travel rotation relative to q0 passes 180 degrees used
    to produce exactly one dense sample with a flipped rotvec axis (
    quat_log's [0, pi]-clamped output snapping back once the true rotation
    passed the halfway point), corrupting the spline TOPP-RA fits attitude
    to and causing a real sawtooth attitude-tracking failure in sim."""
    waypoints = np.array([
        [10.0607, -3.5544, 4.9114],
        [11.3, -3.636, 5.5],
        [11.0, -4.3, 5.0],
        [10.936, -3.636, 4.121],
    ])
    q0 = compute_q_des(waypoints[1] - waypoints[0], None, 1e-9, FORWARD)
    rotvecs = _dense_rotvecs_for(waypoints, q0)
    angles = np.linalg.norm(rotvecs, axis=1)
    assert angles.max() > np.pi  # sanity: this route does cross the 180deg boundary

    for i in range(1, len(rotvecs)):
        a, b = rotvecs[i - 1], rotvecs[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 0.1 or nb < 0.1:
            continue  # near-zero rotation: axis is numerically undefined, not a real discontinuity
        cos_sim = np.dot(a, b) / (na * nb)
        assert cos_sim > 0.0, (
            f"rotvec axis flipped between dense samples {i - 1} and {i} "
            f"(angles {np.degrees(na):.1f}/{np.degrees(nb):.1f} deg, "
            f"cos_sim={cos_sim:.3f})"
        )


def _build_path(waypoints, q0):
    """Same dense-resample + rotvec construction ``ToppraTrajectory.__init__``
    uses, extracted so the baseline test below can swap only the constraint."""
    waypoints = np.asarray(waypoints, dtype=float)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.where(distances < 1e-9, 1.0, distances)
    pos_coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)

    p_list, v_list = [], []
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
    seglens = np.linalg.norm(np.diff(p_arr, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seglens)])
    rotvecs = _dense_travel_rotvecs(v_list, q0, FORWARD, True)
    combined = np.concatenate([p_arr, rotvecs], axis=1)
    return ta.SplineInterpolator(ss, combined)


def _compute_trajectory_with_second_order_constraint(waypoints, q0):
    """Pre-``WrenchEnvelopeConstraint`` baseline: same path, but the wrench
    constraint built with plain ``toppra.constraint.SecondOrderConstraint``
    (per-gridpoint ``constraint_F``/``constraint_g`` lambdas, no
    ``identical=True``) -- the code ``ToppraTrajectory`` used before this
    change. Used only to verify the new ``WrenchEnvelopeConstraint`` path is
    behavior-identical, not a speed comparison (see
    ``test/experiment_toppra_identical_constraint.py`` for that)."""
    path = _build_path(waypoints, q0)
    vel_max = np.array([MAX_VEL] * 3 + [float(MAX_ANGULAR_RATE)] * 3)
    pc_vel = constraint.JointVelocityConstraint(np.vstack([-vel_max, vel_max]).T)
    inertia_matrix = np.diag([MASS, MASS, MASS, INERTIA, INERTIA, INERTIA])
    wrench_F, wrench_g = WRENCH_ENVELOPE

    def inv_dyn(_q, _qd, qdd):
        return inertia_matrix @ qdd

    pc_wrench = constraint.SecondOrderConstraint(
        inv_dyn, lambda _q: wrench_F, lambda _q: wrench_g, dof=_WRENCH_DOF
    )
    instance = algo.TOPPRA([pc_vel, pc_wrench], path, parametrizer="ParametrizeConstAccel")
    return instance.compute_trajectory()


def test_wrench_envelope_constraint_matches_second_order_constraint_baseline():
    """Regression test for the identical=True speedup (docs/archive/achieved/
    2026-08-30_toppra_replanning_sd_start_speed_investigation.md 追記2〜3):
    ToppraTrajectory now builds its wrench constraint with
    WrenchEnvelopeConstraint instead of plain SecondOrderConstraint. Confirms
    this is a pure speedup with no behavior change -- same trajectory
    duration and same sampled states as the old constraint construction,
    across a straight and a sharp-turn route."""
    for waypoints in (
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ):
        q0 = IDENTITY_QUAT.copy()
        traj, _q0 = _build(waypoints, q0=q0)
        baseline_jnt_traj = _compute_trajectory_with_second_order_constraint(
            waypoints, q0
        )
        assert baseline_jnt_traj is not None
        assert traj.global_total_duration == pytest.approx(
            baseline_jnt_traj.duration, abs=1e-9
        )
        for t in np.linspace(0.0, traj.global_total_duration, 5):
            p, v, a, q = traj.sample(t)
            base_state = baseline_jnt_traj(t)
            base_vel = baseline_jnt_traj(t, 1)
            base_acc = baseline_jnt_traj(t, 2)
            assert np.allclose(p, base_state[:3], atol=1e-9)
            assert np.allclose(v, base_vel[:3], atol=1e-9)
            assert np.allclose(a, base_acc[:3], atol=1e-9)
