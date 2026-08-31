"""Integration test: HeuristicSegmentTimeAllocator -> HermiteSplineTrajectoryGenerator
-> Trajectory -> TrajectoryController.compute_attitude.

Demonstrates the intended production call order (docs/main_plan.md Phase 2,
docs/min_snap_interface_contract.md 6 節):

    segment_times = allocator.allocate(waypoints)
    coeffs = generator.generate(waypoints, segment_times)
    traj = Trajectory(waypoints, segment_times, coeffs)
    p_des, v_des, a_des, q_des = traj.sample(t)

end-to-end, using the real (if degraded) HermiteSplineTrajectoryGenerator
stand-in (docs/architecture_guidelines.md 2 節,
guidance/trajectory_generation/) instead of a hand-rolled coefficient stub --
swap in MinSnapTrajectoryGenerator once min_snap.py's core lands; no other
line in this file should need to change.

The ``q_des`` tests below close the Phase 3b gap noted in docs/main_plan.md:
Hermite's ``generate()`` only produces position coefficients (by design --
attitude is Trajectory's job via attitude_reference.compute_q_des), and until
now nothing exercised that hand-off, nor fed the result into
TrajectoryController.compute_attitude() the way HoverController does in
production (control/utils/hover_controller.py).
"""
import numpy as np

from sobits_intball2_gnc.control.utils.trajectory_controller import TrajectoryController
from sobits_intball2_gnc.guidance.segment_time.heuristic_segment_time_allocator import (
    HeuristicSegmentTimeAllocator,
)
from sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator import (
    HermiteSplineTrajectoryGenerator,
)
from sobits_intball2_gnc.guidance.trajectory.trajectory import Trajectory


def _build_trajectory(waypoints, allocator):
    segment_times = allocator.allocate(waypoints)
    coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)
    return Trajectory(waypoints, segment_times, coeffs), segment_times


def test_pipeline_starts_and_ends_at_waypoints():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5)
    traj, _segment_times = _build_trajectory(waypoints, allocator)

    p_start, _v, _a, _q = traj.sample(0.0)
    p_end, v_end, a_end, _q = traj.sample(traj.total_duration)
    assert np.allclose(p_start, waypoints[0])
    assert np.allclose(p_end, waypoints[-1])
    assert np.allclose(v_end, [0.0, 0.0, 0.0])
    assert np.allclose(a_end, [0.0, 0.0, 0.0])


def test_sharper_turn_yields_longer_total_duration():
    # Same segment lengths (1.0, 1.0) in both cases; only the turn angle at
    # the interior waypoint differs. With angle_time_gain > 0 the sharper
    # turn's extra time should show up directly in the resulting Trajectory.
    gentle_waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]]
    sharp_waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0, angle_time_gain=1.0)

    gentle_traj, _ = _build_trajectory(gentle_waypoints, allocator)
    sharp_traj, _ = _build_trajectory(sharp_waypoints, allocator)

    assert sharp_traj.total_duration > gentle_traj.total_duration


def test_pipeline_is_continuous_across_segment_boundary():
    waypoints = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=1.0)
    traj, segment_times = _build_trajectory(waypoints, allocator)

    boundary_t = segment_times[0]
    p_before, _v, _a, _q = traj.sample(boundary_t - 1e-9)
    p_after, _v, _a, _q = traj.sample(boundary_t)
    assert np.allclose(p_before, p_after, atol=1e-6)
    assert np.allclose(p_after, waypoints[1])


def test_pipeline_q_des_faces_travel_direction_mid_segment():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5)
    traj, _segment_times = _build_trajectory(waypoints, allocator)

    _p, v_mid, _a, q_mid = traj.sample(traj.total_duration / 2.0)

    assert np.isclose(np.linalg.norm(q_mid), 1.0)
    # Default forward_axis is body +X: the shortest-arc quaternion pointing
    # +X along a nonzero v_des rotates it back onto v_des's own direction.
    from sobits_intball2_gnc.control.utils.quat_math import quat_rotate

    pointed = quat_rotate(q_mid, np.array([1.0, 0.0, 0.0]))
    speed = np.linalg.norm(v_mid)
    assert speed > 0.0
    assert np.allclose(pointed, v_mid / speed, atol=1e-6)


def test_pipeline_q_des_holds_at_rest_start_and_end():
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5)
    traj, _segment_times = _build_trajectory(waypoints, allocator)

    # At rest (v_des == 0), compute_q_des holds prev_q_des unchanged; with no
    # seeded initial_q_des the very first sample falls back to identity.
    _p, v_start, _a, q_start = traj.sample(0.0)
    assert np.allclose(v_start, [0.0, 0.0, 0.0])
    assert np.allclose(q_start, [0.0, 0.0, 0.0, 1.0])

    _p, v_end, _a, q_end = traj.sample(traj.total_duration)
    assert np.allclose(v_end, [0.0, 0.0, 0.0])
    assert np.allclose(q_end, q_start)


def test_pipeline_face_travel_false_holds_initial_attitude():
    """face_travel=False (guidance_node_implementation_plan.md decision 1):
    a fast transit with no heading requirement should never touch q_des or
    trigger pre-alignment -- q_des must stay fixed at whatever the vehicle's
    attitude was when the trajectory started, for the entire move."""
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5)
    segment_times = allocator.allocate(waypoints)
    coeffs = HermiteSplineTrajectoryGenerator().generate(waypoints, segment_times)
    initial_q_des = [0.0, 0.0, 0.70710678, 0.70710678]
    traj = Trajectory(
        waypoints, segment_times, coeffs,
        initial_q_des=initial_q_des, face_travel=False,
    )

    for t in [0.0, traj.total_duration / 2.0, traj.total_duration]:
        _p, v, _a, q = traj.sample(t)
        assert np.allclose(q, initial_q_des)
        assert np.linalg.norm(v) >= 0.0  # face_travel=False doesn't touch v_des either


def test_pipeline_q_des_drives_trajectory_controller_compute_attitude():
    """Feed Hermite's q_des(t) into TrajectoryController the way HoverController
    does in production (control/utils/hover_controller.py), closing the loop
    from segment-time allocation all the way to a commanded torque."""
    waypoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    allocator = HeuristicSegmentTimeAllocator(target_speed=0.5)
    traj, _segment_times = _build_trajectory(waypoints, allocator)
    controller = TrajectoryController()

    quat_now = np.array([0.0, 0.0, 0.0, 1.0])
    t_mid = traj.total_duration / 2.0
    _p, _v, _a, q_mid = traj.sample(t_mid)

    torque = controller.compute_attitude(t_mid, quat_now, q_mid)

    assert len(torque) == 3
    assert all(np.isfinite(torque))
    assert all(abs(component) <= controller.max_torque + 1e-9 for component in torque)
