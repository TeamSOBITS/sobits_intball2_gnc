"""Unit tests for GuidanceExecutor (ROS-agnostic, no rclpy)."""
import numpy as np

from sobits_intball2_gnc.guidance.utils.guidance_executor import (
    STATUS_ABORTED,
    STATUS_CANCELED,
    STATUS_SUCCESS,
    GuidanceExecutor,
)


class FakeTf:
    def __init__(self, pos, quat):
        self.pos = pos
        self.quat = quat

    def get_pose(self):
        if self.pos is None:
            return None
        return list(self.pos), list(self.quat), 0.0


class FakeSetpointPublisher:
    def __init__(self):
        self.calls = []

    def publish(self, p, v, a, q):
        self.calls.append((list(p), list(v), list(a), list(q)))


class FakeCheckpointPublisher:
    def __init__(self):
        self.published = []

    def publish(self, pos, quat):
        self.published.append((list(pos), list(quat)))

    def wait_for_subscriber(self, timeout_sec=5.0, spin_fn=None):
        return True


class FakeLogger:
    def info(self, msg):
        pass

    def warn(self, msg):
        pass


def _make_clock(dt_per_spin=0.05):
    state = {"t": 0.0}

    def clock_seconds_fn():
        return state["t"]

    def spin_fn(_seconds):
        state["t"] += dt_per_spin

    return clock_seconds_fn, spin_fn


def test_execute_returns_aborted_when_no_tf_pose():
    executor = GuidanceExecutor(
        FakeTf(None, None), FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(), FakeLogger(),
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
    )
    assert status == STATUS_ABORTED


def test_execute_face_travel_false_never_touches_checkpoints():
    """docs/guidance_node_implementation_plan.md decision 1: no heading
    requirement -> no pre-alignment, straight to the translation loop."""
    setpoint_pub = FakeSetpointPublisher()
    checkpoint_pub = FakeCheckpointPublisher()
    executor = GuidanceExecutor(
        FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        setpoint_pub, checkpoint_pub, *_make_clock(), FakeLogger(),
        target_speed=1.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert checkpoint_pub.published == []
    # q_des must stay fixed at the starting attitude throughout.
    for _p, _v, _a, q in setpoint_pub.calls:
        assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])


def test_execute_face_travel_true_publishes_setpoints_and_reaches_target():
    setpoint_pub = FakeSetpointPublisher()
    executor = GuidanceExecutor(
        FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.2),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=180.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert len(setpoint_pub.calls) > 0
    final_p, final_v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(final_v, [0.0, 0.0, 0.0], atol=1e-6)


def test_execute_cancels_mid_trajectory():
    calls = {"n": 0}

    def is_cancel_requested():
        calls["n"] += 1
        return calls["n"] > 2

    executor = GuidanceExecutor(
        FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.01), FakeLogger(),
        target_speed=0.01,  # slow -> long trajectory, won't finish in a few ticks
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=is_cancel_requested,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_CANCELED


def test_execute_pre_aligns_when_facing_travel_and_misaligned():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from +X-facing

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False,
    )
    # TF never reports convergence (fake doesn't move), so this times out but
    # still proceeds -- the point being tested is that a checkpoint WAS
    # published for the initial-tangent alignment.
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
