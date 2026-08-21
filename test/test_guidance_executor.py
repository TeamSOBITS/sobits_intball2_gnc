"""Unit tests for GuidanceExecutor (ROS-agnostic, no rclpy)."""
import numpy as np

from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    compute_camera_relative_quat,
)
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
    def __init__(self):
        self.warnings = []

    def info(self, msg):
        pass

    def warn(self, msg):
        self.warnings.append(msg)


class ScriptedTf:
    """Returns a scripted sequence of quats, one per get_pose() call (the
    last entry repeats once the sequence is exhausted) -- for exercising
    _align_to's convergence-over-time behavior against a TF that changes
    between polls, unlike FakeTf's fixed pose."""

    def __init__(self, pos, quat_sequence):
        self.pos = pos
        self._seq = list(quat_sequence)
        self._i = 0

    def get_pose(self):
        if self.pos is None:
            return None
        quat = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return list(self.pos), list(quat), 0.0


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


def test_execute_pre_aligns_even_when_target_speed_is_slow():
    """Regression test for
    docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md:
    a slow target_speed keeps the trajectory's early-segment velocity below
    attitude_speed_threshold throughout, which previously made compute_q_des
    silently hold q0 and skip pre_align even for a ~180 deg misalignment."""
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from +X-facing

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=0.01, attitude_speed_threshold=0.02,
        align_tolerance_deg=3.0, align_timeout=0.2,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1


def test_execute_skips_pre_align_when_pre_align_false():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from +X-facing

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False, pre_align=False,
    )
    assert status == STATUS_SUCCESS
    assert checkpoint_pub.published == []


def test_align_to_ignores_a_transient_pass_through_tolerance():
    """Regression test for docs/archive/achieved/
    2026-08-21_pre_align_skipped_low_speed_bug.md's second bug: a single
    in-tolerance TF sample mid-oscillation must not be mistaken for having
    settled. q_target is briefly matched once, then drifts back out and
    stays out -- align_settle_time (0.5s) is never satisfied, so this must
    run to align_timeout rather than returning success early."""
    q_target = [0.0, 0.0, 0.0, 1.0]
    q_far = [0.0, 0.0, 1.0, 0.0]  # 180 deg off
    # index 0 is consumed by execute()'s own initial (p0, q0) TF fetch, index
    # 1 by align_at_arrival's pre-check (still far -> triggers _align_to),
    # then the brief pass-through happens inside _align_to's own loop.
    tf = ScriptedTf([0.0, 0.0, 0.0], [q_far, q_far, q_target, q_far])
    logger = FakeLogger()

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.1), logger,
        target_speed=1.0, align_tolerance_deg=3.0, align_timeout=1.0,
        align_settle_time=0.5,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_SUCCESS
    assert any("did not converge" in w for w in logger.warnings)


def test_align_to_converges_once_settle_time_elapses():
    """Counterpart to the transient-pass-through test: once q_target is
    reached and STAYS reached for align_settle_time, _align_to must return
    promptly rather than waiting out the full align_timeout."""
    q_target = [0.0, 0.0, 0.0, 1.0]
    q_far = [0.0, 0.0, 1.0, 0.0]  # 180 deg off
    # index 0: execute()'s initial (p0, q0) fetch. index 1: align_at_arrival's
    # pre-check (still far -> triggers _align_to). From index 2 on, in
    # tolerance and staying there.
    tf = ScriptedTf([0.0, 0.0, 0.0], [q_far, q_far] + [q_target] * 20)
    logger = FakeLogger()

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.1), logger,
        target_speed=1.0, align_tolerance_deg=3.0, align_timeout=5.0,
        align_settle_time=0.5,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_SUCCESS
    assert not any("did not converge" in w for w in logger.warnings)


def test_execute_align_at_arrival_camera_main_uses_target_orientation_as_is():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from q_target

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True, align_at_arrival_camera="main",
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
    _pos, published_quat = checkpoint_pub.published[0]
    assert np.allclose(published_quat, [0.0, 0.0, 0.0, 1.0])


def test_execute_align_at_arrival_camera_stereo_offsets_from_target_orientation():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # far from either candidate

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
    )
    q_target = [0.0, 0.0, 0.0, 1.0]
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True, align_at_arrival_camera="stereo",
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
    expected_quat = compute_camera_relative_quat(
        q_target, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    )
    _pos, published_quat = checkpoint_pub.published[0]
    assert np.allclose(published_quat, expected_quat, atol=1e-9)
    assert not np.allclose(published_quat, q_target, atol=1e-6)


def test_execute_align_at_arrival_unknown_camera_falls_back_to_target_orientation():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from q_target

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True, align_at_arrival_camera="wide",
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
    _pos, published_quat = checkpoint_pub.published[0]
    assert np.allclose(published_quat, [0.0, 0.0, 0.0, 1.0])  # fell back to q_target
