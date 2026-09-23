"""Unit tests for GuidanceExecutor (ROS-agnostic, no rclpy)."""
import numpy as np
import pytest

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle
from sobits_intball2_gnc.guidance.utils.attitude_reference import (
    compute_camera_relative_quat,
    compute_q_des,
)
from sobits_intball2_gnc.guidance.utils.guidance_executor import (
    STATUS_ABORTED,
    STATUS_CANCELED,
    STATUS_SUCCESS,
    GuidanceExecutor,
)


class FakeTf:
    """``stamp=None`` (default) auto-advances by 1.0 per call, so this fake
    reads as a live TF stream for staleness purposes unless a test
    explicitly wants a frozen stamp -- pass a fixed ``stamp`` to simulate a
    stalled TF publisher (get_pose() keeps succeeding from tf2's buffer, but
    the stamp stops advancing)."""

    def __init__(self, pos, quat, stamp=None):
        self.pos = pos
        self.quat = quat
        self._stamp = stamp
        self._state = {"n": 0}

    def get_pose(self):
        if self.pos is None:
            return None
        return list(self.pos), list(self.quat), _next_stamp(self._state, self._stamp)


class SetpointFollowingTf:
    """Ideal-tracking TF fake: reports the last commanded setpoint position
    (or ``pos`` before any command). Needed for trackers that replan from
    their own reference state rather than from measured pose."""

    def __init__(self, pos, quat, setpoint_pub, offset=(0.0, 0.0, 0.0)):
        self._initial = list(pos)
        self.quat = quat
        self._setpoint_pub = setpoint_pub
        self._offset = np.asarray(offset, dtype=float)
        self._state = {"n": 0}

    def get_pose(self):
        calls = self._setpoint_pub.calls
        pos = np.asarray(calls[-1][0] if calls else self._initial, dtype=float)
        return (pos + self._offset).tolist(), list(self.quat), _next_stamp(self._state, None)


class FakeVelocityEstimate:
    def __init__(self, vel):
        self.vel = vel


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


class FakeSpeedPathPublisher:
    """Records each ``publish()`` call's sample count, for asserting how
    many times (and roughly when) GuidanceExecutor re-publishes the RViz
    speed-path preview -- once at goal start, and again on every re-plan
    ([G] "再計画軌道のRVizプレビュー更新" task)."""

    def __init__(self):
        self.calls = []

    def publish(self, samples):
        self.calls.append(list(samples))


def _next_stamp(state, stamp_spec):
    """Shared stamp-generation logic for the Scripted*Tf fakes below.

    ``stamp_spec`` is None (auto-advance by 1.0 per call -- always reads as
    a live TF), a single float (frozen for every call -- simulates a TF
    publisher that died before this fake was ever queried), or a list of
    floats (advances through the list, then freezes on its last entry once
    exhausted -- simulates a publisher that was alive for a while and then
    died, the realistic staleness scenario).
    """
    state["n"] += 1
    if stamp_spec is None:
        return float(state["n"])
    if isinstance(stamp_spec, (list, tuple)):
        return float(stamp_spec[min(state["n"] - 1, len(stamp_spec) - 1)])
    return float(stamp_spec)


class ScriptedTf:
    """Returns a scripted sequence of quats, one per get_pose() call (the
    last entry repeats once the sequence is exhausted) -- for exercising
    _align_to's convergence-over-time behavior against a TF that changes
    between polls, unlike FakeTf's fixed pose. ``stamp=None`` (default)
    auto-advances per call like FakeTf; see :func:`_next_stamp` for the
    other forms (a fixed value, or a list that advances then freezes) used
    to simulate a stalled TF publisher."""

    def __init__(self, pos, quat_sequence, stamp=None):
        self.pos = pos
        self._seq = list(quat_sequence)
        self._i = 0
        self._stamp = stamp
        self._state = {"n": 0}

    def get_pose(self):
        if self.pos is None:
            return None
        quat = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return list(self.pos), list(quat), _next_stamp(self._state, self._stamp)


class ScriptedPosTf:
    """Returns a scripted sequence of positions, one per get_pose() call (the
    last entry repeats once the sequence is exhausted) -- for exercising
    _run_trajectory's post-duration position-convergence poll against a TF
    that changes between polls, unlike FakeTf's fixed pose. Mirrors
    ScriptedTf but scripts position instead of attitude. ``stamp=None``
    (default) auto-advances per call like FakeTf; see :func:`_next_stamp`
    for the other forms used to simulate a stalled TF publisher."""

    def __init__(self, pos_sequence, quat, stamp=None):
        self._seq = list(pos_sequence)
        self._i = 0
        self.quat = quat
        self._stamp = stamp
        self._state = {"n": 0}

    def get_pose(self):
        pos = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return list(pos), list(self.quat), _next_stamp(self._state, self._stamp)


class ScriptedPosOrNoneTf:
    """Like ScriptedPosTf, but entries may be ``None`` to script a transient
    TF outage (get_pose() returning None) -- for exercising
    _run_trajectory's post-duration position-convergence poll against a TF
    that goes briefly unavailable mid-wait, a known recurring failure mode
    in this stack (bridge stalls/bursts, see docs/archive/achieved/
    recording_cpu_load_control_degradation.md). ``stamp=None`` (default)
    auto-advances per non-None call like FakeTf; pass a fixed ``stamp`` to
    simulate a stalled TF publisher."""

    def __init__(self, pos_or_none_sequence, quat, stamp=None):
        self._seq = list(pos_or_none_sequence)
        self._i = 0
        self.quat = quat
        self._stamp = stamp
        self._n = 0

    def get_pose(self):
        entry = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        if entry is None:
            return None
        self._n += 1
        stamp = self._n if self._stamp is None else self._stamp
        return list(entry), list(self.quat), float(stamp)


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
        # FakeTf's pose is fixed, so it never enters align_pos_tolerance_m;
        # kept short so this unrelated test doesn't pay the default 10s
        # position-convergence timeout in loop iterations.
        align_pos_timeout=0.1,
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
    logger = FakeLogger()
    # index 0: execute()'s initial (p0, q0) fetch. Indices 1+: _run_trajectory's
    # post-duration position-convergence poll -- starts one tick short of
    # align_pos_tolerance_m, then settles at p_target and stays there, to
    # exercise the settle-time dwell (not just an instant match).
    tf = ScriptedPosTf(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0, 1.0],
    )
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.2),
        logger, target_speed=1.0, align_tolerance_deg=180.0,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.3,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert not any("did not converge" in w for w in logger.warnings)
    assert len(setpoint_pub.calls) > 0
    final_p, final_v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(final_v, [0.0, 0.0, 0.0], atol=1e-6)


def test_run_trajectory_times_out_and_proceeds_when_position_never_converges():
    """Position-error counterpart of _align_to's own timeout fallback: if TF
    position never enters align_pos_tolerance_m after the planned duration
    elapses, _run_trajectory must not hang forever -- it gives up after
    align_pos_timeout and proceeds with a warning."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # never moves toward p_target
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.2),
        logger, target_speed=1.0, align_pos_tolerance_m=0.05,
        align_pos_settle_time=0.5, align_pos_timeout=0.5,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert any("did not converge" in w for w in logger.warnings)


def test_run_trajectory_survives_transient_tf_outage_without_losing_dwell_progress():
    """A brief TF outage (get_pose() returning None) mid-dwell must not
    reset the settle-time counter -- otherwise a transient bridge stall
    (a real, recurring failure mode in this stack, see docs/archive/
    achieved/recording_cpu_load_control_degradation.md) would repeatedly
    knock out an almost-converged wait and never let it complete quickly.

    Sequence (consumed one entry per get_pose() call): index 0 is execute()'s
    initial (p0, q0) fetch. Index 1: first post-duration poll, already at
    p_target -- dwell starts. Indices 2-3: TF outage (None) while still
    within align_pos_settle_time of the outage. Index 4: TF back, still at
    p_target -- if the outage preserved the dwell start time, elapsed dwell
    is now 3 ticks * 0.2s = 0.6s >= align_pos_settle_time (0.5s), so this
    must succeed on this very tick. A cancellation fires on the 6th
    is_cancel_requested() call (one past what the correct behavior needs) to
    turn "took longer than expected" into a hard test failure instead of a
    silently-slow pass.
    """
    tf = ScriptedPosOrNoneTf(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], None, None, [1.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0, 1.0],
    )
    logger = FakeLogger()
    calls = {"n": 0}

    def is_cancel_requested():
        calls["n"] += 1
        return calls["n"] >= 6

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        # target_speed huge -> total_duration collapses to
        # DEFAULT_MIN_SEGMENT_TIME (1e-3s), so the very first spin (0.2s)
        # already exceeds it -- the translation phase takes exactly one
        # iteration, keeping this test's is_cancel_requested call-count
        # math (see docstring) exact.
        *_make_clock(dt_per_spin=0.2), logger,
        target_speed=1000.0, align_pos_tolerance_m=0.05,
        align_pos_settle_time=0.5, align_pos_timeout=5.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=is_cancel_requested,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert not any("did not converge" in w for w in logger.warnings)


def test_run_trajectory_ignores_a_transient_position_pass_through():
    """Position-error counterpart of test_align_to_ignores_a_transient_pass_
    through_tolerance: a single in-tolerance TF sample (e.g. an overshoot
    swinging through p_target) must not be mistaken for having settled. The
    position briefly enters align_pos_tolerance_m once, then drifts back out
    and stays out -- align_pos_settle_time is never satisfied, so this must
    run to align_pos_timeout rather than returning success early. (Live-sim
    verification attempted first: a real move_to with a 5mm tolerance
    converged monotonically without ever re-crossing the boundary, so this
    exact failure mode wasn't observed on real dynamics -- this test
    exercises the dwell-reset branch directly instead.)
    """
    tf = ScriptedPosTf(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        [0.0, 0.0, 0.0, 1.0],
    )
    logger = FakeLogger()
    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.2), logger,
        target_speed=1000.0, align_pos_tolerance_m=0.05,
        align_pos_settle_time=0.5, align_pos_timeout=0.5,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert any("did not converge" in w for w in logger.warnings)


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
        align_pos_timeout=0.1,
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
        align_tolerance_deg=3.0, align_timeout=0.2, align_pos_timeout=0.1,
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
        align_pos_timeout=0.1,
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
        align_settle_time=0.5, align_pos_timeout=0.1,
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
        # This test's TF fixture never moves in position, so make the
        # position-convergence check trivially pass (this test is about
        # attitude convergence, not translation) -- keeps the "no timeout
        # warning at all" assertion below meaningful for the attitude stage.
        align_pos_tolerance_m=2.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_SUCCESS
    assert not any("did not converge" in w for w in logger.warnings)


def test_align_to_ramps_via_slerp_when_configured():
    """docs/2026-08-27_align_slerp_trapezoid_next_steps.md: with both
    align_angular_speed_deg/align_angular_accel_deg set, _align_to must
    publish a sequence of moving SLERP checkpoints (not just the final one),
    monotonically closing the angle to hold_quat, ending exactly at it."""
    q_from = [0.0, 0.0, 1.0, 0.0]  # 180 deg off
    q_target = [0.0, 0.0, 0.0, 1.0]
    tf = FakeTf([0.0, 0.0, 0.0], q_from)
    checkpoint_pub = FakeCheckpointPublisher()

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=1.0),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
        align_angular_speed_deg=15.0, align_angular_accel_deg=2.4,
        align_traj_publish_rate_hz=20.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) > 1

    angles_to_target = [
        geodesic_angle(quat, q_target) for _pos, quat in checkpoint_pub.published
    ]
    assert all(b <= a + 1e-9 for a, b in zip(angles_to_target, angles_to_target[1:]))
    _last_pos, last_quat = checkpoint_pub.published[-1]
    assert np.allclose(last_quat, q_target, atol=1e-9)


def test_align_to_skips_ramp_when_already_at_target():
    """theta_total==0 (already at hold_quat) must still take the fast,
    single-publish path even with the ramp configured -- mirrors the
    pre-ramp behavior for a no-op align."""
    q_target = [0.0, 0.0, 0.0, 1.0]
    tf = FakeTf([0.0, 0.0, 0.0], q_target)
    checkpoint_pub = FakeCheckpointPublisher()

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
        align_angular_speed_deg=15.0, align_angular_accel_deg=2.4,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    # cur_quat already matches arrival_target_quat within tolerance, so
    # execute() never even calls _align_to here (see the geodesic_angle
    # pre-check in execute()) -- no checkpoint is published at all.
    assert status == STATUS_SUCCESS
    assert checkpoint_pub.published == []


def test_align_to_ramp_respects_cancel():
    """A cancel request mid-ramp must return STATUS_CANCELED immediately,
    same contract as the rest of this class's cancel-checking loops."""
    q_from = [0.0, 0.0, 1.0, 0.0]  # 180 deg off -> long ramp duration
    q_target = [0.0, 0.0, 0.0, 1.0]
    tf = FakeTf([0.0, 0.0, 0.0], q_from)
    checkpoint_pub = FakeCheckpointPublisher()

    calls = {"n": 0}

    def is_cancel_requested():
        calls["n"] += 1
        return calls["n"] > 2

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=1.0),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
        align_angular_speed_deg=15.0, align_angular_accel_deg=2.4,
        align_traj_publish_rate_hz=20.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=is_cancel_requested,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_CANCELED


def test_execute_align_at_arrival_camera_main_uses_target_orientation_as_is():
    checkpoint_pub = FakeCheckpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])  # 180 deg off from q_target

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
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
        align_pos_timeout=0.1,
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
        align_pos_timeout=0.1,
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


def test_execute_aborts_when_initial_tf_pose_is_stale():
    """A dead TF publisher leaves tf2's buffer handing back the same frozen
    sample forever -- get_pose() keeps succeeding, but its stamp never
    advances. A goal that starts against such a stream must abort rather
    than plan against a pose GuidanceExecutor cannot confirm is current
    (docs/guidance_realtime_replanning_design.md 6-8)."""
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], stamp=5.0)
    logger = FakeLogger()
    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.1), logger,
        target_speed=1000.0, align_pos_tolerance_m=2.0,
        align_pos_settle_time=0.01, align_pos_timeout=0.2,
        tf_staleness_timeout=0.05,
    )
    # First goal: the frozen stamp is this executor's very first-ever
    # observation, so it is (correctly) treated as fresh and the goal
    # completes normally -- staleness can only be judged relative to a
    # prior sighting.
    first_status = executor.execute(
        [0.001, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert first_status == STATUS_SUCCESS

    # Second goal: same frozen stamp, but by now well beyond
    # tf_staleness_timeout since it was first observed (the first goal's own
    # loop already advanced sim time past it) -- the initial pose fetch must
    # be rejected as stale.
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_ABORTED
    assert any("stale" in w for w in logger.warnings)


def test_run_trajectory_ignores_stale_tf_position_for_convergence():
    """Once TF stops advancing mid-wait, a frozen position sample must not
    count toward arrival convergence even though its value already matches
    p_target -- otherwise a dead TF publisher would look identical to
    'arrived'."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    # Stamp advances through execute()'s init fetch (1.0) and the first two
    # post-duration polls (2.0, 3.0), then freezes at 3.0 -- as if the TF
    # publisher died right as the position-convergence wait began.
    tf = ScriptedPosTf(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0, 1.0],
        stamp=[1.0, 2.0, 3.0],
    )
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.2),
        logger, target_speed=1000.0, align_pos_tolerance_m=0.05,
        align_pos_settle_time=0.3, align_pos_timeout=0.5,
        tf_staleness_timeout=0.1,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert any("did not converge" in w for w in logger.warnings)


def test_execute_replanning_minco_v3_mode_reaches_target():
    """trajectory_tracking_mode="replanning_minco_v3" (docs/
    2026-09-20_ego_v2_style_replan_migration_plan.md). Does NOT pass
    velocity_fn -- ReplanningMincoV3Tracker never reads it. Uses an
    ideal-tracking TF fake: this tracker replans from its own reference
    state (EGO-Planner v2), so a TF fake advancing independently of the
    commanded setpoint would converge while the reference is still mid-route."""
    pytest.importorskip("minco_native_py")

    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = SetpointFollowingTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], setpoint_pub)
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=1.5, align_pos_timeout=8.0,
        velocity_fn=None,
    )
    status = executor.execute(
        [2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco_v3",
    )
    assert status == STATUS_SUCCESS
    assert not any("falling back to 'static'" in w for w in logger.warnings)
    final_p, _v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [2.0, 0.0, 0.0], atol=0.15)


def test_execute_replanning_minco_v3_mode_terminates_with_steady_state_offset():
    """Live-sim regression (docs/2026-09-23_replanning_minco_v3_fix_live_sim_
    verification.md): a constant in-tolerance TF offset used to keep the
    ETA-based total_duration ahead of elapsed forever, so the goal never ended."""
    pytest.importorskip("minco_native_py")

    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = SetpointFollowingTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], setpoint_pub,
                             offset=(0.0, 0.02, 0.0))
    clock_seconds_fn, spin_fn = _make_clock(dt_per_spin=0.05)
    local_speed_path_pub = FakeSpeedPathPublisher()
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), clock_seconds_fn, spin_fn,
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=1.5, align_pos_timeout=8.0,
        velocity_fn=None,
        local_speed_path_publisher=local_speed_path_pub,
    )
    status = executor.execute(
        [2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None,
        is_cancel_requested=lambda: clock_seconds_fn() > 300.0,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco_v3",
    )
    assert status == STATUS_SUCCESS
    assert not any("did not converge" in w for w in logger.warnings)
    final_p, _v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [2.0, 0.0, 0.0], atol=1e-3)
    assert len(local_speed_path_pub.calls) >= 1
    last_local_end, _speed = local_speed_path_pub.calls[-1][-1]
    assert np.allclose(last_local_end, [2.0, 0.0, 0.0], atol=1e-3)


def test_execute_replanning_minco_v3_mode_falls_back_to_static_without_max_accel():
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=None,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=1.0,
    )
    executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco_v3",
    )
    assert any(
        "replanning_minco_v3' requires max_accel" in w for w in logger.warnings
    )


def test_execute_replanning_minco_v3_mode_passes_via_waypoints_to_the_tracker(monkeypatch):
    """Wiring check: execute()'s via_waypoints must reach the
    ReplanningMincoV3Tracker constructor, not just the static-mode
    Trajectory (covered by
    test_execute_via_waypoints_routes_the_planned_curve_through_the_relay_points)."""
    pytest.importorskip("minco_native_py")
    import sobits_intball2_gnc.guidance.utils.guidance_executor as ge_module

    captured = {}
    real_tracker_cls = ge_module.ReplanningMincoV3Tracker

    class SpyTracker(real_tracker_cls):
        def __init__(self, *args, **kwargs):
            captured["route_waypoints"] = kwargs.get("route_waypoints")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ge_module, "ReplanningMincoV3Tracker", SpyTracker)

    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    via_waypoints = [[0.5, 0.5, 0.0]]
    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.05), FakeLogger(),
        target_speed=1.0, max_accel=0.02,
    )
    executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: True,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco_v3", via_waypoints=via_waypoints,
    )
    assert np.allclose(captured["route_waypoints"], via_waypoints)


def test_execute_replanning_minco_v3_mode_republishes_speed_path_preview_on_replan():
    """The speed-path preview must be re-published beyond the initial
    goal-start call once the tracker actually re-plans, so RViz doesn't show
    a stale first-plan path ([G] "再計画軌道のRVizプレビュー更新" task). The
    goal lies beyond the default 2m planning horizon, otherwise the first
    local plan already touches the goal and never re-plans."""
    pytest.importorskip("minco_native_py")

    setpoint_pub = FakeSetpointPublisher()
    speed_path_pub = FakeSpeedPathPublisher()
    tf = SetpointFollowingTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], setpoint_pub)
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        FakeLogger(), target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=1.5, align_pos_timeout=8.0,
        speed_path_publisher=speed_path_pub,
    )
    status = executor.execute(
        [5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco_v3",
    )
    assert status == STATUS_SUCCESS
    # 1 initial goal-start preview + at least one more from an actual re-plan.
    assert len(speed_path_pub.calls) > 1


def test_execute_static_mode_publishes_speed_path_preview_only_once():
    """Contrast with the re-planning case: static mode never re-plans,
    so the preview must stay published exactly once per goal, unchanged
    from prior behavior."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    speed_path_pub = FakeSpeedPathPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        logger, target_speed=1.0, align_pos_timeout=0.1,
        speed_path_publisher=speed_path_pub,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
    )
    assert status == STATUS_SUCCESS
    assert len(speed_path_pub.calls) == 1


@pytest.mark.parametrize(
    "removed_mode", ["replanning", "replanning_minco", "replanning_minco_v2"])
def test_execute_removed_trajectory_tracking_modes_fall_back_to_static(removed_mode):
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        logger, target_speed=1.0, max_accel=0.02, align_pos_timeout=0.1,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode=removed_mode,
    )
    assert status == STATUS_SUCCESS
    assert any("unknown trajectory_tracking_mode" in w for w in logger.warnings)


def test_execute_unknown_trajectory_tracking_mode_falls_back_to_static():
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        logger, target_speed=1.0, align_pos_timeout=0.1,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="bogus",
    )
    assert status == STATUS_SUCCESS
    assert any("falling back to 'static'" in w for w in logger.warnings)


def test_align_to_ignores_stale_tf_attitude_for_convergence():
    """Same staleness protection as the position-convergence test above,
    but for _align_to's attitude-convergence wait."""
    q_target = [0.0, 0.0, 0.0, 1.0]
    q_far = [0.0, 0.0, 1.0, 0.0]
    # Calls 0-3 (execute()'s initial fetch, _run_trajectory's two
    # position-convergence polls, and align_at_arrival's own pre-check) all
    # see q_far with a distinct, fresh stamp each time -- all four correctly
    # read as live TF, and the pre-check correctly finds the vehicle still
    # misaligned, triggering _align_to. From call 4 on (inside _align_to's
    # own poll loop), the pose reports q_target (would-be convergence), but
    # the stamp freezes at 5.0 -- as if the TF publisher died right as the
    # alignment wait began.
    tf = ScriptedTf(
        [0.0, 0.0, 0.0], [q_far] * 4 + [q_target] * 20,
        stamp=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    logger = FakeLogger()

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.2), logger,
        target_speed=1000.0, align_tolerance_deg=3.0, align_timeout=0.6,
        align_settle_time=0.3, tf_staleness_timeout=0.1,
        # Position convergence isn't under test here -- make it trivial and
        # fast so it doesn't interfere with the attitude-side assertion.
        align_pos_tolerance_m=2.0, align_pos_settle_time=0.01,
        align_pos_timeout=0.1,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], q_target,
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=True,
    )
    assert status == STATUS_SUCCESS
    assert any("alignment did not converge" in w for w in logger.warnings)


def test_execute_via_waypoints_routes_the_planned_curve_through_the_relay_points():
    """Static-mode routing check (docs/
    2026-08-25_guidance_waypoint_insertion_curve_verification.md, generalized
    from a single point to a list 2026-08-31): with a non-collinear via
    waypoint, the planned Hermite curve must actually pass near it -- unlike
    the old 2-waypoint straight line p0->p_target, which would never come
    near an off-line via waypoint."""
    setpoint_pub = FakeSetpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # stationary; only used
    # for the (irrelevant here) post-duration convergence poll, not for the
    # static trajectory's own shape.
    via_waypoints = [[1.0, 1.0, 0.0]]
    p_target = [2.0, 0.0, 0.0]

    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False, pre_align=False,
        via_waypoints=via_waypoints,
    )
    assert status == STATUS_SUCCESS
    positions = np.array([p for p, _v, _a, _q in setpoint_pub.calls])
    min_dist_to_via = np.min(
        np.linalg.norm(positions - np.array(via_waypoints[0]), axis=1)
    )
    assert min_dist_to_via < 0.05
    assert np.allclose(positions[-1], p_target, atol=1e-6)


def test_execute_pre_aligns_toward_via_waypoint_not_final_target():
    """pre_align must face the first leg (p0 -> first via waypoint), not the
    chord to the final target, when via waypoints are given."""
    checkpoint_pub = FakeCheckpointPublisher()
    q0 = [0.0, 0.0, 0.0, 1.0]  # facing +X
    tf = FakeTf([0.0, 0.0, 0.0], q0)
    via_waypoints = [[0.0, 1.0, 0.0]]  # +Y -- a very different direction from
    p_target = [1.0, 1.0, 0.0]         # the chord straight to p_target

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False, via_waypoints=via_waypoints,
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
    _pos, quat = checkpoint_pub.published[0]
    expected = compute_q_des(
        np.array(via_waypoints[0]) - np.array([0.0, 0.0, 0.0]),
        q0, 0.02, (1.0, 0.0, 0.0),
    )
    assert np.allclose(quat, expected, atol=1e-6)


