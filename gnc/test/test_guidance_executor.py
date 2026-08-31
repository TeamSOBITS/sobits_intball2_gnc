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


class MovingTowardTf:
    """A TF fake that actually advances toward a target position by up to
    ``step`` meters on every ``get_pose()`` call, clamping at the target and
    holding there once reached -- unlike ``FakeTf``/``ScriptedPosTf``'s fixed
    or pre-scripted values, this lets a test exercise a genuine TF-feedback
    loop (e.g. ``trajectory_tracking_mode="replanning"``) that requires
    convergence over many polls to terminate, not just a single fixed pose
    (docs/guidance_realtime_replanning_design.md 6-9 節)."""

    def __init__(self, pos, target, quat, step=0.05, stamp=None):
        self.pos = list(pos)
        self._target = np.asarray(target, dtype=float)
        self.quat = quat
        self._step = float(step)
        self._stamp = stamp
        self._state = {"n": 0}

    def get_pose(self):
        current = np.asarray(self.pos, dtype=float)
        delta = self._target - current
        dist = np.linalg.norm(delta)
        current = self._target.copy() if dist <= self._step else current + delta / dist * self._step
        self.pos = current.tolist()
        return list(self.pos), list(self.quat), _next_stamp(self._state, self._stamp)


class DisturbedApproachTf:
    """Like ``MovingTowardTf``, but on the ``disturb_after``-th ``get_pose()``
    call it reports one large one-off displacement from wherever the vehicle
    currently is (simulating an instantaneous collision knock), then resumes
    the normal step-wise approach toward the target *from the disturbed
    position*. Unlike ``MovingTowardTf``, which only ever exercises monotonic
    convergence, this lets a test verify that
    ``trajectory_tracking_mode="replanning"`` actually recovers from a
    mid-flight disturbance rather than just tracking an undisturbed path
    (docs/main_plan.md's outstanding "擬似衝突からの復帰再現" verification
    item, see docs/archive/achieved/
    2026-08-25_guidance_realtime_replanning_sim_verification.md 7 節)."""

    def __init__(self, pos, target, quat, step=0.05, disturb_after=5,
                 disturbance=(0.0, 0.0, 0.0), stamp=None):
        self.pos = list(pos)
        self._target = np.asarray(target, dtype=float)
        self.quat = quat
        self._step = float(step)
        self._disturb_after = int(disturb_after)
        self._disturbance = np.asarray(disturbance, dtype=float)
        self._n = 0
        self._stamp = stamp
        self._state = {"n": 0}

    def get_pose(self):
        self._n += 1
        current = np.asarray(self.pos, dtype=float)
        if self._n == self._disturb_after:
            current = current + self._disturbance
        else:
            delta = self._target - current
            dist = np.linalg.norm(delta)
            current = self._target.copy() if dist <= self._step else current + delta / dist * self._step
        self.pos = current.tolist()
        return list(self.pos), list(self.quat), _next_stamp(self._state, self._stamp)


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
    speed-path preview -- once at goal start, and again on every re-plan in
    ``trajectory_tracking_mode="replanning"`` (docs/main_plan.md [G]
    "再計画軌道のRVizプレビュー更新")."""

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


def test_execute_replanning_mode_reaches_target():
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = MovingTowardTf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], step=0.05)
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=2.0,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
        distance_fallback_m=0.3, replan_rate_hz=20.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning",
    )
    assert status == STATUS_SUCCESS
    assert not any("falling back to 'static'" in w for w in logger.warnings)
    final_p, _v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [1.0, 0.0, 0.0], atol=0.1)


def test_execute_replanning_minco_mode_reaches_target():
    """Same shape as test_execute_replanning_mode_reaches_target above, but
    for trajectory_tracking_mode="replanning_minco" (docs/archive/achieved/
    2026-08-30_minco_attitude_torque_status_and_next_steps.md). Skips if
    minco_native_py isn't built. replan_rate_hz kept low relative to the sim
    rate (unlike the Heuristic-backed test above) since each re-plan here
    runs a real L-BFGS solve, not a closed-form heuristic."""
    pytest.importorskip("minco_native_py")

    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = MovingTowardTf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], step=0.05)
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=5.0,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
        distance_fallback_m=0.3, replan_rate_hz=5.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning_minco",
    )
    assert status == STATUS_SUCCESS
    assert not any("falling back to 'static'" in w for w in logger.warnings)
    final_p, _v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [1.0, 0.0, 0.0], atol=0.15)


def test_execute_replanning_mode_passes_via_waypoint_to_the_tracker(monkeypatch):
    """Wiring check: execute()'s via_waypoint must reach the
    ReplanningTrajectoryTracker constructor it builds (docs/
    2026-08-25_guidance_waypoint_insertion_curve_verification.md step 2), not
    just the initial static-mode Trajectory both modes share (step 1,
    already covered by
    test_execute_via_waypoint_routes_the_planned_curve_through_the_relay_point).

    Deliberately does not run a real simulated flight to observe the
    resulting curve: a fake TF that never advances (needed to keep
    via_waypoint "pending" long enough to observe in the published path)
    combined with a zero v0 reproduces exactly the Zeno-style non-termination
    docs/guidance_realtime_replanning_design.md 6-1 節 warns about --
    ``HeuristicSegmentTimeAllocator``'s v0-aware bound pushes
    ``total_duration`` further out on every re-plan when neither the live
    pose nor v0 ever change, so ``_run_trajectory`` never reaches its
    post-duration convergence check. Spying on the constructor call instead
    sidesteps that hazard entirely while still proving the wiring."""
    import sobits_intball2_gnc.guidance.utils.guidance_executor as ge_module

    captured = {}
    real_tracker_cls = ge_module.ReplanningTrajectoryTracker

    class SpyTracker(real_tracker_cls):
        def __init__(self, *args, **kwargs):
            captured["via_waypoint"] = kwargs.get("via_waypoint")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ge_module, "ReplanningTrajectoryTracker", SpyTracker)

    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    via_waypoint = [0.5, 0.5, 0.0]
    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.05), FakeLogger(),
        target_speed=1.0, max_accel=0.02,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
    )

    def is_cancel_requested():
        return True  # cancel on the very first check -- only the wiring is under test

    executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=is_cancel_requested,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning", via_waypoint=via_waypoint,
    )
    assert np.allclose(captured["via_waypoint"], via_waypoint)


def test_execute_replanning_mode_recovers_from_mid_flight_disturbance():
    """docs/main_plan.md's outstanding verification item: a pseudo-collision
    that knocks the vehicle off its planned path mid-flight must still let
    trajectory_tracking_mode="replanning" reach the goal, since every
    re-plan re-targets from the *current* TF pose rather than the originally
    planned one. The disturbance (-0.4, +0.3, 0) is well past
    distance_fallback_m (0.3m) away from the target, so it forces at least
    one more genuine re-plan (not just the near-target fallback leg)."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = DisturbedApproachTf(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        step=0.05, disturb_after=5, disturbance=(-0.4, 0.3, 0.0),
    )
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=5.0,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
        distance_fallback_m=0.3, replan_rate_hz=20.0,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning",
    )
    assert status == STATUS_SUCCESS
    assert not any("falling back to 'static'" in w for w in logger.warnings)
    final_p, _v, _a, _q = setpoint_pub.calls[-1]
    assert np.allclose(final_p, [1.0, 0.0, 0.0], atol=0.1)


def test_execute_replanning_mode_republishes_speed_path_preview_on_replan():
    """The speed-path preview must be re-published beyond the initial
    goal-start call once trajectory_tracking_mode="replanning" actually
    re-plans, so RViz doesn't show a stale first-plan path (docs/
    main_plan.md [G] "再計画軌道のRVizプレビュー更新")."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    speed_path_pub = FakeSpeedPathPublisher()
    tf = MovingTowardTf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], step=0.05)
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
        logger, target_speed=1.0, max_accel=0.02,
        align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=2.0,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
        distance_fallback_m=0.3, replan_rate_hz=20.0,
        speed_path_publisher=speed_path_pub,
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning",
    )
    assert status == STATUS_SUCCESS
    # 1 initial goal-start preview + at least one more from an actual re-plan.
    assert len(speed_path_pub.calls) > 1


def test_execute_static_mode_publishes_speed_path_preview_only_once():
    """Contrast with the replanning case above: static mode never re-plans,
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


def test_execute_replanning_mode_falls_back_to_static_without_max_accel():
    """max_accel is mandatory for replanning (HeuristicSegmentTimeAllocator's
    v0-aware bound needs it) -- GuidanceExecutor must fall back to static
    with a warning rather than let ReplanningTrajectoryTracker's ValueError
    propagate out of execute()."""
    setpoint_pub = FakeSetpointPublisher()
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        logger, target_speed=1.0, align_pos_timeout=0.1,
        velocity_fn=lambda: FakeVelocityEstimate([0.0, 0.0, 0.0]),
        # max_accel intentionally left at its default (None).
    )
    status = executor.execute(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False,
        trajectory_tracking_mode="replanning",
    )
    assert status == STATUS_SUCCESS
    assert any("falling back to 'static'" in w for w in logger.warnings)


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


def test_execute_via_waypoint_routes_the_planned_curve_through_the_relay_point():
    """Static-mode routing check (docs/
    2026-08-25_guidance_waypoint_insertion_curve_verification.md): with a
    non-collinear via_waypoint, the planned Hermite curve must actually pass
    near it -- unlike the old 2-waypoint straight line p0->p_target, which
    would never come near an off-line via_waypoint."""
    setpoint_pub = FakeSetpointPublisher()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # stationary; only used
    # for the (irrelevant here) post-duration convergence poll, not for the
    # static trajectory's own shape.
    via_waypoint = [1.0, 1.0, 0.0]
    p_target = [2.0, 0.0, 0.0]

    executor = GuidanceExecutor(
        tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False, pre_align=False,
        via_waypoint=via_waypoint,
    )
    assert status == STATUS_SUCCESS
    positions = np.array([p for p, _v, _a, _q in setpoint_pub.calls])
    min_dist_to_via = np.min(np.linalg.norm(positions - np.array(via_waypoint), axis=1))
    assert min_dist_to_via < 0.05
    assert np.allclose(positions[-1], p_target, atol=1e-6)


def test_execute_pre_aligns_toward_via_waypoint_not_final_target():
    """pre_align must face the first leg (p0 -> via_waypoint), not the chord
    to the final target, when a via_waypoint is given."""
    checkpoint_pub = FakeCheckpointPublisher()
    q0 = [0.0, 0.0, 0.0, 1.0]  # facing +X
    tf = FakeTf([0.0, 0.0, 0.0], q0)
    via_waypoint = [0.0, 1.0, 0.0]  # +Y -- a very different direction from
    p_target = [1.0, 1.0, 0.0]      # the chord straight to p_target

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), checkpoint_pub, *_make_clock(dt_per_spin=0.1),
        FakeLogger(), target_speed=1.0, align_tolerance_deg=3.0, align_timeout=0.2,
        align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=True, align_at_arrival=False, via_waypoint=via_waypoint,
    )
    assert status == STATUS_SUCCESS
    assert len(checkpoint_pub.published) == 1
    _pos, quat = checkpoint_pub.published[0]
    expected = compute_q_des(
        np.array(via_waypoint) - np.array([0.0, 0.0, 0.0]),
        q0, 0.02, (1.0, 0.0, 0.0),
    )
    assert np.allclose(quat, expected, atol=1e-6)


def test_execute_warns_on_sharp_via_waypoint_turn():
    """A via_waypoint bending the route more than 90 deg cannot actually be
    flown without stopping first (not implemented) -- GuidanceExecutor logs
    a warning instead of silently planning an unflyable curve."""
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    via_waypoint = [1.0, 0.0, 0.0]
    p_target = [0.5, 1.0, 0.0]  # leg1=(1,0,0), leg2=(-0.5,1,0) -> ~117 deg

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.1), logger, target_speed=1.0,
        align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False, via_waypoint=via_waypoint,
    )
    assert status == STATUS_SUCCESS
    assert any("turn angle" in w for w in logger.warnings)


def test_execute_no_via_waypoint_warning_for_a_gentle_turn():
    """Sanity check: a turn well under 90 deg must not trigger the sharp-turn
    warning (only the >90 deg case should)."""
    logger = FakeLogger()
    tf = FakeTf([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    via_waypoint = [1.0, 0.0, 0.0]
    p_target = [2.0, 0.5, 0.0]  # leg1=(1,0,0), leg2=(1,0.5,0) -> well under 90 deg

    executor = GuidanceExecutor(
        tf, FakeSetpointPublisher(), FakeCheckpointPublisher(),
        *_make_clock(dt_per_spin=0.1), logger, target_speed=1.0,
        align_pos_timeout=0.1,
    )
    status = executor.execute(
        p_target, [0.0, 0.0, 0.0, 1.0],
        feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
        face_travel=False, align_at_arrival=False, via_waypoint=via_waypoint,
    )
    assert status == STATUS_SUCCESS
    assert not any("turn angle" in w for w in logger.warnings)
