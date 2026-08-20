"""Unit tests for the hover control kernels (plain-value, no ROS)."""
import math

from sobits_intball2_gnc.control.utils.hover_controller import (
    HoverController,
    HoverLaw,
    PoseCorrector,
    STATUS_MISSING,
    STATUS_OFF,
    STATUS_OK,
    STATUS_STALE,
    TrajectoryController,
)


def test_gyro_damping_produces_opposing_torque():
    law = HoverLaw(kd_w=[0.02, 0.02, 0.02], deadband_w=0.01)
    _, torque = law.compute([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    # torque = -kd_w * gyro = -0.02 * 0.1
    assert math.isclose(torque[0], -0.002, abs_tol=1e-9)
    assert torque[1] == 0.0 and torque[2] == 0.0


def test_gyro_deadband_zeros_small_rates():
    law = HoverLaw(deadband_w=0.01)
    _, torque = law.compute([0.005, -0.009, 0.0], [0.0, 0.0, 0.0])
    assert torque == [0.0, 0.0, 0.0] or all(abs(t) == 0.0 for t in torque)


def test_output_clamped_to_limits():
    law = HoverLaw(kd_w=[10, 10, 10], max_torque=0.02, deadband_w=0.0)
    _, torque = law.compute([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert all(abs(t) <= 0.02 + 1e-12 for t in torque)


def test_feedforward_force_is_added():
    law = HoverLaw(kp_a=[0.0, 0.0, 0.0], max_force=1.0)
    force, _ = law.compute([0, 0, 0], [0, 0, 0], feedforward_force=[0.1, 0.0, 0.0])
    assert math.isclose(force[0], 0.1, abs_tol=1e-9)


# --- PoseCorrector: pose handling -------------------------------------------

_IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _corrector(**kwargs):
    """Build a corrector with intake gating effectively disabled."""
    kwargs.setdefault("poll_rate", 1000.0)
    kwargs.setdefault("smooth_window", 3)
    kwargs.setdefault("smooth_sigma", 1.0)
    kwargs.setdefault("timeout", 1.0)
    return PoseCorrector(**kwargs)


def test_pose_corrector_zero_without_pose():
    pc = _corrector()
    f, t = pc.update(0.0, None)
    assert f == [0.0, 0.0, 0.0] and t == [0.0, 0.0, 0.0]
    assert pc.status == STATUS_MISSING


def test_pose_corrector_holds_first_pose_zero_error():
    pc = _corrector()
    for i in range(3):
        pc.update(i * 0.01, ([1.0, 2.0, 3.0], _IDENTITY, 100.0 + i * 0.01))
    f, t = pc.update(0.03, ([1.0, 2.0, 3.0], _IDENTITY, 100.03))
    # Steady pose becomes the hold target -> ~zero correction.
    assert all(abs(v) < 1e-6 for v in f)
    assert all(abs(v) < 1e-6 for v in t)


def test_position_error_produces_force_toward_target():
    pc = _corrector(smooth_window=1, kp_pos=[0.1, 0.1, 0.1], max_corr_force=10.0)
    # First pose captures the hold target at the origin.
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    # Drifted +x: correction should push back along -x.
    f, _ = pc.update(0.1, ([1.0, 0.0, 0.0], _IDENTITY, 100.1))
    assert math.isclose(f[0], -0.1, abs_tol=1e-9)
    assert abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9


def test_position_error_rotated_into_body_frame():
    # Attitude = 180 deg about z. A reference-frame -x force becomes +x in body.
    quat_z180 = [0.0, 0.0, 1.0, 0.0]
    pc = _corrector(smooth_window=1, kp_pos=[0.1, 0.1, 0.1], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], quat_z180, 100.0))
    f, _ = pc.update(0.1, ([1.0, 0.0, 0.0], quat_z180, 100.1))
    assert math.isclose(f[0], 0.1, abs_tol=1e-9)


def test_correction_clamped_independently():
    pc = _corrector(smooth_window=1, kp_pos=[10.0, 10.0, 10.0],
                    kp_att=[10.0, 10.0, 10.0],
                    max_corr_force=0.05, max_corr_torque=0.01)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    f, t = pc.update(0.1, ([5.0, 5.0, 5.0], [0.3, 0.0, 0.0, 0.954], 100.1))
    assert all(abs(v) <= 0.05 + 1e-12 for v in f)
    assert all(abs(v) <= 0.01 + 1e-12 for v in t)


def test_smoothing_disabled_with_window_one():
    pc = _corrector(smooth_window=1, kp_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    # With window=1 the latest pose is used verbatim: no averaging with history.
    f, _ = pc.update(0.1, ([0.5, 0.0, 0.0], _IDENTITY, 100.1))
    assert math.isclose(f[0], -0.5, abs_tol=1e-9)


def test_kd_pos_defaults_to_zero_no_damping():
    # kd_pos defaults to [0, 0, 0]: a moving target produces pure-P force,
    # unaffected by the velocity estimate (Phase 0 damping trial, opt-in).
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    f, _ = pc.update(0.1, ([0.5, 0.0, 0.0], _IDENTITY, 100.1))
    assert f == [0.0, 0.0, 0.0]


def test_kd_pos_damps_hold_target_velocity():
    # Isolate the D term (kp_pos=0): force = -kd_pos * vel, vel = dpos/dt.
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0],
                    kd_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    # vel = (0.5 - 0.0) / (0.1 - 0.0) = 5.0 -> force = -1.0 * 5.0 = -5.0
    f, _ = pc.update(0.1, ([0.5, 0.0, 0.0], _IDENTITY, 100.1))
    assert math.isclose(f[0], -5.0, abs_tol=1e-9)


def test_vel_filter_alpha_defaults_to_no_filtering():
    # vel_filter_alpha defaults to 1.0: filtered velocity == raw finite
    # difference on every tick, matching prior (pre-filter) behavior exactly
    # (0.5*raw + 0.5*prev collapses to raw when alpha=1.0).
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0],
                    kd_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    f, _ = pc.update(0.1, ([0.5, 0.0, 0.0], _IDENTITY, 100.1))
    assert math.isclose(f[0], -5.0, abs_tol=1e-9)


def test_vel_filter_alpha_smooths_a_velocity_step():
    # alpha < 1 blends the new raw sample with the previous filtered value,
    # so a step in raw velocity is only partially reflected on the next tick.
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0],
                    kd_pos=[1.0, 1.0, 1.0], vel_filter_alpha=0.5,
                    max_corr_force=10.0)
    # First sample: no prior pos to difference against -> raw vel = 0, so the
    # filter (starting at zero) stays at zero too.
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    # raw vel = (0.1-0.0)/0.1 = 1.0 -> filtered = 0.5*1.0 + 0.5*0.0 = 0.5.
    f1, _ = pc.update(0.1, ([0.1, 0.0, 0.0], _IDENTITY, 100.1))
    assert math.isclose(f1[0], -0.5, abs_tol=1e-9)
    # raw vel jumps to (0.6-0.1)/0.1 = 5.0 -> filtered = 0.5*5.0 + 0.5*0.5 = 2.75.
    f2, _ = pc.update(0.2, ([0.6, 0.0, 0.0], _IDENTITY, 100.2))
    assert math.isclose(f2[0], -2.75, abs_tol=1e-9)


def test_vel_filter_resets_after_tf_loss():
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0],
                    kd_pos=[1.0, 1.0, 1.0], vel_filter_alpha=0.5,
                    max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc.update(0.1, ([0.1, 0.0, 0.0], _IDENTITY, 100.1))
    pc.update(0.2, None)  # TF loss
    # Reacquire: no prior sample, so raw vel is zero on this first post-loss
    # tick and the filter must reset to zero rather than carry over the
    # pre-loss filtered value.
    f, _ = pc.update(0.3, ([5.0, 0.0, 0.0], _IDENTITY, 100.3))
    assert f == [0.0, 0.0, 0.0]


def test_kd_pos_velocity_resets_after_tf_loss():
    # A TF loss must not leave a stale sample that reads back as a spurious
    # huge velocity once TF reacquires.
    pc = _corrector(smooth_window=1, kp_pos=[0.0, 0.0, 0.0],
                    kd_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc.update(0.1, None)  # TF loss
    # Reacquire far away: with no prior sample, velocity must read as zero.
    f, _ = pc.update(0.2, ([5.0, 0.0, 0.0], _IDENTITY, 100.2))
    assert f == [0.0, 0.0, 0.0]


# --- PoseCorrector: liveness by timestamp advance ----------------------------


def test_repeated_stamp_within_timeout_is_accepted():
    pc = _corrector(smooth_window=1, timeout=1.0, kp_pos=[0.1, 0.1, 0.1],
                    max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    # Same stamp 0.5 s later: TF slower than the loop, still live.
    f, _ = pc.update(0.5, ([1.0, 0.0, 0.0], _IDENTITY, 100.0))
    assert pc.status == STATUS_OK
    assert math.isclose(f[0], -0.1, abs_tol=1e-9)


def test_stalled_stamp_past_timeout_goes_stale():
    pc = _corrector(smooth_window=1, timeout=0.5)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    f, t = pc.update(10.0, ([1.0, 0.0, 0.0], _IDENTITY, 100.0))
    assert pc.status == STATUS_STALE
    assert f == [0.0, 0.0, 0.0] and t == [0.0, 0.0, 0.0]


def test_advancing_stamp_stays_live():
    pc = _corrector(smooth_window=1, timeout=0.5)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    for i in range(1, 30):
        pc.update(i * 0.1, ([0.0, 0.0, 0.0], _IDENTITY, 100.0 + i * 0.1))
    assert pc.status == STATUS_OK


def test_zero_stamp_rejected_not_treated_as_reset():
    """A zero stamp means "unset" in ROS; it must not reset the reference."""
    pc = _corrector(smooth_window=1, timeout=1.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc.update(0.1, ([0.0, 0.0, 0.0], _IDENTITY, 0.0))
    assert pc.status == STATUS_MISSING
    # The real stream continues and is still recognised as advancing.
    pc.update(0.2, ([0.0, 0.0, 0.0], _IDENTITY, 100.1))
    assert pc.status == STATUS_OK


def test_stamp_regression_treated_as_clock_reset():
    pc = _corrector(smooth_window=1, timeout=0.5)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 500.0))
    # Simulator restarted: stamp jumps backwards. Must not report stale.
    pc.update(1.0, ([0.0, 0.0, 0.0], _IDENTITY, 3.0))
    assert pc.status == STATUS_OK


def test_liveness_independent_of_local_clock_scale():
    """Check that stamps on an unrelated clock (sim time) still gate correctly."""
    # The local time values here are ~1.7e9 (wall clock) while the stamps are
    # ~100 (sim time). Any implementation subtracting one from the other would
    # classify every sample as ancient.
    pc = _corrector(smooth_window=1, timeout=1.0, kp_pos=[0.1, 0.1, 0.1],
                    max_corr_force=10.0)
    base = 1_784_631_713.0
    pc.update(base, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    f, _ = pc.update(base + 0.1, ([1.0, 0.0, 0.0], _IDENTITY, 100.1))
    assert pc.status == STATUS_OK
    assert math.isclose(f[0], -0.1, abs_tol=1e-9)


def test_hold_target_recaptured_after_loss():
    pc = _corrector(smooth_window=1, timeout=0.5, kp_pos=[0.1, 0.1, 0.1],
                    max_corr_force=10.0)
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc.update(1.0, None)                       # lost -> hold target dropped
    assert pc.status == STATUS_MISSING
    pc.update(2.0, ([5.0, 0.0, 0.0], _IDENTITY, 200.0))   # recovered here
    f, _ = pc.update(2.1, ([5.0, 0.0, 0.0], _IDENTITY, 200.1))
    # New hold target is the recovery pose, not the pre-loss origin.
    assert all(abs(v) < 1e-6 for v in f)


def test_checkpoint_target_survives_loss():
    pc = _corrector(smooth_window=1, timeout=0.5, kp_pos=[0.1, 0.1, 0.1],
                    max_corr_force=10.0)
    pc.set_checkpoints([([0.0, 0.0, 0.0], _IDENTITY)])
    pc.update(0.0, ([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc.update(1.0, None)
    f, _ = pc.update(2.0, ([1.0, 0.0, 0.0], _IDENTITY, 200.0))
    # Checkpoint still the target -> pushed back toward the origin.
    assert math.isclose(f[0], -0.1, abs_tol=1e-9)


def test_checkpoint_advance():
    pc = _corrector()
    pc.set_checkpoints([
        ([0, 0, 0], [0, 0, 0, 1]),
        ([1, 0, 0], [0, 0, 0, 1]),
    ])
    assert pc.advance_checkpoint() is True
    assert pc.advance_checkpoint() is False  # already at last


# --- HoverController --------------------------------------------------------


class _FakeImu:
    def __init__(self, gyro=None, acc=None):
        self.gyro = gyro
        self.acc = acc


class _FakeFan:
    def __init__(self):
        self.last = None

    def set_duty_array(self, duties):
        self.last = list(duties)


class _FakeAllocator:
    def __init__(self):
        self.last_force = None
        self.last_torque = None

    def allocate(self, force, torque):
        self.last_force = list(force)
        self.last_torque = list(torque)
        return [0.5] * 8


class _FakeTf:
    """Stands in for TfClient (common/ros): returns whatever pose the test sets."""

    def __init__(self, pose=None):
        self.pose = pose
        self.calls = 0

    def get_pose(self):
        self.calls += 1
        return self.pose


def test_hover_controller_idles_without_imu():
    fan = _FakeFan()
    hc = HoverController(_FakeImu(), fan, _FakeAllocator(), HoverLaw())
    hc.step(0.0)
    assert fan.last == []  # no IMU yet -> idle


def test_hover_controller_publishes_with_imu():
    fan = _FakeFan()
    hc = HoverController(
        _FakeImu(gyro=[0.1, 0, 0], acc=[0, 0, 0]),
        fan, _FakeAllocator(), HoverLaw(),
    )
    hc.step(0.0)
    assert fan.last == [0.5] * 8


def test_imu_only_mode_never_touches_tf():
    tf = _FakeTf(pose=([1.0, 0.0, 0.0], _IDENTITY, 100.0))
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]),
        _FakeFan(), _FakeAllocator(), HoverLaw(),
        tf_client=None, corrector=_corrector(),
    )
    hc.step(0.0)
    assert tf.calls == 0
    assert hc.tf_status == STATUS_OFF


def test_missing_pose_leaves_imu_term_untouched():
    alloc_imu, alloc_tf = _FakeAllocator(), _FakeAllocator()
    imu = _FakeImu(gyro=[0.1, 0, 0], acc=[0, 0, 0])

    HoverController(imu, _FakeFan(), alloc_imu, HoverLaw()).step(0.0)

    tf = _FakeTf(pose=None)
    hc = HoverController(
        _FakeImu(gyro=[0.1, 0, 0], acc=[0, 0, 0]),
        _FakeFan(), alloc_tf, HoverLaw(),
        tf_client=tf, corrector=_corrector(),
    )
    hc.step(0.0)

    assert alloc_tf.last_force == alloc_imu.last_force
    assert alloc_tf.last_torque == alloc_imu.last_torque
    assert hc.tf_status == STATUS_MISSING


# --- HoverController: trajectory following / checkpoint-hold exclusivity ---
# (Phase 3a, openspec/changes/add-trajectory-following)


class _FakeTrajectorySub:
    """Stands in for MultiDOFJointTrajectorySubscriber: settable setpoint + liveness."""

    def __init__(self, p_des=None, v_des=None, a_des=None,
                 last_received_t=None):
        self.p_des = p_des
        self.v_des = v_des
        self.a_des = a_des
        self.q_des = _IDENTITY
        self.last_received_t = last_received_t

    @property
    def ready(self):
        return self.p_des is not None


def test_trajectory_active_uses_trajectory_force_not_checkpoint():
    tf = _FakeTf(pose=([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc = _corrector(smooth_window=1, kp_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    traj_ctrl = TrajectoryController(mass=1.0, kp_pos=[2.0, 2.0, 2.0],
                                      kd_pos=[0.0, 0.0, 0.0],
                                      vel_filter_alpha=1.0, max_force=10.0)
    traj_sub = _FakeTrajectorySub(
        p_des=[2.0, 0.0, 0.0], v_des=[0.0, 0.0, 0.0], a_des=[0.0, 0.0, 0.0],
        last_received_t=0.0,
    )
    alloc = _FakeAllocator()
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]), _FakeFan(), alloc,
        HoverLaw(max_force=100.0),
        tf_client=tf, corrector=pc,
        trajectory_subscriber=traj_sub, trajectory_controller=traj_ctrl,
        trajectory_timeout=0.2,
    )
    hc.step(0.0)
    # pc alone would push toward its captured hold target (the origin, i.e.
    # zero force); the trajectory force (kp * (p_des - pos) = 2 * 2.0) must
    # be what's actually used instead.
    assert math.isclose(alloc.last_force[0], 4.0, abs_tol=1e-9)
    assert hc.trajectory_active is True


def test_stale_trajectory_falls_back_to_checkpoint_force():
    tf = _FakeTf(pose=([0.0, 0.0, 0.0], _IDENTITY, 100.0))
    pc = _corrector(smooth_window=1, kp_pos=[1.0, 1.0, 1.0], max_corr_force=10.0)
    pc.set_checkpoints([([3.0, 0.0, 0.0], _IDENTITY)])
    traj_ctrl = TrajectoryController(max_force=10.0)
    # last_received_t far in the past -> stale at t=0.0 (timeout=0.2).
    traj_sub = _FakeTrajectorySub(
        p_des=[9.0, 0.0, 0.0], v_des=[0, 0, 0], a_des=[0, 0, 0],
        last_received_t=-10.0,
    )
    alloc = _FakeAllocator()
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]), _FakeFan(), alloc,
        HoverLaw(max_force=100.0),
        tf_client=tf, corrector=pc,
        trajectory_subscriber=traj_sub, trajectory_controller=traj_ctrl,
        trajectory_timeout=0.2,
    )
    hc.step(0.0)
    # Checkpoint hold pushes from [0,0,0] toward [3,0,0]: kp * (3-0) = 3.0.
    assert math.isclose(alloc.last_force[0], 3.0, abs_tol=1e-9)
    assert hc.trajectory_active is False


def test_fallback_recaptures_hold_target_without_jump():
    # A checkpoint hold target set up BEFORE trajectory following starts
    # (simulating a stale target left over from before the vehicle moved
    # under trajectory control).
    pc = _corrector(smooth_window=1, kp_pos=[1.0, 1.0, 1.0], max_corr_force=100.0)
    pc.update(-1.0, ([0.0, 0.0, 0.0], _IDENTITY, 50.0))  # old hold target: origin

    traj_ctrl = TrajectoryController(max_force=10.0)
    traj_sub = _FakeTrajectorySub(
        p_des=[9.0, 9.0, 9.0], v_des=[0, 0, 0], a_des=[0, 0, 0],
        last_received_t=0.0,
    )
    tf = _FakeTf(pose=([9.0, 9.0, 9.0], _IDENTITY, 100.0))
    alloc = _FakeAllocator()
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]), _FakeFan(), alloc,
        HoverLaw(max_force=100.0),
        tf_client=tf, corrector=pc,
        trajectory_subscriber=traj_sub, trajectory_controller=traj_ctrl,
        trajectory_timeout=0.2,
    )
    hc.step(0.0)  # trajectory active: vehicle "moves" to [9,9,9] (per fake TF)
    assert hc.trajectory_active is True

    # Trajectory goes stale -> fall back. Without a hold-target reset, pc
    # would still target the origin and push with kp*(0-9) = -9 per axis.
    traj_sub.last_received_t = -10.0
    hc.step(1.0)
    assert hc.trajectory_active is False
    assert all(abs(v) < 1e-6 for v in alloc.last_force)


def test_trajectory_controller_supplies_torque_when_q_des_is_live():
    # Phase 3b: once a q_des has been received, attitude torque comes from
    # TrajectoryController.compute_attitude, not PoseCorrector's hold target.
    quat_off = [0.3, 0.0, 0.0, 0.954]  # not aligned with either target
    pc = _corrector(smooth_window=1, kp_att=[10.0, 10.0, 10.0],
                    max_corr_torque=1.0)
    pc.update(-1.0, ([0.0, 0.0, 0.0], _IDENTITY, 50.0))  # captures IDENTITY as hold attitude
    tf = _FakeTf(pose=([0.0, 0.0, 0.0], quat_off, 100.0))
    traj_ctrl = TrajectoryController(max_force=10.0, kp_att=[1.0, 1.0, 1.0],
                                      max_torque=1.0)
    traj_sub = _FakeTrajectorySub(
        p_des=[0.0, 0.0, 0.0], v_des=[0, 0, 0], a_des=[0, 0, 0],
        last_received_t=0.0,
    )
    traj_sub.q_des = _IDENTITY  # Guidance's setpoint, distinct from pc's hold
    alloc = _FakeAllocator()
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]), _FakeFan(), alloc,
        HoverLaw(max_force=100.0, max_torque=100.0),
        tf_client=tf, corrector=pc,
        trajectory_subscriber=traj_sub, trajectory_controller=traj_ctrl,
        trajectory_timeout=0.2,
    )
    hc.step(0.0)
    assert hc.trajectory_active is True
    # kp_att differs (1.0 vs pc's 10.0): matching traj_ctrl's smaller torque
    # (not clamped the same way pc's would be) confirms the source.
    expected = TrajectoryController(
        max_force=10.0, kp_att=[1.0, 1.0, 1.0], max_torque=1.0,
    ).compute_attitude(0.0, quat_off, _IDENTITY)
    assert all(math.isclose(a, e, abs_tol=1e-9)
               for a, e in zip(alloc.last_torque, expected))


def test_torque_falls_back_to_corrector_when_q_des_unset():
    # Before Guidance's first setpoint (q_des still None), attitude torque
    # must keep coming from PoseCorrector even while translation is driven by
    # the trajectory controller.
    quat_off = [0.3, 0.0, 0.0, 0.954]  # not aligned with the hold attitude
    pc = _corrector(smooth_window=1, kp_att=[10.0, 10.0, 10.0],
                    max_corr_torque=1.0)
    pc.update(-1.0, ([0.0, 0.0, 0.0], _IDENTITY, 50.0))  # captures IDENTITY as hold attitude
    tf = _FakeTf(pose=([0.0, 0.0, 0.0], quat_off, 100.0))
    traj_ctrl = TrajectoryController(max_force=10.0)
    traj_sub = _FakeTrajectorySub(
        p_des=[0.0, 0.0, 0.0], v_des=[0, 0, 0], a_des=[0, 0, 0],
        last_received_t=0.0,
    )
    traj_sub.q_des = None
    alloc = _FakeAllocator()
    hc = HoverController(
        _FakeImu(gyro=[0, 0, 0], acc=[0, 0, 0]), _FakeFan(), alloc,
        HoverLaw(max_force=100.0, max_torque=100.0),
        tf_client=tf, corrector=pc,
        trajectory_subscriber=traj_sub, trajectory_controller=traj_ctrl,
        trajectory_timeout=0.2,
    )
    hc.step(0.0)
    assert hc.trajectory_active is True
    # Attitude correction must still be nonzero even though translation is
    # being driven by the trajectory controller.
    assert any(abs(v) > 1e-6 for v in alloc.last_torque)
