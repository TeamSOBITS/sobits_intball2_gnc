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
    """Stands in for TfClient: returns whatever pose the test sets."""

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
