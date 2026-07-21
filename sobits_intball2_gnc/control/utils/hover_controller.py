#!/usr/bin/env python3
"""IMU-based hover control logic for IntBall2 (ROS-agnostic).

Holds a stable hover using IMU feedback: the angular rate (gyro) is damped
toward zero and the linear-acceleration disturbance (accelerometer, with an EMA
bias estimate removed) is opposed. In ``tf_imu`` mode a low-gain, independently
clamped position/attitude correction derived from the TF pose (``iss_body`` <-
``body``) is layered on top; losing TF degrades gracefully back to pure IMU
hover.

Layers:
- :class:`HoverLaw` -- pure IMU control law (plain-value constructor, testable).
- :class:`PoseCorrector` -- pure pose correction + checkpoint interface
  (plain-value constructor, testable).
- :class:`HoverController` -- DI orchestration logic: reads the injected
  ``ImuSubscriber`` and (optional) ``TfClient``, combines the IMU law with the
  pose correction, allocates via the injected :class:`ThrustAllocator`, and
  publishes via the injected ``FanDutyPublisher``. It performs no ROS I/O.
"""
import time
from collections import deque

import numpy as np

# --- code defaults -----------------------------------------------------------
DEFAULT_HOVER = {
    "control_rate": 50.0,
    "kd_w": [0.02, 0.02, 0.02],
    "kp_a": [0.5, 0.5, 0.5],
    "deadband_w": 0.01,
    "deadband_a": 0.02,
    "acc_bias_alpha": 0.01,
    "max_force": 0.1,
    "max_torque": 0.02,
    "mode": "tf_imu",
}
# TF-based correction. Defaults are re-derived for a pull source reading the
# Gazebo TF tree (near-truth, ~300 Hz) -- they are NOT the old navigation
# values, which were tuned against a 200 Hz stream with injected Gaussian noise.
DEFAULT_TF = {
    "reference_frame": "iss_body",
    "target_frame": "body",
    "poll_rate": 50.0,
    "timeout": 1.0,
    "smooth_window": 5,
    "smooth_sigma": 2.0,
    "kp_pos": [0.05, 0.05, 0.05],
    "kp_att": [0.01, 0.01, 0.01],
    "max_corr_force": 0.05,
    "max_corr_torque": 0.01,
    "checkpoint_topic": "/gnc/checkpoints",
}

HOVER_MODES = ("imu", "tf_imu")

# PoseCorrector.status values.
STATUS_OFF = "off"          # no TF client injected (imu mode)
STATUS_MISSING = "missing"  # lookup itself fails
STATUS_STALE = "stale"      # lookup succeeds but the stamp stopped advancing
STATUS_OK = "ok"


def _deadband(vec, threshold):
    """Zero out per-axis components whose magnitude is below ``threshold``."""
    return np.where(np.abs(vec) < threshold, 0.0, vec)


def _quat_conj(q):
    """Conjugate of quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a, b):
    """Hamilton product a ⊗ b for quaternions [x, y, z, w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_rotate(q, v):
    """Rotate vector ``v`` by quaternion ``q`` [x, y, z, w]."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]


class HoverLaw:
    """IMU-only hover control law (ROS-agnostic, testable)."""

    def __init__(
        self,
        kd_w=DEFAULT_HOVER["kd_w"],
        kp_a=DEFAULT_HOVER["kp_a"],
        deadband_w=DEFAULT_HOVER["deadband_w"],
        deadband_a=DEFAULT_HOVER["deadband_a"],
        acc_bias_alpha=DEFAULT_HOVER["acc_bias_alpha"],
        max_force=DEFAULT_HOVER["max_force"],
        max_torque=DEFAULT_HOVER["max_torque"],
    ) -> None:
        self.kd_w = np.asarray(kd_w, dtype=float)
        self.kp_a = np.asarray(kp_a, dtype=float)
        self.deadband_w = float(deadband_w)
        self.deadband_a = float(deadband_a)
        self.acc_bias_alpha = float(acc_bias_alpha)
        self.max_force = float(max_force)
        self.max_torque = float(max_torque)
        self._acc_bias = np.zeros(3)

    def compute(self, gyro, acc, feedforward_force=None, attitude_ref=None):
        """Return (force, torque) lists for the given IMU sample.

        ``gyro`` / ``acc`` are 3-element body-frame iterables. ``feedforward_force``
        (optional) is summed with the hover corrective force before clamping, so
        free-path motion can command translation on top of hover. ``attitude_ref``
        is reserved for a future pose reference and is currently unused.
        """
        w = np.asarray(gyro, dtype=float)
        a = np.asarray(acc, dtype=float)

        # Torque: damp angular rate toward zero (deadband to ignore noise).
        w_db = _deadband(w, self.deadband_w)
        torque = -self.kd_w * w_db

        # Force: oppose acceleration disturbance after removing EMA bias.
        self._acc_bias = (
            (1.0 - self.acc_bias_alpha) * self._acc_bias
            + self.acc_bias_alpha * a
        )
        a_res = _deadband(a - self._acc_bias, self.deadband_a)
        force = -self.kp_a * a_res

        # Feed-forward translation hook (for future free-path motion).
        if feedforward_force is not None:
            force = force + np.asarray(feedforward_force, dtype=float)

        force = np.clip(force, -self.max_force, self.max_force)
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        return force.tolist(), torque.tolist()


class PoseCorrector:
    """TF-pose-based hover correction (ROS-agnostic, testable).

    Consumes poses pulled from the TF tree by the caller, smooths them with a
    one-sided Gaussian-weighted window, and produces a low-gain corrective
    (force, torque) toward a hold target. The force is returned in the body
    frame so it can be summed with the IMU hover wrench. The correction is
    clamped independently so it never dominates the IMU law.

    **Liveness**: TF is a pull source -- a lookup keeps succeeding from the
    buffer after the publisher stops. Liveness is therefore judged by whether
    the transform's *stamp advances*, timed on the caller's monotonic clock.
    Stamps are only ever compared to other stamps: the stamp may be on a
    simulation clock unrelated to the caller's clock, and subtracting one from
    the other would be meaningless (there is no ``/clock`` in the simulator).

    Also owns the checkpoint-array interface for the future free-path flight
    program: a received checkpoint list overrides the hold target, and
    ``advance_checkpoint()`` steps through it.
    """

    def __init__(
        self,
        poll_rate=DEFAULT_TF["poll_rate"],
        smooth_window=DEFAULT_TF["smooth_window"],
        smooth_sigma=DEFAULT_TF["smooth_sigma"],
        timeout=DEFAULT_TF["timeout"],
        kp_pos=DEFAULT_TF["kp_pos"],
        kp_att=DEFAULT_TF["kp_att"],
        max_corr_force=DEFAULT_TF["max_corr_force"],
        max_corr_torque=DEFAULT_TF["max_corr_torque"],
    ) -> None:
        self.poll_rate = float(poll_rate)
        self.window = max(1, int(smooth_window))
        self.sigma = float(smooth_sigma)
        self.timeout = float(timeout)
        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kp_att = np.asarray(kp_att, dtype=float)
        self.max_corr_force = float(max_corr_force)
        self.max_corr_torque = float(max_corr_torque)

        self._buf = deque(maxlen=self.window)  # (pos, quat)
        self._last_ingest_t = None   # local monotonic time of last buffered sample
        self._last_stamp = None      # last TF stamp seen (TF's own clock)
        self._last_advance_t = None  # local monotonic time the stamp last advanced
        self._status = STATUS_MISSING
        self._hold_pos = None
        self._hold_quat = None
        self._checkpoints = []  # list of (pos, quat)
        self._cp_index = None

    @property
    def status(self) -> str:
        """Latest liveness verdict: ``ok`` / ``stale`` / ``missing``."""
        return self._status

    # --- liveness ----------------------------------------------------------

    def _classify(self, t, pose) -> bool:
        """Update the liveness verdict. Returns True when the pose is usable.

        ``t`` is the caller's monotonic time in seconds. ``pose`` is
        ``(pos, quat, stamp)`` or None.
        """
        if pose is None:
            self._status = STATUS_MISSING
            return False

        stamp = float(pose[2])
        if stamp == 0.0:
            # In ROS a zero stamp means "unset", not "time zero". tf2 hands one
            # back for the odd lookup while the listener is still filling in the
            # chain. Adopting it would reset the reference to zero, so drop it.
            self._status = STATUS_MISSING
            return False
        if self._last_stamp is None or stamp > self._last_stamp:
            # Normal case: the TF publisher is alive and stamping forward.
            self._last_stamp = stamp
            self._last_advance_t = t
            self._status = STATUS_OK
            return True
        if stamp < self._last_stamp:
            # Clock reset (simulator restarted). Adopt the new epoch rather
            # than reporting a stall that would never clear.
            self._last_stamp = stamp
            self._last_advance_t = t
            self._status = STATUS_OK
            return True

        # Stamp repeated: normal when TF is slower than the control loop, a
        # stopped publisher once it persists beyond the timeout.
        if self._last_advance_t is not None and (t - self._last_advance_t) > self.timeout:
            self._status = STATUS_STALE
            return False
        self._status = STATUS_OK
        return True

    # --- pose intake -------------------------------------------------------

    def _ingest(self, t, pos, quat) -> None:
        """Buffer one pose sample, time-gated to ``poll_rate``."""
        if (
            self._last_ingest_t is not None
            and (t - self._last_ingest_t) < 1.0 / self.poll_rate
        ):
            return
        self._last_ingest_t = t
        q = np.asarray(quat, dtype=float)
        # Sign-align with the previous sample so component-wise averaging of
        # the double-covered quaternion is well defined.
        if self._buf and float(np.dot(self._buf[-1][1], q)) < 0.0:
            q = -q
        self._buf.append((np.asarray(pos, dtype=float), q))

    def _drop(self) -> None:
        """Forget buffered poses and the hold target after a TF loss."""
        self._buf.clear()
        self._last_ingest_t = None
        if self._cp_index is None:
            self._hold_pos = None
            self._hold_quat = None

    def _smoothed(self):
        """Gaussian-weighted average of the buffer (latest sample heaviest)."""
        if self.window == 1 or len(self._buf) == 1:
            pos, quat = self._buf[-1]
            return pos, quat
        n = len(self._buf)
        idx = np.arange(n, dtype=float)
        w = np.exp(-((n - 1 - idx) ** 2) / (2.0 * self.sigma ** 2))
        w /= w.sum()
        pos = sum(wi * s[0] for wi, s in zip(w, self._buf))
        quat = sum(wi * s[1] for wi, s in zip(w, self._buf))
        quat = quat / np.linalg.norm(quat)
        return pos, quat

    # --- hold target / checkpoints -----------------------------------------

    def set_checkpoints(self, poses) -> None:
        """Replace the checkpoint array. ``poses`` is a list of (pos, quat).

        A non-empty list makes its first entry the active hold target; an
        empty list clears checkpoints and re-captures the hover pose.
        """
        self._checkpoints = [
            (np.asarray(p, dtype=float), np.asarray(q, dtype=float))
            for p, q in poses
        ]
        if self._checkpoints:
            self._cp_index = 0
        else:
            self._cp_index = None
            self._hold_pos = None  # re-capture from current smoothed pose
            self._hold_quat = None

    def advance_checkpoint(self) -> bool:
        """Switch to the next checkpoint. Returns False at the last one."""
        if self._cp_index is None:
            return False
        if self._cp_index + 1 >= len(self._checkpoints):
            return False
        self._cp_index += 1
        return True

    def active_target(self):
        """Current hold target (pos, quat) or (None, None)."""
        if self._cp_index is not None:
            return self._checkpoints[self._cp_index]
        return self._hold_pos, self._hold_quat

    # --- correction law -----------------------------------------------------

    def update(self, t, pose):
        """Ingest one polled pose and return (force_body, torque) lists.

        ``t`` is the caller's monotonic time in seconds; ``pose`` is
        ``(pos, quat, stamp)`` from the TF client, or None when the lookup
        failed. Returns zeros whenever the pose is unusable, so the caller
        degrades to pure IMU hover without special-casing.
        """
        zeros = ([0.0] * 3, [0.0] * 3)
        if not self._classify(t, pose):
            self._drop()
            return zeros

        self._ingest(t, pose[0], pose[1])
        if not self._buf:
            return zeros

        pos_s, quat_s = self._smoothed()
        target_pos, target_quat = self.active_target()
        if target_pos is None:
            # First valid pose after (re)acquisition becomes the hold target.
            self._hold_pos, self._hold_quat = pos_s, quat_s
            target_pos, target_quat = pos_s, quat_s

        # Position correction (reference frame) rotated into the body frame.
        # Proportional only: damping is the IMU law's job (no velocity from TF).
        f_ref = self.kp_pos * (target_pos - pos_s)
        f_body = _quat_rotate(_quat_conj(quat_s), f_ref)

        # Attitude correction from the quaternion error (body frame).
        qe = _quat_mul(_quat_conj(target_quat), quat_s)
        torque = -self.kp_att * np.sign(qe[3] if qe[3] != 0.0 else 1.0) * qe[:3]

        f_body = np.clip(f_body, -self.max_corr_force, self.max_corr_force)
        torque = np.clip(torque, -self.max_corr_torque, self.max_corr_torque)
        return f_body.tolist(), torque.tolist()


class HoverController:
    """DI orchestration logic: read IMU + TF, hover, allocate, publish.

    Args:
        imu_subscriber: injected ``ImuSubscriber`` (source of gyro/acc).
        fan_publisher: injected ``FanDutyPublisher`` (duty output).
        allocator: injected :class:`ThrustAllocator`.
        law: :class:`HoverLaw` instance (IMU control law).
        tf_client: optional ``TfClient``. None -> pure IMU hover; the node
            decides this from ``hover_control.mode`` and injects the result.
        corrector: optional :class:`PoseCorrector` (used when ``tf_client`` is
            given).
    """

    def __init__(self, imu_subscriber, fan_publisher, allocator, law,
                 tf_client=None, corrector=None) -> None:
        self._imu = imu_subscriber
        self._fan = fan_publisher
        self._allocator = allocator
        self._law = law
        self._tf = tf_client
        self._corrector = corrector if tf_client is not None else None

    @property
    def tf_status(self) -> str:
        """TF liveness for status logging: ``off`` when running IMU-only."""
        if self._corrector is None:
            return STATUS_OFF
        return self._corrector.status

    @staticmethod
    def declare_parameters(node) -> None:
        """Declare the hover-law and TF-correction control parameters."""
        for key, default in DEFAULT_HOVER.items():
            name = f"hover_control.{key}"
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
        for key, default in DEFAULT_TF.items():
            name = f"tf_correction.{key}"
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

    @classmethod
    def from_node(cls, node, imu_subscriber, fan_publisher, allocator,
                  tf_client=None) -> "HoverController":
        """Build the hover law and, when TF is injected, the pose corrector."""
        cls.declare_parameters(node)

        def h(key):
            return node.get_parameter(f"hover_control.{key}").value

        def f(key):
            return node.get_parameter(f"tf_correction.{key}").value

        law = HoverLaw(
            kd_w=h("kd_w"), kp_a=h("kp_a"),
            deadband_w=h("deadband_w"), deadband_a=h("deadband_a"),
            acc_bias_alpha=h("acc_bias_alpha"),
            max_force=h("max_force"), max_torque=h("max_torque"),
        )
        corrector = None
        if tf_client is not None:
            corrector = PoseCorrector(
                poll_rate=f("poll_rate"), smooth_window=f("smooth_window"),
                smooth_sigma=f("smooth_sigma"), timeout=f("timeout"),
                kp_pos=f("kp_pos"), kp_att=f("kp_att"),
                max_corr_force=f("max_corr_force"),
                max_corr_torque=f("max_corr_torque"),
            )
        return cls(imu_subscriber, fan_publisher, allocator, law,
                   tf_client, corrector)

    # --- checkpoint hooks (delegated to the corrector) ---------------------

    def set_checkpoints(self, poses) -> None:
        """Set the checkpoint hold-target array (no-op in IMU-only mode)."""
        if self._corrector is not None:
            self._corrector.set_checkpoints(poses)

    def advance_checkpoint(self) -> bool:
        """Step the hold target to the next checkpoint (free-path hook)."""
        if self._corrector is None:
            return False
        return self._corrector.advance_checkpoint()

    # --- control tick ------------------------------------------------------

    def step(self, t=None, feedforward=None) -> None:
        """One control tick: read IMU (+TF), compute wrench, allocate, publish."""
        gyro, acc = self._imu.gyro, self._imu.acc
        if gyro is None or acc is None:
            self._fan.set_duty_array([])  # no IMU yet -> idle
            return
        force, torque = self._law.compute(gyro, acc, feedforward_force=feedforward)
        if self._corrector is not None:
            if t is None:
                t = time.monotonic()
            f_corr, t_corr = self._corrector.update(t, self._tf.get_pose())
            force = np.clip(
                np.asarray(force) + f_corr,
                -self._law.max_force, self._law.max_force,
            ).tolist()
            torque = np.clip(
                np.asarray(torque) + t_corr,
                -self._law.max_torque, self._law.max_torque,
            ).tolist()
        duties = self._allocator.allocate(force, torque)
        self._fan.set_duty_array(duties)
