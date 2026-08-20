#!/usr/bin/env python3
"""TF-pose-based hover correction (ROS-agnostic, testable).

See :class:`PoseCorrector` for details. The error-to-wrench math itself lives
in :mod:`sobits_intball2_gnc.control.utils.pose_control_law` so it can be
reused by a future moving-target controller; this module owns TF ingestion,
smoothing, liveness classification, and hold-target/checkpoint bookkeeping.
"""
from collections import deque

import numpy as np

from sobits_intball2_gnc.control.utils.pose_control_law import (
    attitude_error_to_torque,
    position_error_to_force,
)
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul

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
    # Position-loop damping (force per m/s of estimated hold-target velocity).
    # Defaults to zero (pure P control, matching prior behavior); Phase 0
    # found the position loop has no damping in practice (HoverLaw's kp_a
    # term stays near zero, deadbanded away) and oscillates. See
    # docs/phase0_findings.md. Non-zero only in gnc_params.yaml as a
    # Phase 0 experiment -- untuned.
    "kd_pos": [0.0, 0.0, 0.0],
    # EMA low-pass on the finite-difference velocity fed to kd_pos. 1.0 means
    # no filtering (use the raw finite difference, matching prior behavior).
    # Phase 0 found the raw finite difference too noisy to reach JAXA-level
    # (mm) hold precision; see docs/phase0_findings.md. Lower values trade
    # noise rejection for phase lag.
    "vel_filter_alpha": 1.0,
    # Attitude-loop damping (torque per rad/s of *relative* tracking-error
    # rate, estimated from the finite difference of the quaternion error --
    # NOT the IMU's absolute angular rate). Defaults to zero, matching prior
    # behavior (pure P attitude control). See docs/phase0_5_findings.md: using
    # HoverLaw's kd_w (an absolute-rate damper) as the attitude D term made
    # hold accuracy *worse*, because iss_body itself rotates in the world
    # (docs/phase0_findings.md observations 8/11) -- holding a fixed relative
    # attitude requires a non-zero absolute rate, so damping the absolute
    # rate fights the needed tracking motion. Damping the relative error
    # rate instead vanishes once tracking is locked, regardless of how fast
    # the reference frame itself turns.
    "kd_att": [0.0, 0.0, 0.0],
    # EMA low-pass on the finite-difference omega_err fed to kd_att, mirroring
    # vel_filter_alpha above. 1.0 means no filtering. docs/phase0_5_findings.md
    # observation B: the theoretically-derived kp_att/kd_att underperformed
    # the original guessed kp_att (measured by quaternion geodesic angle, not
    # RPY), and the raw finite-difference qe rate is a likely noise source --
    # same failure mode Phase 0 found for kd_pos's raw velocity (docs/
    # phase0_findings.md), fixed there by vel_filter_alpha. Untried here yet.
    "att_filter_alpha": 1.0,
    "max_corr_force": 0.05,
    "max_corr_torque": 0.01,
    "checkpoint_topic": "/gnc/checkpoints",
}

# PoseCorrector.status values.
STATUS_OFF = "off"          # no TF client injected (imu mode)
STATUS_MISSING = "missing"  # lookup itself fails
STATUS_STALE = "stale"      # lookup succeeds but the stamp stopped advancing
STATUS_OK = "ok"


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
        kd_pos=DEFAULT_TF["kd_pos"],
        vel_filter_alpha=DEFAULT_TF["vel_filter_alpha"],
        kd_att=DEFAULT_TF["kd_att"],
        att_filter_alpha=DEFAULT_TF["att_filter_alpha"],
        max_corr_force=DEFAULT_TF["max_corr_force"],
        max_corr_torque=DEFAULT_TF["max_corr_torque"],
    ) -> None:
        self.poll_rate = float(poll_rate)
        self.window = max(1, int(smooth_window))
        self.sigma = float(smooth_sigma)
        self.timeout = float(timeout)
        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kp_att = np.asarray(kp_att, dtype=float)
        self.kd_pos = np.asarray(kd_pos, dtype=float)
        self.kd_att = np.asarray(kd_att, dtype=float)
        self.vel_filter_alpha = float(vel_filter_alpha)
        self.att_filter_alpha = float(att_filter_alpha)
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
        # Velocity estimate for kd_pos: finite difference of the smoothed
        # position between successive update() calls (Phase 0 damping trial).
        # Timed on the TF stamp (not the caller's wall-clock t), since the
        # position delta itself comes from TF-stamped samples -- mixing a
        # wall-clock dt with a TF-clock position delta produces spurious
        # velocity spikes when the two clocks drift apart under scheduling
        # delay. See docs/recording_cpu_load_control_degradation.md.
        self._last_smoothed_pos = None
        self._last_smoothed_stamp = None
        # EMA state for the filtered velocity (see vel_filter_alpha). Starts
        # at zero, matching the raw finite difference's own zero on the first
        # sample after (re)acquisition (no prior sample to difference against).
        self._vel_filtered = np.zeros(3)
        # Relative angular-rate estimate for kd_att: finite difference of the
        # quaternion error's vector part between successive update() calls
        # (shares the same dt as the position velocity estimate above).
        self._last_qe_vec = None
        # EMA state for the filtered omega_err (see att_filter_alpha), mirroring
        # _vel_filtered above.
        self._omega_filtered = np.zeros(3)

    @staticmethod
    def _qe_vec(target_quat, quat):
        """Vector part of the quaternion error (see attitude_error_to_torque)."""
        qe = quat_mul(quat_conj(target_quat), quat)
        sign = np.sign(qe[3] if qe[3] != 0.0 else 1.0)
        return sign * qe[:3]

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
        # Also forget the velocity estimate: the gap left by the TF loss
        # would otherwise be read back as a large, spurious velocity.
        self._last_smoothed_pos = None
        self._last_smoothed_stamp = None
        self._vel_filtered = np.zeros(3)
        self._last_qe_vec = None
        self._omega_filtered = np.zeros(3)
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

        ``t`` is the caller's monotonic time in seconds, used for liveness
        classification and poll-rate gating only. ``pose`` is
        ``(pos, quat, stamp)`` from the TF client, or None when the lookup
        failed; the velocity/omega finite differences are timed on ``stamp``
        instead of ``t`` since the position/quaternion deltas themselves are
        TF-stamped (see the ``_last_smoothed_stamp`` comment in ``__init__``).
        Returns zeros whenever the pose is unusable, so the caller degrades
        to pure IMU hover without special-casing.
        """
        zeros = ([0.0] * 3, [0.0] * 3)
        if not self._classify(t, pose):
            self._drop()
            return zeros

        stamp = float(pose[2])
        self._ingest(t, pose[0], pose[1])
        if not self._buf:
            return zeros

        pos_s, quat_s = self._smoothed()
        target_pos, target_quat = self.active_target()
        if target_pos is None:
            # First valid pose after (re)acquisition becomes the hold target.
            self._hold_pos, self._hold_quat = pos_s, quat_s
            target_pos, target_quat = pos_s, quat_s

        # Velocity estimate for kd_pos: finite difference of the smoothed
        # position since the last update() call, timed on the TF stamp (not
        # the caller's wall-clock t) since pos_s itself is TF-stamped -- see
        # the _last_smoothed_stamp comment in __init__. Zero on the first
        # sample (or right after a TF loss, see _drop()) since there is no
        # prior sample to difference against, and negative (stamp went
        # backwards, e.g. a simulator restart adopted by _classify) since
        # that is not a real elapsed time.
        vel = np.zeros(3)
        dt = 0.0
        if self._last_smoothed_pos is not None:
            dt = stamp - self._last_smoothed_stamp
            if dt > 1e-6:
                vel = (pos_s - self._last_smoothed_pos) / dt
        self._last_smoothed_pos, self._last_smoothed_stamp = pos_s, stamp

        # EMA low-pass on the raw finite-difference velocity (vel_filter_alpha
        # defaults to 1.0 = no filtering, matching prior behavior).
        self._vel_filtered = (
            self.vel_filter_alpha * vel
            + (1.0 - self.vel_filter_alpha) * self._vel_filtered
        )

        # Proportional + damping: kd_pos defaults to zero (see DEFAULT_TF) --
        # Phase 0 found the position loop had no damping in practice (the
        # IMU law's kp_a term stays deadbanded near zero, so it wasn't
        # actually damping this loop as originally assumed).
        f_body = position_error_to_force(
            self.kp_pos, self.kd_pos, target_pos, pos_s, self._vel_filtered,
            quat_s, self.max_corr_force,
        )

        # Relative angular-rate estimate for kd_att: finite difference of the
        # quaternion error's vector part since the last update() call (same
        # dt as the position velocity estimate above). Zero on the first
        # sample (or right after a TF loss, see _drop()).
        qe_vec = self._qe_vec(target_quat, quat_s)
        omega_err = np.zeros(3)
        if self._last_qe_vec is not None and dt > 1e-6:
            omega_err = (qe_vec - self._last_qe_vec) / dt
        self._last_qe_vec = qe_vec

        # EMA low-pass on the raw finite-difference omega_err (att_filter_alpha
        # defaults to 1.0 = no filtering, matching prior behavior). Mirrors
        # vel_filter_alpha above.
        self._omega_filtered = (
            self.att_filter_alpha * omega_err
            + (1.0 - self.att_filter_alpha) * self._omega_filtered
        )

        torque = attitude_error_to_torque(
            self.kp_att, self.kd_att, target_quat, quat_s, self._omega_filtered,
            self.max_corr_torque,
        )
        return f_body.tolist(), torque.tolist()
