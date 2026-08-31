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
from sobits_intball2_gnc.control.utils.quat_math import (
    geodesic_angle,
    quat_conj,
    quat_mul,
)

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
    # kp_att/kd_att are split into an align variant (used while a checkpoint
    # set via the external /gnc/checkpoints path -- pre_align/align_at_arrival
    # -- has not yet converged) and a hold variant (used once converged, and
    # for any internally re-captured checkpoint that was never "align" in the
    # first place). See docs/archive/achieved/2026-08-21_tf_correction_align_hold_gain_split_design.md.
    # Both default to the same placeholder value; align/hold divergence is a
    # future tuning step, not part of the split itself.
    "kp_att_align": [0.01, 0.01, 0.01],
    "kp_att_hold": [0.01, 0.01, 0.01],
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
    "kd_att_align": [0.0, 0.0, 0.0],
    "kd_att_hold": [0.0, 0.0, 0.0],
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
    # Off by default (matches prior per-axis-independent clamp behavior).
    # See attitude_error_to_torque's preserve_direction docstring and
    # docs/2026-08-27_align_hold_gain_oscillation_investigation.md -- flip
    # live via `ros2 param set` to A/B test without a node restart (Category
    # A dynamic parameter, see TF_CORRECTION_DYNAMIC_KEYS in control.py).
    "torque_direction_preserving": False,
    "checkpoint_topic": "/gnc/checkpoints",
    # Below: align/hold gain-switch state machine, see
    # docs/archive/achieved/2026-08-21_tf_correction_align_hold_gain_split_design.md. Only
    # meaningful for checkpoints set with is_align=True (the external
    # /gnc/checkpoints path, i.e. GuidanceExecutor's pre_align/
    # align_at_arrival). Independent of guidance's own
    # guidance.align_tolerance_deg/align_settle_time -- this only decides
    # which gain PoseCorrector applies, not whether GuidanceExecutor reports
    # the align as done, so the two need not match exactly.
    "align_tolerance_deg": 3.0,
    # Minimum unbroken time within align_tolerance_deg before switching to
    # the hold gain, mirroring guidance.align_settle_time's hysteresis (see
    # docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md --
    # a single noisy near-miss sample must not trigger the switch early).
    "align_settle_time": 0.5,
    # Safety-net upper bound on how long the align gain is used for one
    # checkpoint, in case TF noise/loss prevents the angle check above from
    # ever reporting convergence. Deliberately generous -- this is a backstop,
    # not the primary switch trigger.
    "align_gain_max_duration": 30.0,
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
        kp_att_align=DEFAULT_TF["kp_att_align"],
        kp_att_hold=DEFAULT_TF["kp_att_hold"],
        kd_pos=DEFAULT_TF["kd_pos"],
        vel_filter_alpha=DEFAULT_TF["vel_filter_alpha"],
        kd_att_align=DEFAULT_TF["kd_att_align"],
        kd_att_hold=DEFAULT_TF["kd_att_hold"],
        att_filter_alpha=DEFAULT_TF["att_filter_alpha"],
        max_corr_force=DEFAULT_TF["max_corr_force"],
        max_corr_torque=DEFAULT_TF["max_corr_torque"],
        torque_direction_preserving=DEFAULT_TF["torque_direction_preserving"],
        align_tolerance_deg=DEFAULT_TF["align_tolerance_deg"],
        align_settle_time=DEFAULT_TF["align_settle_time"],
        align_gain_max_duration=DEFAULT_TF["align_gain_max_duration"],
    ) -> None:
        self.poll_rate = float(poll_rate)
        self.window = max(1, int(smooth_window))
        self.sigma = float(smooth_sigma)
        self.timeout = float(timeout)
        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kp_att_align = np.asarray(kp_att_align, dtype=float)
        self.kp_att_hold = np.asarray(kp_att_hold, dtype=float)
        self.kd_pos = np.asarray(kd_pos, dtype=float)
        self.kd_att_align = np.asarray(kd_att_align, dtype=float)
        self.kd_att_hold = np.asarray(kd_att_hold, dtype=float)
        self.vel_filter_alpha = float(vel_filter_alpha)
        self.att_filter_alpha = float(att_filter_alpha)
        self.align_tolerance_rad = np.radians(float(align_tolerance_deg))
        self.align_settle_time = float(align_settle_time)
        self.align_gain_max_duration = float(align_gain_max_duration)
        self.max_corr_force = float(max_corr_force)
        self.max_corr_torque = float(max_corr_torque)
        self.torque_direction_preserving = bool(torque_direction_preserving)

        self._buf = deque(maxlen=self.window)  # (pos, quat)
        self._last_ingest_t = None   # local monotonic time of last buffered sample
        self._last_stamp = None      # last TF stamp seen (TF's own clock)
        self._last_advance_t = None  # local monotonic time the stamp last advanced
        self._status = STATUS_MISSING
        self._hold_pos = None
        self._hold_quat = None
        self._checkpoints = []  # list of (pos, quat)
        self._cp_index = None
        # Bumped on every set_checkpoints() call (including internal
        # re-captures) so callers like HoverController can tell whether
        # someone has (re)targeted the hold since a point they recorded --
        # see the trajectory-end race note on set_checkpoints() below.
        self._checkpoint_version = 0
        # Whether the currently active checkpoint was set with is_align=True
        # (see set_checkpoints()).
        self._is_align = False
        # Align/hold gain-switch state, re-derived from _is_align each time
        # _checkpoint_version changes (see the version check at the top of
        # update()). _align_active True -> use the align gain this tick.
        self._gain_state_version = None
        self._align_active = False
        self._align_start_t = None
        self._within_tolerance_since = None
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

    def set_checkpoints(self, poses, is_align=False) -> None:
        """Replace the checkpoint array. ``poses`` is a list of (pos, quat).

        A non-empty list makes its first entry the active hold target; an
        empty list clears checkpoints and re-captures the hover pose.

        ``is_align``: pass True when this checkpoint is a one-shot align
        target (``GuidanceExecutor``'s pre_align/align_at_arrival, via the
        external ``/gnc/checkpoints`` path) so the align gain is used until
        TF reports convergence (or the safety-net duration elapses); see
        ``update()``. Leave False for hold targets, including re-captures --
        an empty ``poses`` list is always treated as a hold target regardless
        of the argument, since there is nothing to align to.

        Every call bumps :attr:`checkpoint_version`, including internal ones
        (e.g. ``HoverController``'s own re-capture-on-trajectory-end) -- see
        that call site for why the version needs to reflect *all* callers.
        The align/hold gain state is (re-)derived lazily from ``is_align`` on
        the next ``update()`` call, keyed off that same version bump.
        """
        self._checkpoints = [
            (np.asarray(p, dtype=float), np.asarray(q, dtype=float))
            for p, q in poses
        ]
        if self._checkpoints:
            self._cp_index = 0
            self._is_align = bool(is_align)
        else:
            self._cp_index = None
            self._hold_pos = None  # re-capture from current smoothed pose
            self._hold_quat = None
            self._is_align = False
        self._checkpoint_version += 1

    @property
    def checkpoint_version(self) -> int:
        """Bumped by every :meth:`set_checkpoints` call, self or caller."""
        return self._checkpoint_version

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

        # Align/hold gain switch (see set_checkpoints()'s is_align and
        # docs/archive/achieved/2026-08-21_tf_correction_align_hold_gain_split_design.md).
        # Re-derived whenever the checkpoint changed since the last tick;
        # reset here (not in set_checkpoints()) because only update() has a
        # timestamp to start the align clock from.
        if self._gain_state_version != self._checkpoint_version:
            self._gain_state_version = self._checkpoint_version
            self._align_active = self._is_align
            self._align_start_t = t
            self._within_tolerance_since = None
        if self._align_active:
            angle = geodesic_angle(target_quat, quat_s)
            if angle <= self.align_tolerance_rad:
                if self._within_tolerance_since is None:
                    self._within_tolerance_since = t
                elif (t - self._within_tolerance_since) >= self.align_settle_time:
                    self._align_active = False
            else:
                self._within_tolerance_since = None
            # Safety net: TF noise/loss could keep the angle check above from
            # ever reporting convergence -- don't stay on the align gain
            # forever regardless.
            if self._align_active and (t - self._align_start_t) >= self.align_gain_max_duration:
                self._align_active = False
        kp_att, kd_att = (
            (self.kp_att_align, self.kd_att_align) if self._align_active
            else (self.kp_att_hold, self.kd_att_hold)
        )

        torque = attitude_error_to_torque(
            kp_att, kd_att, target_quat, quat_s, self._omega_filtered,
            self.max_corr_torque,
            preserve_direction=self.torque_direction_preserving,
        )
        return f_body.tolist(), torque.tolist()

    def set_gains(self, kp_pos=None, kd_pos=None,
                  kp_att_align=None, kd_att_align=None,
                  kp_att_hold=None, kd_att_hold=None,
                  vel_filter_alpha=None, att_filter_alpha=None,
                  max_corr_force=None, max_corr_torque=None,
                  torque_direction_preserving=None,
                  align_tolerance_deg=None, align_settle_time=None,
                  align_gain_max_duration=None,
                  timeout=None) -> None:
        """Update gains/thresholds in place (dynamic reconfiguration).

        Any argument left as ``None`` keeps its current value. Does not touch
        ``poll_rate``/``smooth_window``/``smooth_sigma`` (loop timing / buffer
        sizing, see docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md
        category C) or any liveness/hold-target state.
        """
        if kp_pos is not None:
            self.kp_pos = np.asarray(kp_pos, dtype=float)
        if kd_pos is not None:
            self.kd_pos = np.asarray(kd_pos, dtype=float)
        if kp_att_align is not None:
            self.kp_att_align = np.asarray(kp_att_align, dtype=float)
        if kd_att_align is not None:
            self.kd_att_align = np.asarray(kd_att_align, dtype=float)
        if kp_att_hold is not None:
            self.kp_att_hold = np.asarray(kp_att_hold, dtype=float)
        if kd_att_hold is not None:
            self.kd_att_hold = np.asarray(kd_att_hold, dtype=float)
        if vel_filter_alpha is not None:
            self.vel_filter_alpha = float(vel_filter_alpha)
        if att_filter_alpha is not None:
            self.att_filter_alpha = float(att_filter_alpha)
        if align_tolerance_deg is not None:
            self.align_tolerance_rad = np.radians(float(align_tolerance_deg))
        if align_settle_time is not None:
            self.align_settle_time = float(align_settle_time)
        if align_gain_max_duration is not None:
            self.align_gain_max_duration = float(align_gain_max_duration)
        if max_corr_force is not None:
            self.max_corr_force = float(max_corr_force)
        if max_corr_torque is not None:
            self.max_corr_torque = float(max_corr_torque)
        if torque_direction_preserving is not None:
            self.torque_direction_preserving = bool(torque_direction_preserving)
        if timeout is not None:
            self.timeout = float(timeout)
