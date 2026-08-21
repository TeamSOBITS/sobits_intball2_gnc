#!/usr/bin/env python3
"""Moving-target translation controller (ROS-agnostic, testable).

Follows a Guidance-published trajectory setpoint (p_des, v_des, a_des,
reference frame) with a feedforward + feedback force law:

    F_des = m * a_des + Kp * (p_des - p_now) - Kd * (v_now - v_des)

The feedback term reuses
:func:`~sobits_intball2_gnc.control.utils.pose_control_law.position_error_to_force`
(same P+D math `PoseCorrector` uses for checkpoint holding) so the two modes
share a single, already-validated implementation of "position/velocity error
-> body-frame force". See ``openspec/changes/add-trajectory-following`` and
``docs/phase3.md`` for the full interface contract this implements
(mutual exclusivity with ``PoseCorrector``'s checkpoint hold is owned by the
caller, e.g. :class:`HoverController` -- this class only computes a force
given a setpoint and the current state, with no notion of "is this setpoint
live").

Velocity is estimated independently of ``PoseCorrector`` (TF position finite
difference + EMA low-pass, same technique as ``PoseCorrector.kd_pos`` uses)
because ``PoseCorrector``'s own estimate tracks velocity relative to its
checkpoint hold target, not relative to a moving trajectory setpoint.

Phase 3b adds a parallel attitude path (:meth:`compute_attitude`): while a
trajectory setpoint is live, the desired orientation is Guidance's ``q_des(t)``
rather than ``PoseCorrector``'s static hold/checkpoint quaternion. It reuses
:func:`~sobits_intball2_gnc.control.utils.pose_control_law.attitude_error_to_torque`
and estimates its own relative tracking-rate (``omega_err``) the same way
``PoseCorrector`` does for its checkpoint attitude loop, for the same reason
the translation velocity estimate above is kept independent: the target is
moving, so an estimate relative to a static hold target would be wrong.
"""
import numpy as np

from sobits_intball2_gnc.control.utils.pose_control_law import (
    attitude_error_to_torque,
    position_error_to_force,
)
from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj,
    quat_mul,
    quat_rotate,
)

DEFAULT_TRAJECTORY = {
    "mass": 4.5,
    "max_force": 0.1,
    "timeout": 0.2,
    # Reuses the same theoretically-designed position gains as the checkpoint
    # hold loop (tf_correction.kp_pos/kd_pos, docs/phase0_findings.md
    # observation 7 + attendant 2nd-order design): the underlying mass-
    # spring-damper physics of "drive position/velocity error to zero" is the
    # same whether the target is static or moving.
    "kp_pos": [0.89, 0.89, 0.89],
    "kd_pos": [3.6, 3.6, 3.6],
    # EMA low-pass on this controller's own finite-difference velocity
    # estimate, same technique and default as PoseCorrector.vel_filter_alpha.
    "vel_filter_alpha": 0.3,
    # Attitude gains, seeded from tf_correction.kp_att/kd_att (Phase 3b): the
    # checkpoint-hold gains are the only ones this vehicle has been tuned
    # against so far. Re-tuning against moving-setpoint tracking error is
    # deferred (docs/main_plan.md Phase 3b "追加の宿題").
    "kp_att": [0.01, 0.01, 0.01],
    "kd_att": [0.0, 0.0, 0.0],
    "att_filter_alpha": 1.0,
    "max_torque": 0.01,
}


class TrajectoryController:
    """Feedforward + feedback translation controller for a moving setpoint.

    Args:
        mass: Vehicle mass [kg] for the feedforward term.
        kp_pos: Position-error -> force gain [N/m], 3 elements.
        kd_pos: Velocity-error -> force gain [N/(m/s)], 3 elements.
        vel_filter_alpha: EMA blend weight for this controller's own
            finite-difference velocity estimate (1.0 = no filtering).
        max_force: Output force clamp (per axis) [N].
        kp_att: Quaternion-error -> torque gain, 3 elements.
        kd_att: Relative tracking-rate -> torque gain, 3 elements.
        att_filter_alpha: EMA blend weight for this controller's own
            finite-difference omega_err estimate (1.0 = no filtering).
        max_torque: Output torque clamp (per axis) [N*m].
    """

    def __init__(
        self,
        mass=DEFAULT_TRAJECTORY["mass"],
        kp_pos=DEFAULT_TRAJECTORY["kp_pos"],
        kd_pos=DEFAULT_TRAJECTORY["kd_pos"],
        vel_filter_alpha=DEFAULT_TRAJECTORY["vel_filter_alpha"],
        max_force=DEFAULT_TRAJECTORY["max_force"],
        kp_att=DEFAULT_TRAJECTORY["kp_att"],
        kd_att=DEFAULT_TRAJECTORY["kd_att"],
        att_filter_alpha=DEFAULT_TRAJECTORY["att_filter_alpha"],
        max_torque=DEFAULT_TRAJECTORY["max_torque"],
    ) -> None:
        self.mass = float(mass)
        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kd_pos = np.asarray(kd_pos, dtype=float)
        self.vel_filter_alpha = float(vel_filter_alpha)
        self.max_force = float(max_force)
        self.kp_att = np.asarray(kp_att, dtype=float)
        self.kd_att = np.asarray(kd_att, dtype=float)
        self.att_filter_alpha = float(att_filter_alpha)
        self.max_torque = float(max_torque)

        # Velocity estimate: finite difference of the TF position between
        # successive compute() calls, independent of PoseCorrector's own
        # estimate (see module docstring).
        self._last_pos = None
        self._last_t = None
        self._vel_filtered = np.zeros(3)

        # Relative angular-rate estimate for kd_att: finite difference of the
        # quaternion error's vector part between successive compute_attitude()
        # calls, independent of PoseCorrector's own estimate (see module
        # docstring) since the target here is moving.
        self._last_qe_vec = None
        self._last_att_t = None
        self._omega_filtered = np.zeros(3)

    def reset(self) -> None:
        """Forget the velocity/rate estimates (e.g. after a setpoint gap)."""
        self._last_pos = None
        self._last_t = None
        self._vel_filtered = np.zeros(3)
        self._last_qe_vec = None
        self._last_att_t = None
        self._omega_filtered = np.zeros(3)

    def compute(self, stamp, pos_now, quat_now, p_des, v_des, a_des):
        """Return a clamped body-frame force toward the moving setpoint.

        ``stamp`` is the TF pose's own timestamp (seconds), NOT the caller's
        wall-clock time: velocity is a finite difference of ``pos_now``, which
        is itself timestamped by the TF publisher, so the ``dt`` used to
        divide it must come from that same clock. Using wall-clock ``dt``
        instead diverges from the TF-clock position delta whenever the two
        clocks drift apart under scheduling delay (e.g. CPU contention
        stalling this process while TF delivery bursts once it catches up),
        producing spurious velocity spikes -- see
        docs/recording_cpu_load_control_degradation.md. ``pos_now``/
        ``quat_now`` are the current TF pose (reference frame position,
        reference-frame -> body-frame orientation). ``p_des``/``v_des``/
        ``a_des`` are the setpoint (reference-frame position/velocity/
        acceleration).
        """
        pos_now = np.asarray(pos_now, dtype=float)
        quat_now = np.asarray(quat_now, dtype=float)
        p_des = np.asarray(p_des, dtype=float)
        v_des = np.asarray(v_des, dtype=float)
        a_des = np.asarray(a_des, dtype=float)

        vel_now = np.zeros(3)
        if self._last_pos is not None:
            dt = stamp - self._last_t
            if dt > 1e-6:
                vel_now = (pos_now - self._last_pos) / dt
        self._last_pos, self._last_t = pos_now, stamp

        self._vel_filtered = (
            self.vel_filter_alpha * vel_now
            + (1.0 - self.vel_filter_alpha) * self._vel_filtered
        )

        # TEMPORARY debug instrumentation (docs/recording_cpu_load_control_degradation.md):
        # confirming the "reference races ahead of sim-time capacity" hypothesis --
        # dumping wall-clock time alongside the TF stamp (to derive local RTF) and
        # the position error (p_des - pos_now), to check whether error growth
        # correlates with RTF drops rather than with the (now stamp-based, fixed)
        # velocity estimate. Remove after that investigation concludes.
        try:
            import time as _time
            with open("/tmp/trajectory_reference_race_timing.log", "a") as _dbgf:
                _err = p_des - pos_now
                _dbgf.write(
                    f"{_time.monotonic():.6f},{stamp:.6f},"
                    f"{pos_now[0]:.6f},{pos_now[1]:.6f},{pos_now[2]:.6f},"
                    f"{p_des[0]:.6f},{p_des[1]:.6f},{p_des[2]:.6f},"
                    f"{_err[0]:.6f},{_err[1]:.6f},{_err[2]:.6f}\n"
                )
        except Exception:
            pass

        # position_error_to_force computes `-kd_pos * vel`; passing the
        # velocity ERROR (now - desired) here yields the intended
        # `+kd_pos * (v_des - v_now)` damping-toward-the-setpoint term. Clamp
        # is left effectively open here (np.inf) so the feedforward term can
        # be added before the single final clamp below -- reusing this
        # function purely for its P+D-in-body-frame math, not its clamp.
        vel_err_for_kd = self._vel_filtered - v_des
        feedback_body = position_error_to_force(
            self.kp_pos, self.kd_pos, p_des, pos_now, vel_err_for_kd,
            quat_now, max_force=np.inf,
        )

        feedforward_ref = self.mass * a_des
        feedforward_body = quat_rotate(quat_conj(quat_now), feedforward_ref)

        force_body = feedback_body + feedforward_body
        return np.clip(force_body, -self.max_force, self.max_force).tolist()

    def compute_attitude(self, stamp, quat_now, q_des):
        """Return a clamped body-frame torque toward the moving ``q_des(t)``.

        ``stamp`` is the TF pose's own timestamp (seconds), for the same
        reason :meth:`compute` takes a TF stamp rather than wall-clock time:
        ``omega_err`` is a finite difference of ``quat_now``-derived
        ``qe_vec``, so its ``dt`` must be measured on the same clock that
        stamped ``quat_now``. ``quat_now`` is the current TF orientation
        (reference frame -> body frame). ``q_des`` is Guidance's desired
        orientation (reference frame), held fixed by
        :func:`~sobits_intball2_gnc.guidance.utils.attitude_reference.compute_q_des`
        below its speed threshold -- ``omega_err`` settles to zero once
        tracking is locked, same steady-state behavior as
        :class:`~sobits_intball2_gnc.control.utils.pose_corrector.PoseCorrector`'s
        checkpoint attitude hold.
        """
        quat_now = np.asarray(quat_now, dtype=float)
        q_des = np.asarray(q_des, dtype=float)

        qe = quat_mul(quat_conj(q_des), quat_now)
        sign = np.sign(qe[3] if qe[3] != 0.0 else 1.0)
        qe_vec = sign * qe[:3]

        omega_err = np.zeros(3)
        if self._last_qe_vec is not None:
            dt = stamp - self._last_att_t
            if dt > 1e-6:
                omega_err = (qe_vec - self._last_qe_vec) / dt
        self._last_qe_vec, self._last_att_t = qe_vec, stamp

        self._omega_filtered = (
            self.att_filter_alpha * omega_err
            + (1.0 - self.att_filter_alpha) * self._omega_filtered
        )

        torque = attitude_error_to_torque(
            self.kp_att, self.kd_att, q_des, quat_now, self._omega_filtered,
            self.max_torque,
        )
        return torque.tolist()

    def set_gains(self, kp_pos=None, kd_pos=None, vel_filter_alpha=None,
                  max_force=None, kp_att=None, kd_att=None,
                  att_filter_alpha=None, max_torque=None) -> None:
        """Update gains/clamps in place (dynamic reconfiguration).

        Any argument left as ``None`` keeps its current value. Does not touch
        ``mass`` (measured physical constant, see
        docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md category C) or the
        velocity/rate finite-difference state (``reset()`` clears that
        separately, e.g. on a new trajectory).
        """
        if kp_pos is not None:
            self.kp_pos = np.asarray(kp_pos, dtype=float)
        if kd_pos is not None:
            self.kd_pos = np.asarray(kd_pos, dtype=float)
        if vel_filter_alpha is not None:
            self.vel_filter_alpha = float(vel_filter_alpha)
        if max_force is not None:
            self.max_force = float(max_force)
        if kp_att is not None:
            self.kp_att = np.asarray(kp_att, dtype=float)
        if kd_att is not None:
            self.kd_att = np.asarray(kd_att, dtype=float)
        if att_filter_alpha is not None:
            self.att_filter_alpha = float(att_filter_alpha)
        if max_torque is not None:
            self.max_torque = float(max_torque)
