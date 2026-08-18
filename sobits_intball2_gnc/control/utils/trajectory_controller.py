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
"""
import numpy as np

from sobits_intball2_gnc.control.utils.pose_control_law import position_error_to_force
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_rotate

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
    """

    def __init__(
        self,
        mass=DEFAULT_TRAJECTORY["mass"],
        kp_pos=DEFAULT_TRAJECTORY["kp_pos"],
        kd_pos=DEFAULT_TRAJECTORY["kd_pos"],
        vel_filter_alpha=DEFAULT_TRAJECTORY["vel_filter_alpha"],
        max_force=DEFAULT_TRAJECTORY["max_force"],
    ) -> None:
        self.mass = float(mass)
        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kd_pos = np.asarray(kd_pos, dtype=float)
        self.vel_filter_alpha = float(vel_filter_alpha)
        self.max_force = float(max_force)

        # Velocity estimate: finite difference of the TF position between
        # successive compute() calls, independent of PoseCorrector's own
        # estimate (see module docstring).
        self._last_pos = None
        self._last_t = None
        self._vel_filtered = np.zeros(3)

    def reset(self) -> None:
        """Forget the velocity estimate (e.g. after a setpoint gap)."""
        self._last_pos = None
        self._last_t = None
        self._vel_filtered = np.zeros(3)

    def compute(self, t, pos_now, quat_now, p_des, v_des, a_des):
        """Return a clamped body-frame force toward the moving setpoint.

        ``t`` is the caller's monotonic time in seconds. ``pos_now``/
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
            dt = t - self._last_t
            if dt > 1e-6:
                vel_now = (pos_now - self._last_pos) / dt
        self._last_pos, self._last_t = pos_now, t

        self._vel_filtered = (
            self.vel_filter_alpha * vel_now
            + (1.0 - self.vel_filter_alpha) * self._vel_filtered
        )

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
