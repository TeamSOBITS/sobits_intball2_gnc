#!/usr/bin/env python3
"""IMU-based hover control orchestration for IntBall2 (ROS-agnostic).

Holds a stable hover using IMU feedback: the angular rate (gyro) is damped
toward zero and the linear-acceleration disturbance (accelerometer, with an EMA
bias estimate removed) is opposed. In ``tf_imu`` mode a low-gain, independently
clamped position/attitude correction derived from the TF pose (``iss_body`` <-
``body``) is layered on top; losing TF degrades gracefully back to pure IMU
hover.

Layers (split across sibling modules; re-exported here for backward
compatibility):
- :class:`~sobits_intball2_gnc.control.utils.hover_law.HoverLaw` -- pure IMU
  control law (plain-value constructor, testable).
- :class:`~sobits_intball2_gnc.control.utils.pose_corrector.PoseCorrector` --
  pure pose correction + checkpoint interface (plain-value constructor,
  testable).
- :class:`~sobits_intball2_gnc.control.utils.trajectory_controller.TrajectoryController`
  -- pure feedforward+feedback translation controller for a moving Guidance
  setpoint (Phase 3a, ``openspec/changes/add-trajectory-following``).
- :class:`HoverController` -- DI orchestration logic: reads the injected
  ``ImuSubscriber`` and (optional) ``TfClient`` (common/ros), combines the IMU law with the
  pose correction, allocates via the injected :class:`ThrustAllocator`, and
  publishes via the injected ``FanDutyPublisher``. It performs no ROS I/O.

  When a ``MultiDOFJointTrajectorySubscriber`` is injected and its setpoint is live (a
  message arrived within ``trajectory_controller.timeout``), translation is
  driven entirely by :class:`TrajectoryController` instead of
  :class:`PoseCorrector`'s checkpoint hold -- the two never contribute force
  in the same tick (see ``docs/phase3.md`` / the openspec change for the full
  contract). Attitude (torque) follows the same split (Phase 3b): while
  trajectory following is active and a ``q_des`` has been received,
  :class:`TrajectoryController` also supplies the torque; otherwise (or while
  ``q_des`` is still unset, e.g. before Guidance's first setpoint)
  :class:`PoseCorrector`'s checkpoint/hold attitude torque is used.
  :class:`PoseCorrector` is still called every tick regardless, so its
  liveness/hold state keeps tracking even when its output is discarded.
"""
import time

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor

from sobits_intball2_gnc.control.utils.hover_law import DEFAULT_HOVER, HoverLaw
from sobits_intball2_gnc.control.utils.pose_corrector import (
    DEFAULT_TF,
    PoseCorrector,
    STATUS_MISSING,
    STATUS_OFF,
    STATUS_OK,
    STATUS_STALE,
)
from sobits_intball2_gnc.control.utils.trajectory_controller import (
    DEFAULT_TRAJECTORY,
    TrajectoryController,
)

__all__ = [
    "DEFAULT_HOVER",
    "DEFAULT_TF",
    "DEFAULT_TRAJECTORY",
    "HOVER_MODES",
    "HoverController",
    "HoverLaw",
    "PoseCorrector",
    "STATUS_MISSING",
    "STATUS_OFF",
    "STATUS_OK",
    "STATUS_STALE",
    "TrajectoryController",
]

HOVER_MODES = ("imu", "tf_imu")


class HoverController:
    """DI orchestration logic: read IMU + TF, hover, allocate, publish.

    Args:
        imu_subscriber: injected ``ImuSubscriber`` (source of gyro/acc).
        fan_publisher: injected ``FanDutyPublisher`` (duty output).
        allocator: injected :class:`ThrustAllocator`.
        law: :class:`HoverLaw` instance (IMU control law).
        tf_client: optional ``TfClient`` (common/ros). None -> pure IMU hover; the node
            decides this from ``hover_control.mode`` and injects the result.
        corrector: optional :class:`PoseCorrector` (used when ``tf_client`` is
            given).
        trajectory_subscriber: optional injected ``MultiDOFJointTrajectorySubscriber``
            (source of the Guidance setpoint). Only meaningful alongside
            ``tf_client``/``corrector``; ignored in IMU-only mode.
        trajectory_controller: optional :class:`TrajectoryController`, paired
            with ``trajectory_subscriber``.
        trajectory_timeout: seconds since the last setpoint after which
            trajectory following is considered stale and control falls back
            to ``corrector``'s checkpoint hold.
    """

    def __init__(self, imu_subscriber, fan_publisher, allocator, law,
                 tf_client=None, corrector=None, trajectory_subscriber=None,
                 trajectory_controller=None,
                 trajectory_timeout=DEFAULT_TRAJECTORY["timeout"]) -> None:
        self._imu = imu_subscriber
        self._fan = fan_publisher
        self._allocator = allocator
        self._law = law
        self._tf = tf_client
        self._corrector = corrector if tf_client is not None else None
        self._trajectory_sub = (
            trajectory_subscriber if tf_client is not None else None
        )
        self._trajectory_ctrl = (
            trajectory_controller if tf_client is not None else None
        )
        self._trajectory_timeout = float(trajectory_timeout)
        self._was_trajectory_active = False
        # checkpoint_version recorded when trajectory following last became
        # active -- see the falling-edge re-capture guard below.
        self._checkpoint_version_at_traj_start = None
        # Last-tick (force, torque) split by source, pre-combination -- for
        # Phase 0 diagnosis of whether the TF correction and the IMU law
        # cancel each other out. See docs/main_plan.md Phase 0.
        self._last_force_imu = [0.0, 0.0, 0.0]
        self._last_torque_imu = [0.0, 0.0, 0.0]
        self._last_force_corr = [0.0, 0.0, 0.0]
        self._last_torque_corr = [0.0, 0.0, 0.0]

    @property
    def tf_status(self) -> str:
        """TF liveness for status logging: ``off`` when running IMU-only."""
        if self._corrector is None:
            return STATUS_OFF
        return self._corrector.status

    @property
    def trajectory_active(self) -> bool:
        """True when the last tick's translation force came from the trajectory setpoint."""
        return self._was_trajectory_active

    @property
    def last_force_imu(self) -> list:
        """Last tick's IMU-law force [Fx,Fy,Fz], before TF correction is added."""
        return list(self._last_force_imu)

    @property
    def last_torque_imu(self) -> list:
        """Last tick's IMU-law torque [Tx,Ty,Tz], before TF correction is added."""
        return list(self._last_torque_imu)

    @property
    def last_force_corr(self) -> list:
        """Last tick's translation-correction force [Fx,Fy,Fz] (checkpoint hold or trajectory)."""
        return list(self._last_force_corr)

    @property
    def last_torque_corr(self) -> list:
        """Last tick's TF-correction torque [Tx,Ty,Tz] (zero when corrector is off)."""
        return list(self._last_torque_corr)

    @staticmethod
    def declare_parameters(node) -> None:
        """Declare the hover-law, TF-correction, and trajectory-controller parameters."""
        static_descriptor = ParameterDescriptor(read_only=True)
        hover_static_keys = {"mode", "control_rate"}
        tf_static_keys = {
            "reference_frame", "target_frame", "poll_rate",
            "smooth_window", "smooth_sigma", "checkpoint_topic",
        }
        trajectory_static_keys = {"mass"}
        for key, default in DEFAULT_HOVER.items():
            name = f"hover_control.{key}"
            if not node.has_parameter(name):
                descriptor = static_descriptor if key in hover_static_keys else None
                node.declare_parameter(name, default, descriptor)
        for key, default in DEFAULT_TF.items():
            name = f"tf_correction.{key}"
            if not node.has_parameter(name):
                descriptor = static_descriptor if key in tf_static_keys else None
                node.declare_parameter(name, default, descriptor)
        for key, default in DEFAULT_TRAJECTORY.items():
            name = f"trajectory_controller.{key}"
            if not node.has_parameter(name):
                descriptor = static_descriptor if key in trajectory_static_keys else None
                node.declare_parameter(name, default, descriptor)

    @classmethod
    def from_node(cls, node, imu_subscriber, fan_publisher, allocator,
                  tf_client=None, trajectory_subscriber=None) -> "HoverController":
        """Build the hover law and, when TF is injected, the pose corrector
        and trajectory controller."""
        cls.declare_parameters(node)

        def h(key):
            return node.get_parameter(f"hover_control.{key}").value

        def f(key):
            return node.get_parameter(f"tf_correction.{key}").value

        def g(key):
            return node.get_parameter(f"trajectory_controller.{key}").value

        law = HoverLaw(
            kd_w=h("kd_w"), kp_a=h("kp_a"),
            deadband_w=h("deadband_w"), deadband_a=h("deadband_a"),
            acc_bias_alpha=h("acc_bias_alpha"),
            max_force=h("max_force"), max_torque=h("max_torque"),
        )
        corrector = None
        trajectory_ctrl = None
        if tf_client is not None:
            corrector = PoseCorrector(
                poll_rate=f("poll_rate"), smooth_window=f("smooth_window"),
                smooth_sigma=f("smooth_sigma"), timeout=f("timeout"),
                kp_pos=f("kp_pos"), kp_att=f("kp_att"), kd_pos=f("kd_pos"),
                kd_att=f("kd_att"), vel_filter_alpha=f("vel_filter_alpha"),
                att_filter_alpha=f("att_filter_alpha"),
                max_corr_force=f("max_corr_force"),
                max_corr_torque=f("max_corr_torque"),
            )
            trajectory_ctrl = TrajectoryController(
                mass=g("mass"), kp_pos=g("kp_pos"), kd_pos=g("kd_pos"),
                vel_filter_alpha=g("vel_filter_alpha"), max_force=g("max_force"),
                kp_att=g("kp_att"), kd_att=g("kd_att"),
                att_filter_alpha=g("att_filter_alpha"), max_torque=g("max_torque"),
            )
        return cls(imu_subscriber, fan_publisher, allocator, law,
                   tf_client, corrector, trajectory_subscriber, trajectory_ctrl,
                   trajectory_timeout=g("timeout") if tf_client is not None
                   else DEFAULT_TRAJECTORY["timeout"])

    # --- dynamic gain reconfiguration (delegated to the pure-function
    # objects; see
    # docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md
    # category A) -----

    def set_hover_gains(self, **kwargs) -> None:
        """Update ``hover_control.*`` gains in place. See ``HoverLaw.set_gains``."""
        self._law.set_gains(**kwargs)

    def set_tf_correction_gains(self, **kwargs) -> None:
        """Update ``tf_correction.*`` gains in place (no-op in IMU-only mode).

        See ``PoseCorrector.set_gains``.
        """
        if self._corrector is not None:
            self._corrector.set_gains(**kwargs)

    def set_trajectory_gains(self, timeout=None, **kwargs) -> None:
        """Update ``trajectory_controller.*`` gains in place (no-op in IMU-only
        mode). ``timeout`` is this class's own stale-setpoint threshold (not
        stored on ``TrajectoryController``); the rest is forwarded to
        ``TrajectoryController.set_gains``.
        """
        if timeout is not None:
            self._trajectory_timeout = float(timeout)
        if self._trajectory_ctrl is not None:
            self._trajectory_ctrl.set_gains(**kwargs)

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

    def _trajectory_is_live(self, t) -> bool:
        """True when a trajectory setpoint has arrived within the timeout."""
        if self._trajectory_sub is None or not self._trajectory_sub.ready:
            return False
        last_t = self._trajectory_sub.last_received_t
        if last_t is None:
            return False
        return (t - last_t) <= self._trajectory_timeout

    def step(self, t=None, feedforward=None) -> None:
        """One control tick: read IMU (+TF), compute wrench, allocate, publish."""
        gyro, acc = self._imu.gyro, self._imu.acc
        if gyro is None or acc is None:
            self._fan.set_duty_array([])  # no IMU yet -> idle
            return
        force, torque = self._law.compute(gyro, acc, feedforward_force=feedforward)
        self._last_force_imu, self._last_torque_imu = force, torque
        if self._corrector is not None:
            if t is None:
                t = time.monotonic()

            trajectory_active = self._trajectory_is_live(t)
            if not trajectory_active and self._was_trajectory_active:
                # Falling back this tick: re-capture the checkpoint hold
                # target from the current pose (about to be read by
                # update() below) instead of the stale pre-trajectory
                # target, so translation doesn't jump. See docs/phase3.md.
                #
                # Guarded: skip this if someone (e.g. GuidanceExecutor's
                # align_at_arrival) has already set a fresh checkpoint since
                # trajectory following started. Without this guard, a
                # checkpoint published right after the trajectory ends can
                # arrive before this falling-edge tick fires (bounded by
                # trajectory_controller.timeout) and gets silently
                # overwritten with "hold current pose" a tick later --
                # docs/archive/achieved/2026-08-21_tf_correction_attitude_gain_tuning.md's confirmed
                # root cause for align_at_arrival appearing to never
                # converge on a large residual.
                if (
                    self._checkpoint_version_at_traj_start is None
                    or self._corrector.checkpoint_version
                    == self._checkpoint_version_at_traj_start
                ):
                    self._corrector.set_checkpoints([])
            if trajectory_active and not self._was_trajectory_active:
                # Rising edge: TrajectoryController is a single long-lived
                # instance (constructed once in from_node()), so without this
                # it would still carry the PREVIOUS move's velocity/attitude-
                # rate finite-difference state (_last_pos/_last_qe_vec) into
                # this new one. A Guidance node issuing several back-to-back
                # CtlCommand goals (docs/guidance_node_implementation_plan.md)
                # is exactly the case that exercises this repeatedly -- reset
                # so the first tick of a new move isn't computed against a
                # stale prior-move state.
                self._trajectory_ctrl.reset()
                self._checkpoint_version_at_traj_start = (
                    self._corrector.checkpoint_version
                )
            self._was_trajectory_active = trajectory_active

            pose = self._tf.get_pose()
            # PoseCorrector.update() runs every tick regardless of mode: it
            # is the only source of attitude torque, and it must keep
            # tracking pose/liveness state even while its force output is
            # discarded during trajectory following (see module docstring).
            f_corr, t_corr = self._corrector.update(t, pose)

            if trajectory_active and pose is not None:
                pos_now, quat_now, stamp = pose
                f_corr = self._trajectory_ctrl.compute(
                    stamp, pos_now, quat_now,
                    self._trajectory_sub.p_des, self._trajectory_sub.v_des,
                    self._trajectory_sub.a_des,
                )
                q_des = self._trajectory_sub.q_des
                if q_des is not None:
                    t_corr = self._trajectory_ctrl.compute_attitude(
                        stamp, quat_now, q_des,
                    )

            self._last_force_corr, self._last_torque_corr = f_corr, t_corr
            force = np.clip(
                np.asarray(force) + f_corr,
                -self._law.max_force, self._law.max_force,
            ).tolist()
            torque = np.clip(
                np.asarray(torque) + t_corr,
                -self._law.max_torque, self._law.max_torque,
            ).tolist()
        else:
            self._last_force_corr = [0.0, 0.0, 0.0]
            self._last_torque_corr = [0.0, 0.0, 0.0]
        duties = self._allocator.allocate(force, torque)
        self._fan.set_duty_array(duties)
