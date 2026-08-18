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
  ``ImuSubscriber`` and (optional) ``TfClient``, combines the IMU law with the
  pose correction, allocates via the injected :class:`ThrustAllocator`, and
  publishes via the injected ``FanDutyPublisher``. It performs no ROS I/O.

  When a ``TrajectorySubscriber`` is injected and its setpoint is live (a
  message arrived within ``trajectory_controller.timeout``), translation is
  driven entirely by :class:`TrajectoryController` instead of
  :class:`PoseCorrector`'s checkpoint hold -- the two never contribute force
  in the same tick (see ``docs/phase3.md`` / the openspec change for the full
  contract). Attitude (torque) always comes from :class:`PoseCorrector`
  regardless of which mode is active: it is still called every tick so its
  torque output keeps working, only its *force* return value is discarded
  while trajectory following is active.
"""
import time

import numpy as np

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
        tf_client: optional ``TfClient``. None -> pure IMU hover; the node
            decides this from ``hover_control.mode`` and injects the result.
        corrector: optional :class:`PoseCorrector` (used when ``tf_client`` is
            given).
        trajectory_subscriber: optional injected ``TrajectorySubscriber``
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
        for key, default in DEFAULT_HOVER.items():
            name = f"hover_control.{key}"
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
        for key, default in DEFAULT_TF.items():
            name = f"tf_correction.{key}"
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
        for key, default in DEFAULT_TRAJECTORY.items():
            name = f"trajectory_controller.{key}"
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

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
            )
        return cls(imu_subscriber, fan_publisher, allocator, law,
                   tf_client, corrector, trajectory_subscriber, trajectory_ctrl,
                   trajectory_timeout=g("timeout") if tf_client is not None
                   else DEFAULT_TRAJECTORY["timeout"])

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
                self._corrector.set_checkpoints([])
            self._was_trajectory_active = trajectory_active

            pose = self._tf.get_pose()
            # PoseCorrector.update() runs every tick regardless of mode: it
            # is the only source of attitude torque, and it must keep
            # tracking pose/liveness state even while its force output is
            # discarded during trajectory following (see module docstring).
            f_corr, t_corr = self._corrector.update(t, pose)

            if trajectory_active and pose is not None:
                pos_now, quat_now, _stamp = pose
                f_corr = self._trajectory_ctrl.compute(
                    t, pos_now, quat_now,
                    self._trajectory_sub.p_des, self._trajectory_sub.v_des,
                    self._trajectory_sub.a_des,
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
