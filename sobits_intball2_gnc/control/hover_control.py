#!/usr/bin/env python3
"""IMU-based hover control for IntBall2.

Holds a stable hover using IMU feedback only (``/imu/imu``, ``ib2_msgs/msg/IMU``):
the angular rate (gyro) is damped toward zero and the linear-acceleration
disturbance (accelerometer, with an EMA bias estimate removed) is opposed. The
resulting corrective wrench is allocated to the 8 fans via
:class:`ThrustAllocator` and published through :class:`FanControlNode`.

The control law lives in :class:`HoverController`, which is ROS-agnostic and
exposes a feed-forward translation-force hook so future free-path motion can be
stacked on top of a stable hover without changing the hover logic.

With IMU only there is no absolute attitude/position reference, so attitude and
position slowly drift. When ``nav_correction.enable`` is set in ``gnc.yaml``,
:class:`NavCorrector` layers a low-gain, independently clamped position/attitude
correction from ``/sensor_fusion/navigation`` (Gaussian-smoothed, down-sampled)
on top of the IMU law, and accepts a checkpoint array (``PoseArray``) as the
interface for the future free-path flight program. Navigation loss degrades
gracefully back to pure IMU hover.
"""
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sobits_intball2_gnc.control.fan_control import FanControlNode
from sobits_intball2_gnc.control.gnc_params import load_gnc_config
from sobits_intball2_gnc.control.thrust_allocator import ThrustAllocator

IMU_TOPIC = "/imu/imu"
FEEDFORWARD_TOPIC = "/gnc/feedforward_force"
NAV_TOPIC = "/sensor_fusion/navigation"


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


class NavCorrector:
    """Navigation-based hover correction (ROS-agnostic, testable).

    Consumes down-sampled ``/sensor_fusion/navigation`` poses (DS frame),
    smooths them with a one-sided Gaussian-weighted window, and produces a
    low-gain corrective (force, torque) toward a hold target. The force is
    returned in the body frame so it can be summed with the IMU hover wrench.
    The correction is clamped independently so it never dominates the IMU law.

    Also owns the checkpoint-array interface for the future free-path flight
    program: a received checkpoint list overrides the hold target, and
    ``advance_checkpoint()`` steps through it.
    """

    def __init__(self, params: dict) -> None:
        self.nav_rate = float(params["nav_rate"])
        self.window = int(params["gauss_window"])
        self.sigma = float(params["gauss_sigma"])
        self.timeout = float(params["timeout"])
        self.kp_pos = np.asarray(params["kp_pos"], dtype=float)
        self.kd_pos = np.asarray(params["kd_pos"], dtype=float)
        self.kp_att = np.asarray(params["kp_att"], dtype=float)
        self.max_corr_force = float(params["max_corr_force"])
        self.max_corr_torque = float(params["max_corr_torque"])

        self._buf: deque = deque(maxlen=self.window)  # (pos, quat, vel)
        self._last_sample_t: float | None = None
        self._last_msg_t: float | None = None
        self._hold_pos: np.ndarray | None = None
        self._hold_quat: np.ndarray | None = None
        self._checkpoints: list = []  # list of (pos, quat)
        self._cp_index: int | None = None

    # --- navigation intake -------------------------------------------------

    def add_sample(self, t: float, pos, quat, vel) -> None:
        """Feed one navigation sample; time-gated to ``nav_rate``.

        ``t`` is a monotonic timestamp in seconds supplied by the caller.
        """
        self._last_msg_t = t
        if (
            self._last_sample_t is not None
            and (t - self._last_sample_t) < 1.0 / self.nav_rate
        ):
            return
        self._last_sample_t = t
        q = np.asarray(quat, dtype=float)
        # Sign-align with the previous sample so component-wise averaging of
        # the double-covered quaternion is well defined.
        if self._buf and float(np.dot(self._buf[-1][1], q)) < 0.0:
            q = -q
        self._buf.append(
            (np.asarray(pos, dtype=float), q, np.asarray(vel, dtype=float))
        )

    def _smoothed(self):
        """Gaussian-weighted average of the buffer (latest sample heaviest)."""
        n = len(self._buf)
        idx = np.arange(n, dtype=float)
        w = np.exp(-((n - 1 - idx) ** 2) / (2.0 * self.sigma ** 2))
        w /= w.sum()
        pos = sum(wi * s[0] for wi, s in zip(w, self._buf))
        vel = sum(wi * s[2] for wi, s in zip(w, self._buf))
        quat = sum(wi * s[1] for wi, s in zip(w, self._buf))
        quat = quat / np.linalg.norm(quat)
        return pos, quat, vel

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

    def compute_correction(self, t: float):
        """Return (force_body, torque) lists, zeros when nav is unavailable."""
        zeros = ([0.0] * 3, [0.0] * 3)
        if not self._buf or self._last_msg_t is None:
            return zeros
        if (t - self._last_msg_t) > self.timeout:
            # Nav lost: degrade to pure IMU and re-capture on recovery
            # (unless a checkpoint target is active).
            if self._cp_index is None:
                self._hold_pos = None
                self._hold_quat = None
            self._buf.clear()
            self._last_sample_t = None
            return zeros

        pos_s, quat_s, vel_s = self._smoothed()
        target_pos, target_quat = self.active_target()
        if target_pos is None:
            # First valid pose after (re)acquisition becomes the hold target.
            self._hold_pos, self._hold_quat = pos_s, quat_s
            target_pos, target_quat = pos_s, quat_s

        # Position correction (DS frame) rotated into the body frame.
        f_ds = self.kp_pos * (target_pos - pos_s) - self.kd_pos * vel_s
        f_body = _quat_rotate(_quat_conj(quat_s), f_ds)

        # Attitude correction from the quaternion error (body frame).
        qe = _quat_mul(_quat_conj(target_quat), quat_s)
        torque = -self.kp_att * np.sign(qe[3] if qe[3] != 0.0 else 1.0) * qe[:3]

        f_body = np.clip(f_body, -self.max_corr_force, self.max_corr_force)
        torque = np.clip(torque, -self.max_corr_torque, self.max_corr_torque)
        return f_body.tolist(), torque.tolist()


class HoverController:
    """IMU-only hover control law (ROS-agnostic, testable)."""

    def __init__(self, params: dict) -> None:
        self.kd_w = np.asarray(params["kd_w"], dtype=float)
        self.kp_a = np.asarray(params["kp_a"], dtype=float)
        self.deadband_w = float(params["deadband_w"])
        self.deadband_a = float(params["deadband_a"])
        self.acc_bias_alpha = float(params["acc_bias_alpha"])
        self.max_force = float(params["max_force"])
        self.max_torque = float(params["max_torque"])
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


class HoverControlNode(Node):
    """Subscribe to IMU, run the hover law, publish fan duties."""

    def __init__(self) -> None:
        super().__init__("hover_control_node")
        cfg = load_gnc_config()
        hc = cfg["hover_control"]
        self._rate = float(hc["control_rate"])
        self._controller = HoverController(hc)
        self._allocator = ThrustAllocator(cfg)
        self._fan = FanControlNode("hover_fan_pub")

        self._gyro = None
        self._acc = None
        self._feedforward = None

        nc = cfg["nav_correction"]
        self._nav_enabled = bool(nc["enable"])
        self._corrector = NavCorrector(nc) if self._nav_enabled else None
        self._standby_enabled = bool(nc["standby_ctl_on_start"])
        self._last_standby_t = 0.0

        # ib2_msgs is only needed at runtime; import here so the package builds
        # without it present.
        from ib2_msgs.msg import IMU
        from geometry_msgs.msg import Vector3

        self._sub_imu = self.create_subscription(
            IMU, IMU_TOPIC, self._on_imu, 10
        )
        self._sub_ff = self.create_subscription(
            Vector3, FEEDFORWARD_TOPIC, self._on_feedforward, 1
        )

        if self._nav_enabled:
            from ib2_msgs.msg import Navigation
            from geometry_msgs.msg import PoseArray

            self._sub_nav = self.create_subscription(
                Navigation, NAV_TOPIC, self._on_nav, 10
            )
            self._sub_cp = self.create_subscription(
                PoseArray, str(nc["checkpoint_topic"]), self._on_checkpoints, 1
            )
            if self._standby_enabled:
                from ib2_msgs.msg import CtlStatus
                from ib2_msgs.action import CtlCommand
                from rclpy.action import ActionClient

                # Supervise the JAXA controller: it re-enters KEEP_POSE on
                # every Navigation OFF->ON toggle, so watch /ctl/status and
                # push it back to STAND_BY whenever it wakes up.
                self._ctl_client = ActionClient(
                    self, CtlCommand, "/ctl/command_ros2"
                )
                self._sub_ctl_status = self.create_subscription(
                    CtlStatus, "/ctl/status", self._on_ctl_status, 1
                )
                self._send_standby()

        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(
            "HoverControlNode up: subscribing %s (nav correction: %s), "
            "publishing /ctl/duty at %.1f Hz"
            % (IMU_TOPIC, "on" if self._nav_enabled else "off", self._rate)
        )

    def _on_imu(self, msg) -> None:
        self._gyro = [msg.gyro_x, msg.gyro_y, msg.gyro_z]
        self._acc = [msg.acc_x, msg.acc_y, msg.acc_z]

    def _on_feedforward(self, msg) -> None:
        self._feedforward = [msg.x, msg.y, msg.z]

    def _on_nav(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.linear
        self._corrector.add_sample(
            time.monotonic(),
            [p.x, p.y, p.z],
            [q.x, q.y, q.z, q.w],
            [v.x, v.y, v.z],
        )

    def _on_ctl_status(self, msg) -> None:
        # STAND_BY = 0. Anything else means the JAXA controller is driving
        # the fans; push it back (debounced to one goal per 2 s).
        now = time.monotonic()
        if msg.type.type != 0 and (now - self._last_standby_t) > 2.0:
            self.get_logger().info(
                "JAXA controller state %d != STAND_BY; re-sending STAND_BY"
                % msg.type.type
            )
            self._send_standby()

    def _on_checkpoints(self, msg) -> None:
        poses = [
            ([p.position.x, p.position.y, p.position.z],
             [p.orientation.x, p.orientation.y,
              p.orientation.z, p.orientation.w])
            for p in msg.poses
        ]
        self._corrector.set_checkpoints(poses)
        self.get_logger().info(
            "checkpoints: %d received%s"
            % (len(poses), "" if poses else " (cleared, re-capturing hold)")
        )

    def advance_checkpoint(self) -> bool:
        """Step the hold target to the next checkpoint (free-path hook)."""
        if self._corrector is None:
            return False
        return self._corrector.advance_checkpoint()

    def _send_standby(self) -> None:
        """Idle the JAXA controller so it stops competing on /ctl/duty."""
        from ib2_msgs.action import CtlCommand

        if not self._ctl_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn(
                "/ctl/command_ros2 action server unavailable; "
                "JAXA controller may keep publishing /ctl/duty"
            )
            return
        goal = CtlCommand.Goal()
        goal.target.header.frame_id = "body"
        goal.type.type = 0  # STAND_BY
        self._ctl_client.send_goal_async(goal)
        self._last_standby_t = time.monotonic()
        self.get_logger().info("sent STAND_BY to /ctl/command_ros2")

    def _on_timer(self) -> None:
        if self._gyro is None or self._acc is None:
            self._fan.set_duty_array([])  # no IMU yet -> idle
            return
        force, torque = self._controller.compute(
            self._gyro, self._acc, feedforward_force=self._feedforward
        )
        if self._nav_enabled:
            f_corr, t_corr = self._corrector.compute_correction(
                time.monotonic()
            )
            force = np.clip(
                np.asarray(force) + f_corr,
                -self._controller.max_force, self._controller.max_force,
            ).tolist()
            torque = np.clip(
                np.asarray(torque) + t_corr,
                -self._controller.max_torque, self._controller.max_torque,
            ).tolist()
        duties = self._allocator.allocate(force, torque)
        self._fan.set_duty_array(duties)

    def destroy_node(self) -> bool:
        self._fan.destroy_node()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HoverControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
