#!/usr/bin/env python3
"""Keyboard teleoperation node: operator keys -> moving reference on ``/gnc/trajectory_setpoint``.

The reference generator (``reference.py``) is ROS-free; this node only wires it to TF, the guidance
goal status, the JAXA ctl status and the trajectory topic, and runs the enable/stop state machine.
tkinter owns the main thread (``keyboard_gui.py``); rclpy spins in a background thread, and the two
meet only in :class:`~sobits_intball2_gnc.teleop.link.TeleopLink`.

Start it only while no ``/gnc/move_to`` goal is running: while one is, input is ignored and nothing
is published (docs/teleop_keyboard_design.md 3, 6).
"""
import signal
import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.ctl_status_subscriber import CtlStatusSubscriber
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.constraints.actuation_envelope import wrench_envelope_halfspaces
from sobits_intball2_gnc.guidance.ros.multi_dof_joint_trajectory_publisher import MultiDOFJointTrajectoryPublisher
from sobits_intball2_gnc.teleop.link import TeleopLink
from sobits_intball2_gnc.teleop.reference import TeleopReference, limits_for_levels
from sobits_intball2_gnc.teleop.ros.guidance_status_subscriber import GuidanceStatusSubscriber
from sobits_intball2_gnc.teleop.ros.fan_duty_subscriber import FanDutySubscriber
from sobits_intball2_gnc.teleop.ros.reference_marker_publisher import ReferenceMarkerPublisher
from sobits_intball2_gnc.teleop.state import KeyState, Status, TeleopState

TF_STARTUP_TIMEOUT = 5.0
JAXA_CTL_STATUS_TIMEOUT = 3.0   # same as jaxa_control_node
MAX_DT = 0.1                    # clamp one integration step (a stalled clock must not jump the reference)
MIN_DT = 0.005
STATUS_COLORS = {
    Status.TRACKING: (0.1, 0.9, 0.2, 0.8),
    Status.STALLED: (1.0, 0.8, 0.0, 0.8),
    Status.STOPPING: (0.3, 0.7, 1.0, 0.8),
    Status.STALL_STOPPED: (1.0, 0.2, 0.2, 0.8),
}

# Phases of the publish state machine.
IDLE, RUN, STOPPING, HOLD = "idle", "run", "stopping", "hold"


class TeleopNode(Node):
    def __init__(self, link: TeleopLink) -> None:
        super().__init__("teleop_node", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self._link = link

        def param(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        self._rate = float(param("teleop.rate", 50.0))
        self._guidance_cooldown = float(param("teleop.guidance_cooldown", 2.0))
        self._estop_hold = float(param("teleop.estop_hold", 1.0))
        self._stall_timeout = float(param("teleop.stall_timeout", 2.0))
        self._tf_timeout = float(param("teleop.tf_staleness_timeout", 1.0))
        reference_frame = str(param("tf_correction.reference_frame", "iss_body"))
        target_frame = str(param("tf_correction.target_frame", "body"))
        mass = float(param("trajectory_controller.mass", 3.216))
        inertia = float(param("trajectory_controller.inertia", 0.0136))

        allocator = ThrustAllocator.from_node(self)
        positions = self.get_parameter("thrust_allocator.fan_positions").value
        self._fan_positions = tuple(tuple(positions[i:i + 3]) for i in range(0, len(positions), 3))
        self._fan_duty = FanDutySubscriber(self)
        envelope = wrench_envelope_halfspaces(allocator.A, allocator.fj_max,
                                              safety_margin=float(param("teleop.wrench_envelope_safety_margin", 1.0)))
        self._envelope, self._mass, self._inertia = envelope, mass, inertia
        self._speed_values = tuple(float(v) for v in param("teleop.speed_levels", [0.03, 0.05, 0.075, 0.10, 0.15]))
        self._accel_values = tuple(float(v) for v in param("teleop.acc_frac_levels", [0.3, 0.4, 0.5, 0.6, 0.7]))
        self._wmax_per_vmax = float(param("teleop.wmax_per_vmax", 2.0))
        self._alpha_per_acc = float(param("teleop.alpha_per_acc", 0.5))
        self._resume = float(param("teleop.resume_ratio", 0.8))
        self._speed_level = min(max(int(param("teleop.speed_level_default", 1)), 0), len(self._speed_values) - 1)
        self._accel_level = min(max(int(param("teleop.acc_level_default", 2)), 0), len(self._accel_values) - 1)
        limits = self._limits_for(self._speed_level, self._accel_level)
        self._ref = TeleopReference(limits, mass, inertia, envelope=envelope)
        self._limits = limits

        self._tf = TfClient(self, reference_frame, target_frame)
        if not self._tf.wait_for_frame(TF_STARTUP_TIMEOUT):
            self.get_logger().warn("TF frames unavailable at startup; teleop stays disabled until they appear")
        self._guidance = GuidanceStatusSubscriber(self)
        self._ctl_status = CtlStatusSubscriber(self)
        self._pub = MultiDOFJointTrajectoryPublisher(self, reference_frame=reference_frame)
        self._marker = ReferenceMarkerPublisher(self, reference_frame=reference_frame)

        self._phase = IDLE
        self._hold_until = 0.0
        self._estop_latched = False
        self._stalled_since = None
        self._stall_stopped = False     # set by an automatic stop, cleared at the next start
        self._last_t = None
        self._last_logged = None
        self._timer = self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(
            "TeleopNode up: vmax=%.3f m/s wmax=%.3f rad/s acc=%s m/s^2 alpha=%s rad/s^2 err=%.3f m/%.1f deg, "
            "frames %s <- %s, use_sim_time=%s"
            % (limits.vmax, limits.wmax, np.round(limits.acc, 4), np.round(limits.alpha, 4), limits.err_pos,
               np.degrees(limits.err_att), reference_frame, target_frame, self.get_parameter("use_sim_time").value))

    def _limits_for(self, speed_level, accel_level):
        return limits_for_levels(self._envelope, self._mass, self._inertia, self._speed_values[speed_level],
                                 self._accel_values[accel_level], self._wmax_per_vmax, self._alpha_per_acc,
                                 self._resume)

    def _apply_levels(self, key: KeyState) -> None:
        """Apply a requested speed/acceleration level, only while the reference is at rest (or not running)."""
        speed = self._speed_level if key.speed_level is None else min(max(int(key.speed_level), 0),
                                                                     len(self._speed_values) - 1)
        accel = self._accel_level if key.accel_level is None else min(max(int(key.accel_level), 0),
                                                                     len(self._accel_values) - 1)
        if (speed, accel) == (self._speed_level, self._accel_level):
            return
        if self._phase != IDLE and not self._ref.at_rest:
            return
        self._speed_level, self._accel_level = speed, accel
        self._limits = self._limits_for(speed, accel)
        self._ref.set_limits(self._limits)
        self.get_logger().info(
            "teleop limits: vmax=%.3f m/s wmax=%.3f rad/s acc=%s m/s^2 err=%.3f m/%.1f deg"
            % (self._limits.vmax, self._limits.wmax, np.round(self._limits.acc, 4), self._limits.err_pos,
               np.degrees(self._limits.err_att)))

    def _publish_state(self, status: Status) -> None:
        r, lim = self._ref, self._limits
        duties, fan_status = self._fan_duty.snapshot()
        v_ratio = tuple(np.concatenate([r.vb / lim.vmax, r.wb / lim.wmax]))
        self._link.set_state(TeleopState(
            status=status, v_ratio=v_ratio, v_body=tuple(r.vb), w_body=tuple(r.wb),
            pos_err=r.pos_err, att_err=r.att_err, err_pos_limit=lim.err_pos, err_att_limit=lim.err_att,
            shaped=r.scaled, speed_values=self._speed_values, accel_values=self._accel_values,
            speed_level=self._speed_level, accel_level=self._accel_level,
            fan_duties=duties, fan_positions=self._fan_positions, fan_status=fan_status))

    def _go_idle(self, status: Status) -> None:
        if self._phase != IDLE:
            self._marker.clear()
        self._phase = IDLE
        self._publish_state(status)

    def _on_timer(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 1.0 / self._rate if self._last_t is None else min(max(now - self._last_t, MIN_DT), MAX_DT)
        self._last_t = now
        key = self._link.take_key()
        self._apply_levels(key)
        if not key.enable:
            self._estop_latched = False

        pose = self._tf.get_pose()
        if pose is None or now - pose[2] > self._tf_timeout:
            return self._go_idle(Status.NO_TF)
        if self._guidance.blocked(now, self._guidance_cooldown):
            return self._go_idle(Status.GUIDANCE_ACTIVE)
        if not self._ctl_status.jaxa_ctl_idle(now, JAXA_CTL_STATUS_TIMEOUT):
            return self._go_idle(Status.CONTROL_BUSY)
        p_meas, q_meas = pose[0], pose[1]

        if key.estop and self._phase in (RUN, STOPPING):
            # Brake at the measured pose: the controller sees a small error and zero reference speed.
            self._ref.reset(p_meas, q_meas)
            self._phase, self._hold_until, self._estop_latched = HOLD, now + self._estop_hold, True
        elif key.enable and not self._estop_latched:
            if self._phase == IDLE:
                self._ref.reset(p_meas, q_meas)
                self._phase = RUN
                self._stall_stopped = False
                self._stalled_since = None
            elif self._phase == STOPPING:
                self._phase = RUN
        elif self._phase == RUN:
            self._phase = STOPPING

        off = Status.STALL_STOPPED if self._stall_stopped else Status.DISABLED
        if self._phase == IDLE:
            return self._publish_state(off)
        if self._phase == HOLD and now >= self._hold_until:
            return self._go_idle(off)

        axes = key.axes if self._phase == RUN else (0.0,) * 6
        sp = self._ref.step(dt, axes, p_meas, q_meas)
        self._pub.publish(sp.p, sp.v, sp.a, sp.q, sp.w, (0.0, 0.0, 0.0))
        if self._phase == STOPPING and self._ref.at_rest:
            return self._go_idle(off)
        # A stall that does not clear (the vehicle is blocked, e.g. against a wall) would keep the controller
        # pushing toward a reference that stays ahead of it: brake at the measured pose and disable instead.
        if self._ref.stalled and self._phase in (RUN, STOPPING):
            self._stalled_since = now if self._stalled_since is None else self._stalled_since
            if now - self._stalled_since >= self._stall_timeout:
                self.get_logger().warn("teleop: reference stalled for %.1f s (error %.1f mm / %.1f deg): "
                                       "braking at the vehicle and disabling" % (
                                           self._stall_timeout, self._ref.pos_err * 1e3,
                                           np.degrees(self._ref.att_err)))
                self._ref.reset(p_meas, q_meas)
                self._phase, self._hold_until = HOLD, now + self._estop_hold
                self._estop_latched = self._stall_stopped = True
                self._stalled_since = None
        else:
            self._stalled_since = None
        if self._stall_stopped and self._phase == HOLD:
            status = Status.STALL_STOPPED
        else:
            status = Status.STALLED if self._ref.stalled else (Status.TRACKING if self._phase == RUN
                                                               else Status.STOPPING)
        self._marker.publish(sp.p, sp.q, STATUS_COLORS[status])
        self._publish_state(status)
        if status != self._last_logged:
            self.get_logger().info("teleop: %s" % status.value)
            self._last_logged = status


def main(args=None) -> None:
    """Run the node in a background thread and the tkinter GUI on the main thread."""
    from sobits_intball2_gnc.teleop.keyboard_gui import KeyboardGui

    rclpy.init(args=args)
    link = TeleopLink()
    node = TeleopNode(link)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    gui = KeyboardGui(link)
    # tkinter swallows KeyboardInterrupt raised inside its callbacks, so Ctrl-C / SIGTERM ask the GUI to close.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: gui.request_close())
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        link.set_key(KeyState())   # released, disabled
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
