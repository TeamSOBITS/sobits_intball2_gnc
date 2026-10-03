#!/usr/bin/env python3
"""Control node driving IntBall2 with the ported JAXA controller (comparison baseline).

Alternative to ``control/control.py`` for the JAXA baseline comparison
(``docs/jaxa_controller_port_plan.md``): same inputs (TF pose, IMU, Guidance's
``/gnc/trajectory_setpoint`` and ``/gnc/checkpoints``), but translation and
attitude come from the JAXA position/attitude controllers and the wrench goes to
JAXA's own ``fsm`` on ``/ctl/wrench`` for allocation, like
``control.thrust_allocation: jaxa_fsm``. Only one of the two control nodes may
run: both take the same singleton lock.

Launch with ``control_jaxa.launch.py`` (``gnc_params.yaml`` + ``jaxa_control.yaml``).
"""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.ctl_status_subscriber import CtlStatusSubscriber
from sobits_intball2_gnc.control.ros.imu_subscriber import ImuSubscriber
from sobits_intball2_gnc.control.ros.multi_dof_joint_trajectory_subscriber import (
    MultiDOFJointTrajectorySubscriber,
)
from sobits_intball2_gnc.control.ros.pose_array_subscriber import PoseArraySubscriber
from sobits_intball2_gnc.control.ros.wrench_publisher import (
    JAXA_FSM_WRENCH_TOPIC, WRENCH_TOTAL_TOPIC, WrenchPublisher,
)
from sobits_intball2_gnc.control.utils.jaxa_control_params import (
    make_attitude_controller,
    make_position_controller,
    unflatten,
)
from sobits_intball2_gnc.control.utils.jaxa_tracking_controller import JaxaTrackingController
from sobits_intball2_gnc.control.utils.singleton_lock import (
    SingletonLockError,
    acquire_singleton_lock,
)
from sobits_intball2_gnc.control.utils.trajectory_controller import DEFAULT_TRAJECTORY

# Polls TF faster than it arrives (~42 Hz) only to cut latency; the control law
# itself runs once per new TF pose.
POLL_RATE_HZ = 200.0
STATUS_LOG_PERIOD_S = 2.0
# /ctl/wrench crosses the ROS1 bridge (CPU-bound under load), so it is sent only
# when a new command is computed, plus this keep-alive.
WRENCH_KEEPALIVE_S = 0.5
TF_STARTUP_TIMEOUT = 5.0
# Same as control.py: /ctl/status is 10 Hz but the bridge can drop it 5-15x.
JAXA_CTL_STATUS_TIMEOUT = 3.0


class JaxaControlNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "jaxa_control_node",
            automatically_declare_parameters_from_overrides=True,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        cfg = unflatten({k: p.value for k, p in self.get_parameters_by_prefix("jaxa_control").items()})
        if not cfg:
            raise RuntimeError("jaxa_control.* parameters missing: launch with control_jaxa.launch.py")

        def param(name, default):
            return self.get_parameter(name).value if self.has_parameter(name) else default

        reference_frame = str(param("tf_correction.reference_frame", "iss_body"))
        target_frame = str(param("tf_correction.target_frame", "body"))
        self._ctrl = JaxaTrackingController(
            make_position_controller(cfg), make_attitude_controller(cfg),
            vel_filter_alpha=float(param("trajectory_controller.vel_filter_alpha",
                                         DEFAULT_TRAJECTORY["vel_filter_alpha"])),
            trajectory_timeout=float(param("trajectory_controller.timeout", 0.2)),
            tf_timeout=float(param("tf_correction.timeout", 1.0)))

        self._tf = TfClient(self, reference_frame, target_frame)
        if not self._tf.wait_for_frame(TF_STARTUP_TIMEOUT):
            self.get_logger().warn("TF frames unavailable at startup; zero wrench until they appear")
        self._imu = ImuSubscriber(self)
        self._trajectory = MultiDOFJointTrajectorySubscriber(self, expected_frame=reference_frame)
        self._checkpoints = PoseArraySubscriber(self, on_path=self._ctrl.set_checkpoints,
                                                expected_frame=reference_frame)
        self._ctl_status = CtlStatusSubscriber(self)
        self._fsm_pub = WrenchPublisher(self, topic=JAXA_FSM_WRENCH_TOPIC)
        self._total_pub = WrenchPublisher(self, topic=WRENCH_TOTAL_TOPIC)
        self._sent = self._held = 0
        # Free drift: fans are cut (zero wrench) until released; nothing resumes control on its own.
        self._free_drift = False
        self.create_service(SetBool, "~/free_drift", self._on_free_drift)
        self._free_drift_pub = self.create_publisher(
            Bool, "~/free_drift_active",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._free_drift_pub.publish(Bool(data=False))
        self._last_sent_t = None
        self._timer = self.create_timer(1.0 / POLL_RATE_HZ, self._on_timer)
        self._status_timer = self.create_timer(STATUS_LOG_PERIOD_S, self._on_status_log)
        p = cfg["pos_ctl"]
        self.get_logger().info(
            "JaxaControlNode up: pos kp=%.4f ki=%.4f kd=%.4f, att kp=%.4f kd=%.4f, frames %s <- %s, "
            "publishing %s, use_sim_time=%s"
            % (p["kp"], p["ki"], p["kd"], cfg["att_ctl"]["kp"], cfg["att_ctl"]["kd"],
               reference_frame, target_frame, JAXA_FSM_WRENCH_TOPIC,
               self.get_parameter("use_sim_time").value))

    def _setpoint(self):
        sub = self._trajectory
        if not sub.ready:
            return None, None
        return ({"p_des": sub.p_des, "v_des": sub.v_des, "a_des": sub.a_des,
                 "q_des": sub.q_des, "omega_des": sub.omega_des}, sub.last_received_t)

    def _on_free_drift(self, request, response):
        if request.data == self._free_drift:
            response.success, response.message = True, "unchanged"
            return response
        self._free_drift = request.data
        if self._free_drift:
            self._publish_zero_wrench()
            self.get_logger().warn("free drift ON: fans cut until released")
        else:
            self._ctrl.release()
            self._last_sent_t = None
            self.get_logger().info("free drift OFF: control resumed, holding the current pose")
        self._free_drift_pub.publish(Bool(data=self._free_drift))
        response.success, response.message = True, "free drift %s" % ("on" if self._free_drift else "off")
        return response

    def _publish_zero_wrench(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self._total_pub.publish([0.0] * 3, [0.0] * 3)
        if self._ctl_status.jaxa_ctl_idle(now, JAXA_CTL_STATUS_TIMEOUT):
            self._fsm_pub.publish([0.0] * 3, [0.0] * 3)
        self._last_sent_t = now

    def _on_timer(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._free_drift:
            if self._last_sent_t is None or now - self._last_sent_t >= WRENCH_KEEPALIVE_S:
                self._publish_zero_wrench()
            return
        setpoint, setpoint_t = self._setpoint()
        force, torque = self._ctrl.step(now, self._tf.get_pose(), self._imu.gyro, setpoint, setpoint_t)
        due = self._last_sent_t is None or now - self._last_sent_t >= WRENCH_KEEPALIVE_S
        if not (self._ctrl.updated or due):
            return
        self._total_pub.publish(force, torque)
        if self._ctl_status.jaxa_ctl_idle(now, JAXA_CTL_STATUS_TIMEOUT):
            self._fsm_pub.publish(force, torque)
            self._last_sent_t = now
            self._sent += 1
        else:
            self._held += 1

    def _on_status_log(self) -> None:
        sent, held = self._sent, self._held
        self._sent = self._held = 0
        msg = ("jaxa-control: %s, %s msgs=%d, held=%d, jaxa ctl status type=%s, imu=%s, v=[%s]"
               % (self._ctrl.status, JAXA_FSM_WRENCH_TOPIC, sent, held, self._ctl_status.type,
                  "ok" if self._imu.ready else "WAITING",
                  ", ".join("%.3f" % x for x in self._ctrl.velocity)))
        if held:
            self.get_logger().warn(msg + "  <-- JAXA ctl_only not confirmed idle: not publishing")
        else:
            self.get_logger().info(msg)

    def publish_zero(self) -> None:
        self._fsm_pub.publish([0.0] * 3, [0.0] * 3)


def main(args=None) -> None:
    import sys
    try:
        lock_file = acquire_singleton_lock()  # noqa: F841  shared with control.py
    except SingletonLockError as exc:
        print("jaxa_control: %s" % exc, file=sys.stderr)
        sys.exit(1)
    rclpy.init(args=args)
    node = JaxaControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # fsm only reacts to new wrench messages, so stop the fans explicitly.
        if rclpy.ok():
            node.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
