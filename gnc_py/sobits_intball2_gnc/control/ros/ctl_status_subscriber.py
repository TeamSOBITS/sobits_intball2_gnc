#!/usr/bin/env python3
"""JAXA ``ctl_only`` status subscriber (`/ctl/status`, ``ib2_msgs/CtlStatus``).

ROS I/O wrapper (does not subclass Node): keeps the latest control status type
and when it arrived (node clock, i.e. sim time), so ``control_node`` can tell
whether JAXA's controller is idle before feeding JAXA's fsm itself.
``ib2_msgs`` is imported lazily so the package builds without it present.
"""
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

CTL_STATUS_TOPIC = "/ctl/status"
# ctl_only publishes a continuous wrench only at or above this status type
# (Ctl::navinfoCallback, ib2_msgs/CtlStatusType); DOCKING_STAND_BY also does.
KEEP_POSE = 10
DOCKING_STAND_BY = 65


class CtlStatusSubscriber:
    """Buffer the latest JAXA control status type from ``/ctl/status``."""

    def __init__(self, node: Node, topic: str = CTL_STATUS_TOPIC,
                 qos_profile: QoSProfile = qos_profile_sensor_data) -> None:
        self._node = node
        self._type = None
        self._received_t = None
        from ib2_msgs.msg import CtlStatus
        self._sub = node.create_subscription(CtlStatus, topic, self._callback, qos_profile)
        node.get_logger().info(f"[CtlStatusSubscriber] subscribing {topic}")

    def _callback(self, msg) -> None:
        self._type = int(msg.type.type)
        self._received_t = self._node.get_clock().now().nanoseconds * 1e-9

    @property
    def type(self):
        """Latest ``CtlStatusType.type``, or None."""
        return self._type

    def jaxa_ctl_idle(self, now: float, timeout: float) -> bool:
        """True if a status arrived within ``timeout`` [s] of ``now`` and says
        ``ctl_only`` is not publishing its own wrench."""
        if self._type is None or now - self._received_t > timeout:
            return False
        return self._type < KEEP_POSE and self._type != DOCKING_STAND_BY
