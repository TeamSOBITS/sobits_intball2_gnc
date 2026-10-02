"""Read fan commands independently of whether teleoperation is enabled."""
import math

from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from sobits_intball2_gnc.teleop.state import FanDutyStatus


class FanDutySubscriber:
    def __init__(self, node, timeout=1.0):
        self._node = node
        self._timeout = float(timeout)
        self._received_at = None
        self._duties = ()
        self._valid = False
        # Best effort can receive both reliable control and best-effort bridge publishers.
        self._sub = node.create_subscription(
            Float64MultiArray, "/ctl/duty", self._callback, qos_profile_sensor_data)

    def _callback(self, msg):
        values = tuple(msg.data)
        self._received_at = self._node.get_clock().now().nanoseconds * 1e-9
        self._valid = len(values) == 8 and all(
            math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
        self._duties = values if self._valid else ()

    def snapshot(self):
        if self._received_at is None:
            return (), FanDutyStatus.WAITING
        age = self._node.get_clock().now().nanoseconds * 1e-9 - self._received_at
        if age < 0.0 or age > self._timeout:
            return (), FanDutyStatus.STALE
        if not self._valid:
            return (), FanDutyStatus.INVALID
        return self._duties, FanDutyStatus.LIVE
