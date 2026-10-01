#!/usr/bin/env python3
"""``/gnc/move_to`` goal-status subscriber for the teleop node.

ROS I/O wrapper (does not subclass Node): tells the teleop node whether a guidance goal is running,
so the two never publish ``/gnc/trajectory_setpoint`` at once. The status topic is created by the
action server; without guidance running it simply never arrives, which counts as "no goal".
"""
from action_msgs.msg import GoalStatus, GoalStatusArray
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default

GUIDANCE_STATUS_TOPIC = "/gnc/move_to/_action/status"
_ACTIVE = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)


def any_active(statuses) -> bool:
    """True if any of the goal status codes is ACCEPTED, EXECUTING or CANCELING."""
    return any(s in _ACTIVE for s in statuses)


class GuidanceStatusSubscriber:
    """Track whether a guidance goal is active, and when one last was (node clock, seconds)."""

    def __init__(self, node: Node, topic: str = GUIDANCE_STATUS_TOPIC) -> None:
        self._node = node
        self._active = False
        self._last_active_t = None
        self._sub = node.create_subscription(GoalStatusArray, topic, self._callback,
                                             qos_profile_action_status_default)
        node.get_logger().info("[GuidanceStatusSubscriber] subscribing %s" % topic)

    def _callback(self, msg: GoalStatusArray) -> None:
        self._active = any_active(s.status for s in msg.status_list)
        if self._active:
            self._last_active_t = self._node.get_clock().now().nanoseconds * 1e-9

    @property
    def active(self) -> bool:
        return self._active

    def blocked(self, now: float, cooldown: float) -> bool:
        """True while a goal is active or ended less than ``cooldown`` seconds ago."""
        if self._active:
            return True
        return self._last_active_t is not None and now - self._last_active_t < cooldown
