"""ROS action client for ``ib2_msgs/action/CtlCommand`` goals."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ib2_msgs.action import CtlCommand
from ib2_msgs.msg import CtlStatusType

ACTION_NAME = "/gnc/move_to"
DEFAULT_REFERENCE_FRAME = "iss_body"
SPIN_POLL_SEC = 0.1
CANCEL_WAIT_SEC = 5.0


class CtlCommandActionClient:
    """Send an absolute or body-relative goal using a caller-owned node."""

    def __init__(self, node: Node, action_name: str = ACTION_NAME,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._action_name = action_name
        self._client = ActionClient(node, CtlCommand, action_name)

    def send_goal(self, pos, quat, feedback_cb=None, timeout_sec: float = 10.0,
                  result_timeout_sec=None,
                  goal_type: int = CtlStatusType.MOVE_TO_ABSOLUTE_TARGET,
                  frame_id=None):
        """Send a goal and wait for its result, canceling on a sim-time timeout.

        ``timeout_sec`` bounds the server wait. ``result_timeout_sec`` bounds
        the accepted goal's result wait; ``None`` waits indefinitely.
        Returns a ``CtlCommand.Result`` or ``None``.
        """
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().error(
                "[CtlCommandActionClient] action server '%s' not available"
                % self._action_name
            )
            return None

        goal = CtlCommand.Goal()
        goal.target.header.frame_id = self._reference_frame if frame_id is None else frame_id
        goal.target.header.stamp = self._node.get_clock().now().to_msg()
        (goal.target.pose.position.x, goal.target.pose.position.y,
         goal.target.pose.position.z) = pos
        (goal.target.pose.orientation.x, goal.target.pose.orientation.y,
         goal.target.pose.orientation.z, goal.target.pose.orientation.w) = quat
        goal.type.type = goal_type

        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=(
                (lambda fb: feedback_cb(fb.feedback)) if feedback_cb else None
            ),
        )
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("[CtlCommandActionClient] goal rejected")
            return None

        result_future = goal_handle.get_result_async()
        if result_timeout_sec is None:
            rclpy.spin_until_future_complete(self._node, result_future)
            return result_future.result().result

        if self._spin_until_done_or_sim_timeout(result_future, result_timeout_sec):
            return result_future.result().result

        self._node.get_logger().warn(
            "[CtlCommandActionClient] no result within %.1fs (sim time), canceling goal"
            % result_timeout_sec
        )
        cancel_future = goal_handle.cancel_goal_async()
        self._spin_until_done_or_sim_timeout(cancel_future, CANCEL_WAIT_SEC)
        self._spin_until_done_or_sim_timeout(result_future, CANCEL_WAIT_SEC)
        return None

    def _spin_until_done_or_sim_timeout(self, future, timeout_sec) -> bool:
        clock = self._node.get_clock()
        deadline_ns = None
        while not future.done():
            rclpy.spin_once(self._node, timeout_sec=SPIN_POLL_SEC)
            now_ns = clock.now().nanoseconds
            # Sim clock reads 0 until the first /clock arrives.
            if now_ns == 0:
                continue
            if deadline_ns is None:
                deadline_ns = now_ns + int(timeout_sec * 1e9)
            elif now_ns >= deadline_ns:
                return future.done()
        return True
