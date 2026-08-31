#!/usr/bin/env python3
"""CtlCommand action client for IntBall2 (move-to-named-location).

ROS I/O wrapper (does not subclass Node): the client-side counterpart to
:class:`~sobits_intball2_gnc.guidance.ros.ctl_command_action_server.CtlCommandActionServer`.
Resolves a named location (e.g. ``above_dock_2``, published as a TF frame by
``navigation/location_broadcaster.py`` from ``maps/iss_location.yaml``) via
:class:`~sobits_intball2_gnc.common.ros.tf_client.TfClient`, and sends it as a
``ib2_msgs/action/CtlCommand`` goal.

Replaces the per-destination ``test/manual/send_curve_via_naventry_to_*``
scripts' hardcoded ``TARGET_POS`` coordinates: this client looks the location
up live from TF instead, so a single script serves every named location.
"""
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from ib2_msgs.action import CtlCommand
from ib2_msgs.msg import CtlStatusType

from sobits_intball2_gnc.common.ros.tf_client import TfClient

ACTION_NAME = "/gnc/move_to"
DEFAULT_REFERENCE_FRAME = "iss_body"


class MoveToClient:
    """Send a move-to-target ``CtlCommand`` goal, resolved from a TF frame.

    Args:
        node: The rclpy Node that owns this client.
        action_name: Action server name (must match
            ``CtlCommandActionServer``'s, default ``/gnc/move_to``).
        reference_frame: Frame the target pose is looked up/expressed in
            (default ``iss_body``, matching every other interface in this
            package).
    """

    def __init__(self, node: Node, action_name: str = ACTION_NAME,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._action_name = action_name
        self._client = ActionClient(node, CtlCommand, action_name)

    def resolve_location(self, location_name: str, timeout_sec: float = 5.0):
        """Look up ``location_name``'s ``(pos, quat)`` via TF, or ``None``.

        ``location_name`` must be a TF frame with ``self._reference_frame``
        as its parent (e.g. a name from ``maps/iss_location.yaml``,
        published by ``navigation/location_broadcaster.py``).
        """
        local_tf = TfClient(
            self._node, reference_frame=self._reference_frame,
            target_frame=location_name,
        )
        if not local_tf.wait_for_frame(timeout_sec):
            return None
        pos, quat, _stamp = local_tf.get_pose()
        return pos, quat

    def send_goal(self, pos, quat, feedback_cb=None, timeout_sec: float = 10.0):
        """Send a ``MOVE_TO_ABSOLUTE_TARGET`` goal and wait for the result.

        Returns the ``CtlCommand.Result``, or ``None`` if the server wasn't
        available or the goal was rejected.
        """
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().error(
                "[MoveToClient] action server '%s' not available"
                % self._action_name
            )
            return None

        goal = CtlCommand.Goal()
        goal.target.header.frame_id = self._reference_frame
        goal.target.header.stamp = self._node.get_clock().now().to_msg()
        (goal.target.pose.position.x, goal.target.pose.position.y,
         goal.target.pose.position.z) = pos
        (goal.target.pose.orientation.x, goal.target.pose.orientation.y,
         goal.target.pose.orientation.z, goal.target.pose.orientation.w) = quat
        goal.type.type = CtlStatusType.MOVE_TO_ABSOLUTE_TARGET

        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=(
                (lambda fb: feedback_cb(fb.feedback)) if feedback_cb else None
            ),
        )
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("[MoveToClient] goal rejected")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        return result_future.result().result


def main(args=None) -> None:
    """CLI: resolve a named location via TF and send it as a move-to goal.

    Run with ``ros2 run sobits_intball2_gnc move_to_client <location_name>
    [options]``.
    """
    import argparse
    import sys
    from rclpy.parameter import Parameter
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="move_to_client",
        description="Resolve a named location via TF and send it as a "
                    "CtlCommand move-to-target goal.",
    )
    parser.add_argument("location_name",
                        help="TF frame name to move to, e.g. above_dock_2 "
                             "(see maps/iss_location.yaml)")
    parser.add_argument("--action-name", default=ACTION_NAME,
                        help="action server name (default: %(default)s)")
    parser.add_argument("--reference-frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node(
        "move_to_client",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    client = MoveToClient(node, ns.action_name, ns.reference_frame)

    node.get_logger().info(
        "[move_to_client] resolving '%s' via TF..." % ns.location_name
    )
    resolved = client.resolve_location(ns.location_name)
    if resolved is None:
        node.get_logger().error(
            "[move_to_client] could not resolve TF frame '%s'" % ns.location_name
        )
        node.destroy_node()
        rclpy.shutdown()
        return
    pos, quat = resolved
    node.get_logger().info(
        "[move_to_client] resolved %s -> pos=%s quat=%s, sending goal..."
        % (ns.location_name, pos, quat)
    )

    def on_feedback(feedback):
        node.get_logger().info(
            "[move_to_client] time_to_go=%.1fs pose_to_go=(%.3f, %.3f, %.3f)"
            % (feedback.time_to_go.sec + feedback.time_to_go.nanosec * 1e-9,
               feedback.pose_to_go.position.x, feedback.pose_to_go.position.y,
               feedback.pose_to_go.position.z)
        )

    result = client.send_goal(pos, quat, feedback_cb=on_feedback, timeout_sec=10.0)
    if result is None:
        node.get_logger().error("[move_to_client] goal did not complete")
    else:
        node.get_logger().info(
            "[move_to_client] finished with result type=%d" % result.type
        )
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
