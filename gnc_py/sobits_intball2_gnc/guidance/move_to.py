"""Resolve a named TF location and send an absolute Guidance goal."""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.ctl_command_action_client import (
    ACTION_NAME,
    DEFAULT_REFERENCE_FRAME,
    CtlCommandActionClient,
)


def resolve_location(node: Node, location_name: str,
                     reference_frame: str = DEFAULT_REFERENCE_FRAME,
                     timeout_sec: float = 5.0):
    """Return the named frame's ``(pos, quat)`` in ``reference_frame`` or None."""
    tf_client = TfClient(node, reference_frame=reference_frame,
                         target_frame=location_name)
    if not tf_client.wait_for_frame(timeout_sec):
        return None
    pos, quat, _stamp = tf_client.get_pose()
    return pos, quat


def main(args=None) -> None:
    """Send a named location with the ``move_to_client`` console command."""
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
    try:
        node.get_logger().info(
            "[move_to_client] resolving '%s' via TF..." % ns.location_name
        )
        resolved = resolve_location(node, ns.location_name, ns.reference_frame)
        if resolved is None:
            node.get_logger().error(
                "[move_to_client] could not resolve TF frame '%s'" % ns.location_name
            )
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

        client = CtlCommandActionClient(node, ns.action_name, ns.reference_frame)
        result = client.send_goal(pos, quat, feedback_cb=on_feedback, timeout_sec=10.0)
        if result is None:
            node.get_logger().error("[move_to_client] goal did not complete")
        else:
            node.get_logger().info(
                "[move_to_client] finished with result type=%d" % result.type
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
