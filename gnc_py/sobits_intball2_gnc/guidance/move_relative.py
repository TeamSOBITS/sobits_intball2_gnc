"""Send a body-frame relative Guidance goal from CLI arguments."""

import argparse
import sys

import rclpy
from ib2_msgs.msg import CtlStatusType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from sobits_intball2_gnc.guidance.ros.ctl_command_action_client import (
    ACTION_NAME,
    CtlCommandActionClient,
)

BODY_FRAME = "body"


def rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """Return an ``[x, y, z, w]`` quaternion for extrinsic xyz angles."""
    return Rotation.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat().tolist()


def main(args=None) -> None:
    """Send a relative goal with the ``move_relative_client`` console command."""
    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="move_relative_client",
        description="Send a body-frame relative CtlCommand goal.",
    )
    parser.add_argument("-x", type=float, default=0.0, help="relative move x [m]")
    parser.add_argument("-y", type=float, default=0.0, help="relative move y [m]")
    parser.add_argument("-z", type=float, default=0.0, help="relative move z [m]")
    parser.add_argument("-r", "--roll", type=float, default=0.0, help="relative roll [deg]")
    parser.add_argument("-p", "--pitch", type=float, default=0.0, help="relative pitch [deg]")
    parser.add_argument("-w", "--yaw", type=float, default=0.0, help="relative yaw [deg]")
    parser.add_argument("--action-name", default=ACTION_NAME,
                        help="action server name (default: %(default)s)")
    parser.add_argument("--result-timeout", type=float, default=None, metavar="SEC",
                        help="cancel if no result within SEC of sim time (default: wait)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    dp = [ns.x, ns.y, ns.z]
    dq = rpy_deg_to_quat(ns.roll, ns.pitch, ns.yaw)

    rclpy.init(args=argv)
    node = Node(
        "move_relative_client",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    try:
        node.get_logger().info(
            "[move_relative_client] sending relative goal dp=%s rpy_deg=%s"
            % (dp, [ns.roll, ns.pitch, ns.yaw])
        )

        def on_feedback(feedback):
            node.get_logger().info(
                "[move_relative_client] time_to_go=%.1fs pose_to_go=(%.3f, %.3f, %.3f)"
                % (feedback.time_to_go.sec + feedback.time_to_go.nanosec * 1e-9,
                   feedback.pose_to_go.position.x, feedback.pose_to_go.position.y,
                   feedback.pose_to_go.position.z)
            )

        client = CtlCommandActionClient(node, ns.action_name)
        result = client.send_goal(
            dp, dq, feedback_cb=on_feedback, timeout_sec=10.0,
            result_timeout_sec=ns.result_timeout,
            goal_type=CtlStatusType.MOVE_TO_RELATIVE_TARGET, frame_id=BODY_FRAME,
        )
        if result is None:
            node.get_logger().error("[move_relative_client] goal did not complete")
        else:
            node.get_logger().info(
                "[move_relative_client] finished with result type=%d" % result.type
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
