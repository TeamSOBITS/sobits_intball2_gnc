#!/usr/bin/env python3
"""CtlCommand action client for IntBall2 (body-frame relative move).

ROS I/O wrapper (does not subclass Node): the ``MOVE_TO_RELATIVE_TARGET``
counterpart to :class:`~sobits_intball2_gnc.guidance.ros.move_to_client.MoveToClient`,
which it delegates the send/wait/cancel plumbing to. Same arguments as
``intball2_common``'s ``move_relative`` (which drives JAXA's
``ctl/command_ros2``), served by our own guidance instead
(``docs/archive/achieved/2026-09-29_move_relative_design.md``).
"""
import rclpy
from rclpy.node import Node
from ib2_msgs.msg import CtlStatusType
from scipy.spatial.transform import Rotation

from sobits_intball2_gnc.guidance.ros.move_to_client import ACTION_NAME, MoveToClient

BODY_FRAME = "body"


def rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """``[x, y, z, w]`` from roll/pitch/yaw [deg].

    Extrinsic xyz: numerically equal to intball2_common
    ``ctl_command_client.quaternion_from_euler``.
    """
    return Rotation.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat().tolist()


class MoveRelativeClient:
    """Send a body-frame relative ``CtlCommand`` goal.

    Args:
        node: The rclpy Node that owns this client.
        action_name: Action server name (default ``/gnc/move_to``).
    """

    def __init__(self, node: Node, action_name: str = ACTION_NAME) -> None:
        self._move_to_client = MoveToClient(node, action_name)

    def send_goal(self, dp, dq, feedback_cb=None, timeout_sec: float = 10.0,
                  result_timeout_sec=None):
        """Send ``dp`` [m] / ``dq`` ``[x, y, z, w]`` (body frame at goal receipt).

        Same timeouts and return value as ``MoveToClient.send_goal``.
        """
        return self._move_to_client.send_goal(
            dp, dq, feedback_cb=feedback_cb, timeout_sec=timeout_sec,
            result_timeout_sec=result_timeout_sec,
            goal_type=CtlStatusType.MOVE_TO_RELATIVE_TARGET, frame_id=BODY_FRAME,
        )


def main(args=None) -> None:
    """CLI: send a relative move goal.

    Run with ``ros2 run sobits_intball2_gnc move_relative_client -x 0.3 -w 90``.
    """
    import argparse
    import sys
    from rclpy.parameter import Parameter
    from rclpy.utilities import remove_ros_args

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
    client = MoveRelativeClient(node, ns.action_name)
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

    result = client.send_goal(dp, dq, feedback_cb=on_feedback, timeout_sec=10.0,
                              result_timeout_sec=ns.result_timeout)
    if result is None:
        node.get_logger().error("[move_relative_client] goal did not complete")
    else:
        node.get_logger().info(
            "[move_relative_client] finished with result type=%d" % result.type
        )
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
