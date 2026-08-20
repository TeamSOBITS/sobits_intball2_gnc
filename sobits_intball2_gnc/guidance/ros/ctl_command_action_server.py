#!/usr/bin/env python3
"""CtlCommand action server for IntBall2 (goal-driven move-to-target).

ROS I/O wrapper (does not subclass Node): exposes an ``ib2_msgs/action/
CtlCommand`` server as the ``send_goal``-style entry point for "move to this
target pose" (``docs/future_design_notes.md`` 4). This type is reused from
JAXA's own ``ctl_only`` interface -- ``ib2_msgs`` is already a dependency, and
its result/feedback fields (termination reason, time-to-go, pose-to-go) are a
good fit as-is -- but this server is a **separate, independently-named**
action server, not JAXA's ``ctl_only`` itself: giving it a distinct
``action_name`` at construction avoids a client accidentally sending a goal to
the real onboard controller.

As with every other ROS wrapper in this package, the *logic* for actually
reaching the target is not here: it is injected as ``execute_fn`` (this
package's DI convention, e.g. ``PoseArraySubscriber``'s ``on_path``). Guidance
(``docs/future_design_notes.md`` 3-1, not yet implemented -- Phase 2) supplies
that function; this class only converts between the ROS action messages and
plain Python values, and drives the ``rclpy`` action-server state machine.

``execute_fn`` signature::

    execute_fn(p_target, q_target, feedback_cb, is_cancel_requested) -> status

    p_target: [x, y, z]
    q_target: [x, y, z, w]
    feedback_cb: call as feedback_cb(time_to_go_sec, p_to_go, q_to_go) to
        publish progress; p_to_go/q_to_go are target - current (position
        difference, orientation as a quaternion), same convention as
        ``pose_to_go`` in ``CtlCommand.action``.
    is_cancel_requested: callable, returns True once the client has asked to
        cancel; ``execute_fn`` should stop and return promptly when this is
        True.
    status: one of the ``TERMINATE_*`` class constants below.

Requires a callback group that allows ``execute_fn`` to run alongside other
callbacks (e.g. a ``MultiThreadedExecutor``, or a dedicated
``ReentrantCallbackGroup``) -- ``execute_fn`` is expected to run for the
duration of the move, not return immediately.
"""
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from ib2_msgs.action import CtlCommand
from ib2_msgs.msg import CtlStatusType

# Termination status, re-exported from ib2_msgs.action.CtlCommand.Result for
# execute_fn implementations that don't want to import the action type
# directly.
TERMINATE_SUCCESS = CtlCommand.Result.TERMINATE_SUCCESS
TERMINATE_ABORTED = CtlCommand.Result.TERMINATE_ABORTED
TERMINATE_TIME_OUT = CtlCommand.Result.TERMINATE_TIME_OUT
TERMINATE_INVALID_NAV = CtlCommand.Result.TERMINATE_INVALID_NAV
TERMINATE_INVALID_CMD = CtlCommand.Result.TERMINATE_INVALID_CMD


def _seconds_to_duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int((seconds - d.sec) * 1e9)
    return d


class CtlCommandActionServer:
    """Serve ``ib2_msgs/action/CtlCommand`` goals via an injected ``execute_fn``.

    Args:
        node: The rclpy Node that owns this server.
        action_name: Action server name. **Must not** be JAXA's actual
            ``ctl_only`` action name -- pick a distinct namespace (e.g.
            ``/gnc/move_to``) so a client can't send a goal to the real
            onboard controller by mistake.
        execute_fn: Callable implementing the move (see module docstring for
            signature). Only ``CtlStatusType.MOVE_TO_ABSOLUTE_TARGET`` goals
            are accepted; anything else is rejected before ``execute_fn``
            runs (this server's scope, ``docs/future_design_notes.md`` 4).
        expected_frame: Reference frame ``goal.target`` (a
            ``geometry_msgs/PoseStamped``) must be expressed in. An empty
            ``header.frame_id`` is accepted as "unspecified"; any other
            mismatch rejects the goal before ``execute_fn`` runs -- the same
            policy ``PoseArraySubscriber`` uses for ``/gnc/checkpoints``
            (default ``""``: no check, matching this class's behavior before
            this parameter existed).
        callback_group: Callback group for the action server (default: a new
            ``ReentrantCallbackGroup``, since ``execute_fn`` runs for the
            duration of the move and must not block other callbacks).
    """

    def __init__(self, node: Node, action_name: str, execute_fn,
                 expected_frame: str = "", callback_group=None) -> None:
        self._node = node
        self._execute_fn = execute_fn
        self._expected_frame = expected_frame
        # Only one execute_fn may run at a time: it drives a shared
        # publisher/TF loop for the duration of the move, and this server's
        # own callback_group is Reentrant (so execute_fn doesn't block other
        # callbacks) -- without this flag, rclpy's ActionServer would happily
        # accept and run a second goal CONCURRENTLY with the first, and both
        # execute_fn calls would race on /gnc/trajectory_setpoint and TF. A
        # client that wants to switch targets must cancel the current goal
        # first (standard action cancel API); this server does not preempt.
        self._executing = False
        self._server = ActionServer(
            node,
            CtlCommand,
            action_name,
            execute_callback=self._on_execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=callback_group or ReentrantCallbackGroup(),
        )
        node.get_logger().info(
            "[CtlCommandActionServer] serving %s" % action_name
        )

    def _on_goal(self, goal_request) -> GoalResponse:
        if self._executing:
            self._node.get_logger().warn(
                "[CtlCommandActionServer] rejecting goal: a move is already "
                "in progress -- cancel it first"
            )
            return GoalResponse.REJECT
        if goal_request.type.type != CtlStatusType.MOVE_TO_ABSOLUTE_TARGET:
            self._node.get_logger().warn(
                "[CtlCommandActionServer] rejecting goal: unsupported "
                "CtlStatusType.type=%d (only MOVE_TO_ABSOLUTE_TARGET=%d is "
                "served)" % (goal_request.type.type,
                             CtlStatusType.MOVE_TO_ABSOLUTE_TARGET)
            )
            return GoalResponse.REJECT
        frame = goal_request.target.header.frame_id
        if self._expected_frame and frame and frame != self._expected_frame:
            self._node.get_logger().warn(
                "[CtlCommandActionServer] rejecting goal: target frame '%s' "
                "!= expected '%s'" % (frame, self._expected_frame)
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _on_execute(self, goal_handle):
        target = goal_handle.request.target.pose
        p_target = [target.position.x, target.position.y, target.position.z]
        q_target = [target.orientation.x, target.orientation.y,
                    target.orientation.z, target.orientation.w]

        def feedback_cb(time_to_go_sec, p_to_go, q_to_go):
            feedback = CtlCommand.Feedback()
            feedback.time_to_go = _seconds_to_duration(time_to_go_sec)
            (feedback.pose_to_go.position.x, feedback.pose_to_go.position.y,
             feedback.pose_to_go.position.z) = p_to_go
            (feedback.pose_to_go.orientation.x, feedback.pose_to_go.orientation.y,
             feedback.pose_to_go.orientation.z, feedback.pose_to_go.orientation.w) = q_to_go
            goal_handle.publish_feedback(feedback)

        self._executing = True
        try:
            status = self._execute_fn(
                p_target, q_target, feedback_cb,
                lambda: goal_handle.is_cancel_requested,
            )
        finally:
            self._executing = False

        result = CtlCommand.Result()
        result.stamp = self._node.get_clock().now().to_msg()
        result.type = status
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif status == TERMINATE_SUCCESS:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result


def main(args=None) -> None:
    """Standalone manual test: serve goals with a trivial fake execute_fn.

    The fake ``execute_fn`` counts down a fixed duration, reporting
    decreasing ``time_to_go`` and a ``pose_to_go`` that shrinks linearly to
    zero, then reports success. Not a real controller -- exercises only the
    action-server plumbing (feedback, cancellation, result).

    Run with ``ros2 run sobits_intball2_gnc ctl_command_action_server [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="ctl_command_action_server",
        description="Manual test for CtlCommandActionServer: serve goals "
                    "with a fake, fixed-duration execute_fn.",
    )
    parser.add_argument("--action-name", default="/gnc/move_to_test",
                        help="action server name (default: %(default)s)")
    parser.add_argument("--move-duration", type=float, default=5.0,
                        metavar="SEC",
                        help="fake move duration in seconds (default: 5.0)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    def fake_execute_fn(p_target, q_target, feedback_cb, is_cancel_requested):
        start = time.monotonic()
        end = start + ns.move_duration
        while time.monotonic() < end:
            if is_cancel_requested():
                return TERMINATE_ABORTED
            remaining = end - time.monotonic()
            fraction = max(0.0, remaining / ns.move_duration)
            feedback_cb(
                remaining,
                [c * fraction for c in p_target],
                q_target,
            )
            time.sleep(0.2)
        return TERMINATE_SUCCESS

    rclpy.init(args=argv)
    node = Node("ctl_command_action_server_test")
    CtlCommandActionServer(node, ns.action_name, fake_execute_fn)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
