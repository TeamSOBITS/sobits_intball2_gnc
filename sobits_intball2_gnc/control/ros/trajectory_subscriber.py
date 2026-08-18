#!/usr/bin/env python3
"""Trajectory setpoint subscriber for IntBall2 (`/gnc/trajectory_setpoint`).

ROS I/O wrapper (does not subclass Node): receives a single-point moving
target (``trajectory_msgs/MultiDOFJointTrajectory``, ``points[0]`` only,
expressed in the TF reference frame) from the Guidance node, and exposes the
latest ``p_des``/``v_des``/``a_des``/``q_des`` for the trajectory controller
(Phase 3a, see ``openspec/changes/add-trajectory-following``).

The array's ``frame_id`` is validated against the expected reference frame,
same policy as ``PathSubscriber``: a message in the wrong frame is discarded
wholesale rather than silently followed to the wrong place.

Unlike TF (a pull source), this is a push subscription: liveness is judged
by comparing the caller's monotonic clock against ``last_received_t``, not by
polling. The caller decides the staleness timeout (see
``trajectory_controller.timeout`` in ``config/gnc_params.yaml``) and computes
it against ``last_received_t``.
"""
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory

TRAJECTORY_TOPIC = "/gnc/trajectory_setpoint"


def frame_accepted(frame: str, expected_frame: str) -> bool:
    """True when a message's ``frame_id`` may be used.

    Pure predicate extracted out of the ROS callback so it is testable
    without starting rclpy (this package's convention: ROS-agnostic pure
    functions get unit tests, ROS I/O wrappers do not). An empty
    ``expected_frame`` accepts anything ("unspecified" caller); an empty
    incoming ``frame`` is also accepted (message didn't set one). Any other
    mismatch is rejected -- same policy as ``PathSubscriber``.
    """
    return not expected_frame or not frame or frame == expected_frame


class TrajectorySubscriber:
    """Subscribe to a single-point ``MultiDOFJointTrajectory`` setpoint.

    Args:
        node: The rclpy Node that owns this subscriber.
        topic: Trajectory setpoint topic name.
        expected_frame: Reference frame the setpoint must be expressed in. An
            empty ``frame_id`` is accepted as "unspecified"; any other
            mismatch rejects the whole message.
    """

    def __init__(self, node: Node, topic: str = TRAJECTORY_TOPIC,
                 expected_frame: str = "") -> None:
        self._node = node
        self._expected_frame = expected_frame
        self._p_des = None
        self._v_des = None
        self._a_des = None
        self._q_des = None
        self._last_received_t = None  # caller's monotonic clock, set on accept
        self._sub = node.create_subscription(
            MultiDOFJointTrajectory, topic, self._callback, 1
        )
        node.get_logger().info(
            "[TrajectorySubscriber] subscribing %s (frame: %s)"
            % (topic, expected_frame or "<any>")
        )

    def _callback(self, msg: MultiDOFJointTrajectory) -> None:
        frame = msg.header.frame_id
        if not frame_accepted(frame, self._expected_frame):
            # Reject wholesale: a setpoint in the wrong frame is worse than
            # sticking with the last valid one.
            self._node.get_logger().warn(
                "[TrajectorySubscriber] discarding setpoint in frame '%s': "
                "expected '%s'" % (frame, self._expected_frame)
            )
            return
        if not msg.points:
            self._node.get_logger().warn(
                "[TrajectorySubscriber] discarding setpoint with no points"
            )
            return

        point = msg.points[0]
        transform = point.transforms[0]
        velocity = point.velocities[0]
        accel = point.accelerations[0]

        self._p_des = [transform.translation.x, transform.translation.y,
                        transform.translation.z]
        self._q_des = [transform.rotation.x, transform.rotation.y,
                        transform.rotation.z, transform.rotation.w]
        self._v_des = [velocity.linear.x, velocity.linear.y, velocity.linear.z]
        self._a_des = [accel.linear.x, accel.linear.y, accel.linear.z]
        self._last_received_t = time.monotonic()

    @property
    def ready(self) -> bool:
        """True once at least one valid setpoint has been received."""
        return self._p_des is not None

    @property
    def p_des(self):
        """Latest desired position [x, y, z] (reference frame), or None."""
        return self._p_des

    @property
    def v_des(self):
        """Latest desired velocity [x, y, z] (reference frame), or None."""
        return self._v_des

    @property
    def a_des(self):
        """Latest desired acceleration [x, y, z] (reference frame), or None."""
        return self._a_des

    @property
    def q_des(self):
        """Latest desired orientation [x, y, z, w] (reference frame), or None.

        Unused by Phase 3a's translation-only trajectory_controller; exposed
        for Phase 3b's attitude tracking.
        """
        return self._q_des

    @property
    def last_received_t(self):
        """Monotonic time (this process's clock) of the last accepted setpoint, or None."""
        return self._last_received_t


def main(args=None) -> None:
    """Standalone manual test: report received trajectory setpoints.

    Run with ``ros2 run sobits_intball2_gnc trajectory_subscriber [options]``.
    """
    import argparse
    import sys
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="trajectory_subscriber",
        description="Manual test for TrajectorySubscriber: log setpoints "
                    "received on the trajectory topic.",
    )
    parser.add_argument("--topic", default=TRAJECTORY_TOPIC,
                        help="trajectory setpoint topic (default: %(default)s)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("trajectory_subscriber_test")
    sub = TrajectorySubscriber(node, ns.topic)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if sub.ready:
                node.get_logger().info(
                    f"[TrajectorySubscriber] p_des={sub.p_des} v_des={sub.v_des} "
                    f"a_des={sub.a_des} q_des={sub.q_des}"
                )
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
