#!/usr/bin/env python3
"""MultiDOFJointTrajectory publisher for IntBall2 (`/gnc/trajectory_setpoint`).

ROS I/O wrapper (does not subclass Node): publishes a single-point moving
target (``trajectory_msgs/MultiDOFJointTrajectory``, ``points[0]`` only) for
Control's ``MultiDOFJointTrajectorySubscriber`` to consume (Phase 3a contract,
see ``docs/archive/achieved/phase3a_interface_contract.md``).

This is the producer side of that contract. Phase 3a itself was verified with
a stand-in script publishing directly (``test/manual/``); this wrapper is the
reusable form a Guidance node samples its trajectory into, once one exists
(``docs/main_plan.md`` Phase 2, ``docs/future_design_notes.md`` 3-1).

Continuous republishing (at the control rate, one message per
``ControlNode`` tick) is the caller's responsibility -- this wrapper only
converts one ``(p, v, a, q)`` sample into a message and publishes it.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Transform, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

TRAJECTORY_TOPIC = "/gnc/trajectory_setpoint"
DEFAULT_REFERENCE_FRAME = "iss_body"

# This package's publishers default to reliable (subscribers default to
# best-effort/sensor_data instead, see docs/future_design_notes.md 6-4).
DEFAULT_QOS = QoSProfile(
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


class MultiDOFJointTrajectoryPublisher:
    """Publish a single-point ``MultiDOFJointTrajectory`` setpoint.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: Trajectory setpoint topic name.
        reference_frame: ``frame_id`` stamped on the message (must match
            ``MultiDOFJointTrajectorySubscriber``'s ``expected_frame`` on the
            Control side, i.e. ``iss_body``).
        qos_profile: QoS for the publisher (default: best-effort, see
            ``DEFAULT_QOS``).
    """

    def __init__(self, node: Node, topic: str = TRAJECTORY_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 qos_profile: QoSProfile = DEFAULT_QOS) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._pub = node.create_publisher(
            MultiDOFJointTrajectory, topic, qos_profile
        )
        node.get_logger().info(
            "[MultiDOFJointTrajectoryPublisher] publishing to %s (frame: %s)"
            % (topic, reference_frame)
        )

    def publish(self, p_des, v_des, a_des, q_des=_IDENTITY_QUAT) -> None:
        """Publish one setpoint sample.

        Args:
            p_des: Desired position ``[x, y, z]``.
            v_des: Desired velocity ``[x, y, z]``.
            a_des: Desired acceleration ``[x, y, z]``.
            q_des: Desired orientation ``[x, y, z, w]``. Unused by Phase 3a's
                translation-only trajectory_controller (identity by default);
                exposed for Phase 3b's attitude tracking.
        """
        transform = Transform()
        transform.translation.x, transform.translation.y, transform.translation.z = p_des
        (transform.rotation.x, transform.rotation.y,
         transform.rotation.z, transform.rotation.w) = q_des

        velocity = Twist()
        velocity.linear.x, velocity.linear.y, velocity.linear.z = v_des

        accel = Twist()
        accel.linear.x, accel.linear.y, accel.linear.z = a_des

        point = MultiDOFJointTrajectoryPoint()
        point.transforms = [transform]
        point.velocities = [velocity]
        point.accelerations = [accel]
        point.time_from_start = Duration()

        msg = MultiDOFJointTrajectory()
        msg.header.frame_id = self._reference_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.points = [point]
        self._pub.publish(msg)


def main(args=None) -> None:
    """Standalone manual test: publish a fixed setpoint at 50 Hz.

    Run with ``ros2 run sobits_intball2_gnc multi_dof_joint_trajectory_publisher [options]``.
    """
    import argparse
    import sys
    import time
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="multi_dof_joint_trajectory_publisher",
        description="Manual test for MultiDOFJointTrajectoryPublisher: "
                    "publish a fixed p_des/v_des/a_des setpoint at 50 Hz.",
    )
    parser.add_argument("--topic", default=TRAJECTORY_TOPIC,
                        help="setpoint topic (default: %(default)s)")
    parser.add_argument("--frame", default=DEFAULT_REFERENCE_FRAME,
                        help="reference frame (default: %(default)s)")
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"), help="p_des (default: 0 0 0)")
    parser.add_argument("--duration", type=float, default=5.0, metavar="SEC",
                        help="publish duration in seconds (default: 5.0)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("multi_dof_joint_trajectory_publisher_test")
    pub = MultiDOFJointTrajectoryPublisher(node, ns.topic, ns.frame)
    end = time.monotonic() + ns.duration
    try:
        while rclpy.ok() and time.monotonic() < end:
            pub.publish(ns.pos, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            rclpy.spin_once(node, timeout_sec=1.0 / 50.0)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
