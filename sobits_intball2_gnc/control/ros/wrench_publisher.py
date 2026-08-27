#!/usr/bin/env python3
"""Requested-wrench publisher for IntBall2 (`/ctl/wrench`).

ROS I/O wrapper (does not subclass Node): attaches a ``/ctl/wrench`` publisher
to the node passed in and turns a body-frame (force, torque) pair into a
``geometry_msgs/WrenchStamped``. This publishes the *requested* wrench (the
pre-allocation, pre-clamp value ``HoverController.last_force_raw``/
``last_torque_raw`` -- see docs/main_plan.md "[C] Controller内部値の可観測性
強化"), not the realized one: the realized per-fan output is already
observable via ``/ctl/duty``, but nothing published the requested wrench
itself before this, which delayed root-causing a replanning attitude
degradation (docs/archive/achieved/
2026-08-25_guidance_attitude_saturation_investigation.md). The topic is
already registered in the ROS1<->ROS2 bridge's ``bridge_topics.yaml`` with
this exact message type, but neither side had ever actually published to it.
"""
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile

WRENCH_TOPIC = "/ctl/wrench"
# (IMU-law + correction), summed and clamped -- the exact wrench passed to
# ThrustAllocator.allocate() each tick, as opposed to WRENCH_TOPIC's
# correction-only value. See docs/2026-08-27_thrust_allocator_single_axis_
# saturation_findings.md.
WRENCH_TOTAL_TOPIC = "/ctl/wrench_total"
# The wrench actually realized by this tick's duties (ThrustAllocator.
# achieved_wrench()), published from the same tick as WRENCH_TOTAL_TOPIC so
# the two can be compared without cross-topic staleness. See docs/
# 2026-08-27_thrust_allocator_single_axis_saturation_findings.md.
WRENCH_ACHIEVED_TOPIC = "/ctl/wrench_achieved"
DEFAULT_QOS = QoSProfile(depth=1)


class WrenchPublisher:
    """Publish a body-frame (force, torque) pair to ``/ctl/wrench``.

    Args:
        node: The rclpy Node that owns this publisher.
        frame_id: TF frame the force/torque are expressed in (default:
            ``body``, matching ``TrajectoryController``/``PoseCorrector``'s
            body-frame convention).
        qos_profile: QoS for the publisher (default: depth 1, matching
            ``FanDutyPublisher``).
    """

    def __init__(self, node: Node, frame_id: str = "body",
                 qos_profile: QoSProfile = DEFAULT_QOS,
                 topic: str = WRENCH_TOPIC) -> None:
        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(WrenchStamped, topic, qos_profile)

    def publish(self, force, torque, stamp=None) -> None:
        """Publish ``force``/``torque`` (each a 3-element sequence [N]/[N*m])."""
        msg = WrenchStamped()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = stamp if stamp is not None else self._node.get_clock().now().to_msg()
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = force
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = torque
        self._pub.publish(msg)
