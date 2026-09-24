#!/usr/bin/env python3
"""MarkerArray publisher for IntBall2 (active obstacles, `/guidance/obstacles_active`).

ROS I/O wrapper (does not subclass Node): publishes a set of axis-aligned
boxes as ``visualization_msgs/MarkerArray`` ``CUBE`` markers, replacing
whatever was published before (a leading ``DELETEALL``). Transient local, so
RViz started later still shows the current set. Named after its message type,
like ``marker_array_subscriber.py``.
"""
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

ACTIVE_OBSTACLES_TOPIC = "/guidance/obstacles_active"
DEFAULT_REFERENCE_FRAME = "iss_body"

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


class MarkerArrayPublisher:
    """Publish ``{(ns, id): (center, half_extents)}`` boxes as one MarkerArray.

    Args:
        node: The rclpy Node that owns this publisher.
        topic: MarkerArray topic name.
        reference_frame: ``frame_id`` stamped on every marker.
        rgba: Box color.
    """

    def __init__(self, node: Node, topic: str = ACTIVE_OBSTACLES_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 rgba=(1.0, 0.3, 0.0, 0.5)) -> None:
        self._node = node
        self._reference_frame = reference_frame
        self._rgba = tuple(float(c) for c in rgba)
        self._pub = node.create_publisher(MarkerArray, topic, LATCHED_QOS)

    def publish(self, boxes) -> None:
        msg = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self._reference_frame
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)
        stamp = self._node.get_clock().now().to_msg()
        for (ns, marker_id), (center, half) in sorted(boxes.items()):
            marker = Marker()
            marker.header.frame_id = self._reference_frame
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, center)
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = (float(2.0 * h) for h in half)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self._rgba
            marker.frame_locked = True
            msg.markers.append(marker)
        self._pub.publish(msg)
