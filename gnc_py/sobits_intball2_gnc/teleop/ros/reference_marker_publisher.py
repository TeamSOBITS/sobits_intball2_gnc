#!/usr/bin/env python3
"""Reference-pose marker for RViz (`/gnc/teleop/reference_marker`).

ROS I/O wrapper (does not subclass Node): a sphere plus a forward arrow at the teleop reference, so
the operator sees where the vehicle is being asked to go. Coloured by status.
"""
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

REFERENCE_MARKER_TOPIC = "/gnc/teleop/reference_marker"
DEFAULT_REFERENCE_FRAME = "iss_body"
ARROW_LENGTH = 0.3


class ReferenceMarkerPublisher:
    def __init__(self, node: Node, topic: str = REFERENCE_MARKER_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME) -> None:
        self._node = node
        self._frame = reference_frame
        self._pub = node.create_publisher(MarkerArray, topic, 1)

    def _marker(self, marker_id, kind, rgba):
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = "teleop_reference"
        m.id = marker_id
        m.type = kind
        m.color.r, m.color.g, m.color.b, m.color.a = (float(x) for x in rgba)
        return m

    def publish(self, p, q, rgba=(0.1, 0.9, 0.2, 0.8)) -> None:
        sphere = self._marker(0, Marker.SPHERE, rgba)
        arrow = self._marker(1, Marker.ARROW, rgba)
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.12
        arrow.scale.x, arrow.scale.y, arrow.scale.z = ARROW_LENGTH, 0.03, 0.03
        for m in (sphere, arrow):
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(x) for x in p)
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = (
                float(x) for x in q)
        self._pub.publish(MarkerArray(markers=[sphere, arrow]))

    def clear(self) -> None:
        m = Marker()
        m.header.frame_id = self._frame
        m.action = Marker.DELETEALL
        self._pub.publish(MarkerArray(markers=[m]))
