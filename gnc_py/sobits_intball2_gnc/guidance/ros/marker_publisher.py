#!/usr/bin/env python3
"""Marker publisher for IntBall2 (depth-seen occupied cells, `/guidance/depth_occupied`).

ROS I/O wrapper (does not subclass Node): publishes grid cell centres as one
``visualization_msgs/Marker`` ``CUBE_LIST``. Named after its message type, like
``marker_array_publisher.py``.
"""
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

DEPTH_OCCUPIED_TOPIC = "/guidance/depth_occupied"
DEFAULT_REFERENCE_FRAME = "iss_body"


class MarkerPublisher:
    """Publish cell centres ``(N, 3)`` as cubes of edge ``cell_size``.

    Args:
        node: The rclpy Node that owns this publisher.
        cell_size: Cube edge [m] (the grid resolution).
        topic: Marker topic name.
        reference_frame: ``frame_id`` stamped on the marker.
        rgba: Cube color.
    """

    def __init__(self, node: Node, cell_size: float, topic: str = DEPTH_OCCUPIED_TOPIC,
                 reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 rgba=(0.1, 0.6, 1.0, 0.6)) -> None:
        self._node = node
        self._cell_size = float(cell_size)
        self._reference_frame = reference_frame
        self._color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
        self._pub = node.create_publisher(Marker, topic, 1)

    def publish(self, centers) -> None:
        marker = Marker()
        marker.header.frame_id = self._reference_frame
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = "depth_occupied"
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = self._cell_size
        marker.color = self._color
        marker.points = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in centers]
        self._pub.publish(marker)
