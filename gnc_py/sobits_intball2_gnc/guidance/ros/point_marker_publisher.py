#!/usr/bin/env python3
"""Single-point sphere marker for RViz (visualization only).

ROS I/O wrapper (does not subclass Node). Used for the jaxa_rrt tracking point
(``/gnc/jaxa_tracking_point``): its ``p_des`` goes out on
``/gnc/trajectory_setpoint``, which RViz cannot display.
"""
from rclpy.node import Node
from visualization_msgs.msg import Marker

DEFAULT_REFERENCE_FRAME = "iss_body"


class PointMarkerPublisher:
    def __init__(self, node: Node, topic: str, reference_frame: str = DEFAULT_REFERENCE_FRAME,
                 rgba=(1.0, 0.1, 0.1, 0.9), diameter=0.08) -> None:
        self._node = node
        self._frame = reference_frame
        self._rgba = tuple(float(x) for x in rgba)
        self._diameter = float(diameter)
        self._pub = node.create_publisher(Marker, topic, 1)

    def publish(self, p) -> None:
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = "point"
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = self._diameter
        m.color.r, m.color.g, m.color.b, m.color.a = self._rgba
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(x) for x in p)
        m.pose.orientation.w = 1.0
        self._pub.publish(m)
