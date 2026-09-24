#!/usr/bin/env python3
"""Publishes an OctoMap .bt (default: the JEM crop, maps/jem_octomap.bt) once as a
latched RViz cube-list Marker in iss_body."""
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

import minco_native_py


class OctomapMarkerPublisher(Node):
    def __init__(self):
        super().__init__('octomap_marker_publisher')
        default_map = os.path.join(
            get_package_share_directory('sobits_intball2_gnc'), 'maps', 'jem_octomap.bt')
        map_file = self.declare_parameter('map_file', default_map).value
        frame_id = self.declare_parameter('frame_id', 'iss_body').value
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(Marker, 'octomap_marker', qos)
        resolution, flat = minco_native_py.load_octomap_points(map_file)
        points = np.asarray(flat).reshape(-1, 3)
        self._pub.publish(self._marker(points, resolution, frame_id))
        self.get_logger().info(
            f"Published {len(points)} voxels ({resolution} m) of {map_file} in {frame_id}")

    @staticmethod
    def _marker(points, resolution, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        # Stamp 0 with frame_locked: RViz re-renders in the latest iss_body pose as the ISS moves.
        marker.frame_locked = True
        marker.ns = 'octomap'
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(resolution)
        z = points[:, 2]
        heights = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
        marker.points = [Point(x=float(x), y=float(y), z=float(zz)) for x, y, zz in points]
        marker.colors = [ColorRGBA(r=float(h), g=0.4, b=float(1.0 - h), a=0.6) for h in heights]
        return marker


def main():
    rclpy.init()
    node = OctomapMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
