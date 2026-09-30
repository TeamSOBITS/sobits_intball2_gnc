#!/usr/bin/env python3
"""Show a candidate global-A* search box in RViz; it does not plan or command motion.

Run after the usual RViz/TF launch, for example:

  ros2 run sobits_intball2_gnc octomap_marker_publisher
  python3 gnc_py/test/manual/publish_global_astar_bounds.py \
    --ros-args -p lower:='[9.6, -11.9, 3.6]' -p upper:='[12.3, -2.4, 6.0]'

Add the ``/global_astar/search_bounds`` Marker topic in RViz.  The marker is
only a visual candidate boundary; it does not modify the occupancy map or the
planner.
"""
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker


class GlobalAstarBoundsPublisher(Node):
    def __init__(self):
        super().__init__('global_astar_bounds_publisher')
        lower = self.declare_parameter('lower', [9.6, -11.9, 3.6]).value
        upper = self.declare_parameter('upper', [12.3, -2.4, 6.0]).value
        frame_id = self.declare_parameter('frame_id', 'iss_body').value
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("parameters 'lower' and 'upper' must each have three values")
        if any(float(lo) >= float(hi) for lo, hi in zip(lower, upper)):
            raise ValueError("each lower coordinate must be less than upper")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(Marker, '/global_astar/search_bounds', qos)
        self._marker_msg = self._marker(lower, upper, frame_id)
        # RViz's Marker display commonly requests VOLATILE QoS, so a display
        # added after this node starts would miss a publish-once marker.
        self._pub.publish(self._marker_msg)
        self._timer = self.create_timer(1.0, self._publish)
        self.get_logger().info(
            'Published global-A* candidate bounds in %s: lower=%s upper=%s'
            % (frame_id, list(lower), list(upper)))

    def _publish(self):
        self._pub.publish(self._marker_msg)

    @staticmethod
    def _marker(lower, upper, frame_id):
        lo = [float(v) for v in lower]
        hi = [float(v) for v in upper]
        corners = [
            (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
            (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
            (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
            (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
        ]
        edges = ((0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.frame_locked = True
        marker.ns = 'global_astar'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03
        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 0.95
        marker.points = [
            Point(x=corners[index][0], y=corners[index][1], z=corners[index][2])
            for edge in edges for index in edge
        ]
        return marker


def main():
    rclpy.init()
    node = GlobalAstarBoundsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
