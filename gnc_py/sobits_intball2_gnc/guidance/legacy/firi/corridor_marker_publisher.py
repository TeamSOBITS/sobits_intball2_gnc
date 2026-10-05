"""RViz-only publisher for an A* route's FIRI corridor half-spaces."""
import itertools

import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from sobits_intball2_gnc.guidance.ros.marker_array_publisher import LATCHED_QOS


class CorridorMarkerPublisher:
    """Publish each FIRI polytope as a translucent wireframe MarkerArray."""

    def __init__(self, node: Node, topic: str, reference_frame: str) -> None:
        self._node = node
        self._frame = reference_frame
        self._pub = node.create_publisher(MarkerArray, topic, LATCHED_QOS)

    def publish(self, planes) -> None:
        groups = {}
        values = list(planes)
        for offset in range(0, len(values), 5):
            groups.setdefault(int(values[offset]), []).append(values[offset + 1:offset + 5])
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)
        stamp = self._node.get_clock().now().to_msg()
        for segment, rows in sorted(groups.items()):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = self._frame, stamp
            marker.ns, marker.id = "firi_corridor", segment
            marker.type, marker.action, marker.scale.x = Marker.LINE_LIST, Marker.ADD, 0.012
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.1, 0.8, 0.9)
            marker.pose.orientation.w = 1.0
            for left, right in _edges(_vertices(rows)):
                marker.points.extend([_point(left[0]), _point(right[0])])
            msg.markers.append(marker)
        self._pub.publish(msg)


def _vertices(rows):
    """Vertices of n.x+b <= 0; retain their active planes to identify edges."""
    out = []
    for active in itertools.combinations(range(len(rows)), 3):
        matrix = np.asarray([rows[i][:3] for i in active], dtype=float)
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        point = np.linalg.solve(matrix, -np.asarray([rows[i][3] for i in active]))
        if all(np.dot(row[:3], point) + row[3] <= 1e-6 for row in rows):
            if not any(np.linalg.norm(point - old[0]) < 1e-6 for old in out):
                out.append((point, frozenset(active)))
    return out


def _edges(vertices):
    for left, right in itertools.combinations(vertices, 2):
        if len(left[1] & right[1]) >= 2:
            yield left, right


def _point(value):
    point = Point()
    point.x, point.y, point.z = map(float, value)
    return point
