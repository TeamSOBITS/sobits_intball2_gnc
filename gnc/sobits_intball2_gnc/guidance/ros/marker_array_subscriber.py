#!/usr/bin/env python3
"""MarkerArray subscriber for IntBall2 (virtual obstacles, `/guidance/virtual_obstacles`).

ROS I/O wrapper (does not subclass Node): receives axis-aligned boxes as
``visualization_msgs/MarkerArray`` ``CUBE`` markers and forwards the parsed
edits to a caller-supplied callback (guidance_node's ``ObstacleMap``). Each
marker is one box keyed by ``(ns, id)``: ``pose.position`` is its center,
``scale`` its full size (orientation is ignored). ``ADD``/``MODIFY`` set a box,
``DELETE`` drops one, ``DELETEALL`` drops all. The topic is the single entry
point for obstacles not in the static map, so a perception source can later
publish the same message instead of ``test/manual/virtual_obstacle.py``.

Each marker's ``frame_id`` is validated against the expected reference frame,
like ``control/ros/pose_array_subscriber.py``. Named after its message type,
not the "virtual obstacle" role, following the same convention.
"""
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

VIRTUAL_OBSTACLES_TOPIC = "/guidance/virtual_obstacles"
DEFAULT_REFERENCE_FRAME = "iss_body"

# Transient local so a late-starting subscriber still gets the last message.
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


class MarkerArraySubscriber:
    """Parse ``CUBE`` markers into box edits for ``callback(clear, set_boxes, remove)``.

    Args:
        node: The rclpy Node that owns this subscription.
        callback: Called once per message with ``clear`` (bool),
            ``set_boxes`` (``{(ns, id): (center, half_extents)}``) and
            ``remove`` (list of ``(ns, id)``), to be applied in that order.
        topic: MarkerArray topic name.
        expected_frame: Markers in any other ``frame_id`` are ignored.
        callback_group: Passed to ``create_subscription``.
    """

    def __init__(self, node: Node, callback, topic: str = VIRTUAL_OBSTACLES_TOPIC,
                 expected_frame: str = DEFAULT_REFERENCE_FRAME,
                 callback_group=None) -> None:
        self._node = node
        self._callback = callback
        self._expected_frame = expected_frame
        self._sub = node.create_subscription(
            MarkerArray, topic, self._on_markers, LATCHED_QOS, callback_group=callback_group)

    def _on_markers(self, msg: MarkerArray) -> None:
        clear, set_boxes, remove = False, {}, []
        for marker in msg.markers:
            key = (marker.ns, marker.id)
            if marker.action == Marker.DELETEALL:
                clear, set_boxes, remove = True, {}, []
            elif marker.action == Marker.DELETE:
                set_boxes.pop(key, None)
                remove.append(key)
            elif marker.type != Marker.CUBE:
                self._node.get_logger().warn(
                    "[MarkerArraySubscriber] ignoring non-CUBE marker %s" % (key,))
            elif marker.header.frame_id != self._expected_frame:
                self._node.get_logger().warn(
                    "[MarkerArraySubscriber] ignoring marker %s in frame '%s' (expected '%s')"
                    % (key, marker.header.frame_id, self._expected_frame))
            else:
                p, s = marker.pose.position, marker.scale
                set_boxes[key] = ((p.x, p.y, p.z), (s.x / 2.0, s.y / 2.0, s.z / 2.0))
                if key in remove:
                    remove.remove(key)
        self._callback(clear, set_boxes, remove)
