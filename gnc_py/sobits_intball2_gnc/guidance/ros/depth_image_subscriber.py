#!/usr/bin/env python3
"""Depth image subscriber for IntBall2 (`/virtual_camera/<camera>/depth` + `camera_info`).

ROS I/O wrapper (does not subclass Node): hands each 32FC1 depth frame (REP 117:
``-inf`` too close, ``+inf`` nothing within range) to a caller-supplied callback
together with its intrinsics and the optical frame's pose in the reference frame
at the image stamp. The vehicle pose comes from :class:`TfClient` (``/tf`` only,
see its ``/tf_static`` race note); the camera mount ``body -> optical`` is static
and comes from a separate ``/tf_static`` listener, which that race does not affect.
"""
import numpy as np
import rclpy.time
import tf2_ros
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException

DEFAULT_DEPTH_TOPIC = "/virtual_camera/main/depth"
# /tf is serviced on another executor thread and can land after the image it stamped.
POSE_RETRY_PERIOD_S = 0.02
POSE_WAIT_S = 0.5
STATS_PERIOD_S = 10.0
TF_STATIC_QOS = QoSProfile(
    depth=100,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


def _seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _transform_to_rt(t):
    q = t.transform.rotation
    p = t.transform.translation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z])


class DepthImageSubscriber:
    """Call ``callback(depth, fx, fy, cx, cy, rotation, origin, stamp)`` per frame.

    ``depth`` is a float32 ``(H, W)`` array, ``rotation``/``origin`` the optical
    frame in ``tf_client.reference_frame``, ``stamp`` the image stamp in seconds.

    Args:
        node: The rclpy Node that owns the subscriptions.
        tf_client: :class:`TfClient` for the vehicle pose.
        callback: Called once per usable frame.
        depth_topic: Image topic; ``camera_info`` is its sibling topic.
        callback_group: Passed to ``create_subscription``.
    """

    def __init__(self, node: Node, tf_client, callback, depth_topic: str = DEFAULT_DEPTH_TOPIC,
                 callback_group=None) -> None:
        self._node = node
        self._tf = tf_client
        self._callback = callback
        self._info = None
        self._mount = None
        self._pending = None
        self._integrated = 0
        self._dropped = 0
        self._static_buffer = tf2_ros.Buffer()
        self._static_sub = node.create_subscription(
            TFMessage, "/tf_static", self._on_tf_static, TF_STATIC_QOS, callback_group=callback_group)
        info_topic = depth_topic.rsplit("/", 1)[0] + "/camera_info"
        self._info_sub = node.create_subscription(
            CameraInfo, info_topic, self._on_info, qos_profile_sensor_data, callback_group=callback_group)
        self._depth_sub = node.create_subscription(
            Image, depth_topic, self._on_depth, qos_profile_sensor_data, callback_group=callback_group)
        self._retry_timer = node.create_timer(POSE_RETRY_PERIOD_S, self._retry_pending,
                                              callback_group=callback_group)
        self._stats_timer = node.create_timer(STATS_PERIOD_S, self._log_stats,
                                              callback_group=callback_group)

    def _on_tf_static(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._static_buffer.set_transform_static(transform, "depth_image_subscriber")

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg

    def _camera_mount(self, optical_frame):
        if self._mount is None or self._mount[0] != optical_frame:
            try:
                t = self._static_buffer.lookup_transform(self._tf.target_frame, optical_frame,
                                                         rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                self._node.get_logger().warn(
                    "[DepthImageSubscriber] no static %s <- %s yet: %s"
                    % (self._tf.target_frame, optical_frame, exc), throttle_duration_sec=5.0)
                return None
            self._mount = (optical_frame,) + _transform_to_rt(t)
        return self._mount[1], self._mount[2]

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self._node.get_logger().warn(
                "[DepthImageSubscriber] ignoring encoding '%s' (expected 32FC1)" % msg.encoding,
                throttle_duration_sec=5.0)
            return
        info = self._info
        if info is None:
            return
        mount = self._camera_mount(msg.header.frame_id)
        if mount is None:
            return
        if self._pending is not None:
            self._drop("superseded by a newer frame")
        self._pending = (msg, info, mount)
        self._retry_pending()

    def _retry_pending(self) -> None:
        if self._pending is None:
            return
        msg, info, mount = self._pending
        body = self._tf.get_transform(stamp=msg.header.stamp)
        if body is None:
            latest = self._tf.get_transform()
            stamp = _seconds(msg.header.stamp)
            if latest is not None and _seconds(latest.header.stamp) - stamp > POSE_WAIT_S:
                self._drop("no vehicle pose at the image stamp")
            return
        self._pending = None
        r_body, p_body = _transform_to_rt(body)
        r_mount, p_mount = mount
        rows = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        depth = np.ascontiguousarray(rows[:, :msg.width])
        self._callback(depth, info.k[0], info.k[4], info.k[2], info.k[5],
                       r_body @ r_mount, p_body + r_body @ p_mount, _seconds(msg.header.stamp))
        self._integrated += 1

    def _drop(self, reason) -> None:
        self._pending = None
        self._dropped += 1
        self._node.get_logger().warn("[DepthImageSubscriber] frame dropped: %s" % reason,
                                     throttle_duration_sec=5.0)

    def _log_stats(self) -> None:
        self._node.get_logger().info(
            "[DepthImageSubscriber] last %.0fs: %d frames integrated, %d dropped"
            % (STATS_PERIOD_S, self._integrated, self._dropped))
        self._integrated = self._dropped = 0
