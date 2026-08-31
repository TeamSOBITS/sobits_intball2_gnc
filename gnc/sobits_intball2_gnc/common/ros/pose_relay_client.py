#!/usr/bin/env python3
"""Lightweight self-position client for a dedicated pose-relay topic.

Companion to :class:`~sobits_intball2_gnc.common.ros.tf_client.TfClient`, for
the same ``iss_body`` <- ``body`` pose feedback, but reading a small,
throttled ``geometry_msgs/TransformStamped`` topic (default
``/gnc/body_pose_raw``) instead of the full ``/tf`` tree.

Why this exists (docs/recording_cpu_load_control_degradation.md): under CPU
contention in the ROS1<->ROS2 bridge container, the generic bridge's ``/tf``
conversion (every frame, ~1700Hz on the ROS1 side) degrades badly on the ROS2
side -- confirmed the bottleneck is bridge-side conversion/scheduling cost,
not the simulator's own publishing (which stays near-unaffected). Raising the
bridge's scheduling priority, splitting it into a dedicated process, and
CPU-pinning it all failed to fix this from inside the container (see the doc
for the full investigation). The fix that actually reduces load instead of
trying to buy more of it: a small ROS1-side relay node
(``/root/bridge/gnc_pose_relay.py``) does the ``iss_body -> body`` tf2 lookup
once, on the ROS1 side where the full-rate stream already lives cheaply, and
republishes just that one composed transform at a fixed, throttled rate
(default 50Hz, matching this control loop) -- a small fraction of the
message volume of forwarding the raw ``/tf`` firehose through the bridge.

This does NOT replace the general ``/tf`` bridge: other nodes (waypoint
registration, RViz, Guidance's arbitrary-frame lookups via
:class:`TfClient`) keep using it unchanged. This is an additional, narrow
channel for this specific feedback path only -- currently wired up in
``control.py``'s ``ControlNode`` alone.

Interface deliberately mirrors :class:`TfClient` (``get_pose()`` ->
``(pos, quat, stamp)`` or ``None``, ``wait_for_frame()``, ``ready``) so
:class:`~sobits_intball2_gnc.control.utils.hover_controller.HoverController`
and its ``PoseCorrector``/``TrajectoryController`` need no changes -- only the
object ``control.py`` injects as ``tf_client`` differs.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped

DEFAULT_TOPIC = "/gnc/body_pose_raw"

# Small, low-rate topic (default 50Hz from the ROS1-side relay) -- best-effort
# is fine, matching this package's other streaming state topics.
DEFAULT_QOS = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class PoseRelayClient:
    """Look up the body pose from the dedicated pose-relay topic.

    Args:
        node: The rclpy Node that owns this client.
        topic: Pose-relay topic name (default ``/gnc/body_pose_raw``).
        qos_profile: QoS for the subscription (default: best-effort, see
            ``DEFAULT_QOS``).
    """

    def __init__(
        self,
        node: Node,
        topic: str = DEFAULT_TOPIC,
        qos_profile: QoSProfile = DEFAULT_QOS,
    ) -> None:
        self._node = node
        self._latest = None  # (pos, quat, stamp)
        self._sub = node.create_subscription(
            TransformStamped, topic, self._on_transform, qos_profile
        )
        node.get_logger().info(
            "[PoseRelayClient] subscribing %s" % topic
        )

    def _on_transform(self, msg: TransformStamped) -> None:
        tr = msg.transform.translation
        q = msg.transform.rotation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest = ([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w], stamp)

    @property
    def ready(self) -> bool:
        """True once at least one pose has been received."""
        return self._latest is not None

    def get_pose(self):
        """Return ``(pos, quat, stamp)`` from the last received message, or None.

        ``stamp`` is the transform's original header timestamp in seconds
        (the ROS1-side relay forwards the tf2 lookup's own stamp, not its own
        publish time) -- same contract as
        :meth:`~sobits_intball2_gnc.common.ros.tf_client.TfClient.get_pose`:
        compare it only against other stamps from this method.
        """
        return self._latest

    def wait_for_frame(self, timeout_sec: float = 5.0) -> bool:
        """Spin until the first pose arrives, or timeout.

        Mirrors :meth:`TfClient.wait_for_frame` so ``control.py`` can use
        either client interchangeably at startup.
        """
        deadline = time.monotonic() + timeout_sec
        self._node.get_logger().info(
            "[PoseRelayClient] waiting up to %.1fs for the first pose"
            % timeout_sec
        )
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self.ready:
                self._node.get_logger().info("[PoseRelayClient] pose received")
                return True
        self._node.get_logger().error(
            "[PoseRelayClient] no pose received within %.1fs" % timeout_sec
        )
        return False
