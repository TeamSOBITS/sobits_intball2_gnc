#!/usr/bin/env python3
"""TF-based self-position client for IntBall2.

ROS I/O wrapper (does not subclass Node): looks up the vehicle pose from the TF
tree (``iss_body`` <- ``body``). Unlike the topic subscribers in this package,
TF is a **pull** source: the control loop asks for the latest buffered transform
each tick instead of being handed samples by a callback.

Two consequences shape this wrapper:

- Lookups use a **zero wait time**. The ``/tf`` subscription that fills the
  buffer runs on the same single-threaded executor as the control timer, so a
  blocking lookup would prevent the very callbacks that could satisfy it.
- ``get_pose`` returns the transform's header timestamp alongside the pose. A
  lookup keeps succeeding from the buffer after the publisher stops, so the
  caller needs the timestamp to tell a live pose from a frozen one. The
  timestamp is returned as-is (it may be on a simulation clock unrelated to this
  process's clock) and is only meaningful when compared to another timestamp
  from this same source.

Shared between ``control`` (position/attitude feedback) and ``guidance``
(Action Server feedback, e.g. ``pose_to_go``): the ``/tf_static`` race
workaround below must stay in one place, not be duplicated per package (see
``docs/archive/achieved/tf_race_investigation.md``).
"""
import time

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import tf2_ros
from tf2_msgs.msg import TFMessage
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException

DEFAULT_REFERENCE_FRAME = "iss_body"
DEFAULT_TARGET_FRAME = "body"

# Lookups from the control loop must not wait: the /tf callbacks that fill the
# buffer share this node's executor thread with the control timer.
_NO_WAIT = rclpy.duration.Duration(seconds=0)

# Default QoS for the /tf subscription: best-effort (this package's default
# for streaming state topics), volatile, with enough depth to buffer a burst
# of transforms without dropping the ones the control loop needs.
DEFAULT_QOS = QoSProfile(
    depth=100,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class TfClient:
    """Look up the body pose from the TF tree.

    Args:
        node: The rclpy Node that owns this client.
        reference_frame: Parent/reference frame (default ``iss_body``).
        target_frame: Child/body frame (default ``body``).
        qos_profile: QoS for the ``/tf`` subscription (default: best-effort,
            see ``DEFAULT_QOS``).
    """

    def __init__(
        self,
        node: Node,
        reference_frame: str = DEFAULT_REFERENCE_FRAME,
        target_frame: str = DEFAULT_TARGET_FRAME,
        qos_profile: QoSProfile = DEFAULT_QOS,
    ) -> None:
        self._node = node
        self.reference_frame = reference_frame
        self.target_frame = target_frame
        self._buffer = tf2_ros.Buffer()
        # Deliberately does NOT use tf2_ros.TransformListener, which also
        # subscribes to /tf_static. The ROS1-side robot_state_publisher
        # latches a one-time identity transform for base->body onto
        # /tf_static (its URDF declares that edge as fixed; the vehicle's
        # real pose is published separately, dynamically, onto /tf for the
        # same edge). Because /tf_static uses transient_local durability, a
        # freshly created listener can pick up that frozen identity sample
        # before the first real /tf sample arrives, and lookup_transform then
        # returns it without raising. Every edge this client needs
        # (base->body, base->iss_body) is also published dynamically on
        # /tf, so /tf_static is not needed here.
        self._tf_sub = node.create_subscription(
            TFMessage, "/tf", self._on_tf, qos_profile
        )
        self._ready = False
        node.get_logger().info(
            "[TfClient] looking up %s <- %s" % (reference_frame, target_frame)
        )

    def _on_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            self._buffer.set_transform(transform, "tf_client")

    @property
    def ready(self) -> bool:
        """True once at least one transform has been looked up successfully."""
        return self._ready

    def get_transform(self, target_frame=None, source_frame=None):
        """Return the latest buffered ``source_frame -> target_frame`` or None.

        Never waits: an unavailable transform is reported immediately.
        """
        target = target_frame or self.reference_frame
        source = source_frame or self.target_frame
        try:
            return self._buffer.lookup_transform(
                target, source, rclpy.time.Time(), timeout=_NO_WAIT
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self._node.get_logger().debug(
                "[TfClient] lookup %s <- %s failed: %s" % (target, source, exc)
            )
            return None

    def get_pose(self):
        """Return ``(pos, quat, stamp)`` in the reference frame, or None.

        ``stamp`` is the transform's header timestamp in seconds, on whatever
        clock the TF publisher uses. Compare it only against other stamps from
        this method -- never against this process's clock.
        """
        t = self.get_transform()
        if t is None:
            return None
        tr = t.transform.translation
        q = t.transform.rotation
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        self._ready = True
        return [tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w], stamp

    def wait_for_frame(self, timeout_sec: float = 5.0) -> bool:
        """Spin until the configured frames are connected, or timeout.

        Called at startup, before the control loop runs. On failure the frames
        actually present in the TF tree are logged, so a misspelled frame
        parameter is diagnosable from the launch output alone.
        """
        deadline = time.monotonic() + timeout_sec
        self._node.get_logger().info(
            "[TfClient] waiting up to %.1fs for %s <- %s"
            % (timeout_sec, self.reference_frame, self.target_frame)
        )
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._buffer.can_transform(
                self.reference_frame, self.target_frame,
                rclpy.time.Time(), timeout=_NO_WAIT,
            ):
                self._node.get_logger().info(
                    "[TfClient] %s <- %s is available"
                    % (self.reference_frame, self.target_frame)
                )
                return True
        self._node.get_logger().error(
            "[TfClient] %s <- %s not available within %.1fs. "
            "Frames present in the TF tree:\n%s"
            % (self.reference_frame, self.target_frame, timeout_sec,
               self._buffer.all_frames_as_string() or "(none)")
        )
        return False


def main(args=None) -> None:
    """Standalone manual test: look up and print the body pose from TF.

    Run with ``ros2 run sobits_intball2_gnc tf_client [options]``.
    """
    import argparse
    import sys
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="tf_client",
        description="Manual test for TfClient: print the body pose "
                    "(reference_frame <- target_frame) and its TF timestamp.",
    )
    parser.add_argument("--reference-frame", default=DEFAULT_REFERENCE_FRAME,
                        help="parent/reference frame (default: %(default)s)")
    parser.add_argument("--target-frame", default=DEFAULT_TARGET_FRAME,
                        help="child/body frame (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=0.0, metavar="SEC",
                        help="seconds to run; 0 runs until Ctrl-C (default: 0)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("tf_client_test")
    client = TfClient(node, ns.reference_frame, ns.target_frame)
    client.wait_for_frame(timeout_sec=5.0)
    end = None if ns.duration <= 0 else time.monotonic() + ns.duration
    try:
        while rclpy.ok() and (end is None or time.monotonic() < end):
            rclpy.spin_once(node, timeout_sec=0.5)
            pose = client.get_pose()
            if pose is None:
                node.get_logger().warn("[TfClient] no transform available")
            else:
                pos, quat, stamp = pose
                node.get_logger().info(
                    "[TfClient] stamp=%.3f pos=%s quat=%s" % (stamp, pos, quat)
                )
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
