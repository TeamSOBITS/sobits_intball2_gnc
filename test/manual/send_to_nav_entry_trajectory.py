#!/usr/bin/env python3
"""Send a min-jerk trajectory to /gnc/trajectory_setpoint targeting the
absolute 'above_dock_2' named location (maps/iss_location.yaml), and publish
the visualization path via TrajectoryPathPublisher.

Difference from send_trajectory.py: absolute target (not a relative offset
from current position), and a unique node name per run (timestamp suffix) to
rule out any stale-discovery / node-name-reuse effect on the previous test.

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist

from sobits_intball2_gnc.control.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.trajectory_path_publisher import (
    TrajectoryPathPublisher,
)

TRAJECTORY_TOPIC = "/gnc/trajectory_setpoint"
PATH_SAMPLE_DT = 0.1
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 50.0
DURATION_SEC = 15.0

# nav_entry, from maps/iss_location.yaml (iss_body frame)
TARGET_POS = [11.0, -4.3, 5.0]
TARGET_QUAT = [-0.707106, 0.707106, 0.0, 0.0]  # x,y,z,w


def quintic(tau):
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def main():
    rclpy.init(args=sys.argv)
    node_name = f"send_to_nav_entry_trajectory_{int(time.monotonic() * 1000) % 1000000}"
    node = Node(node_name)
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    pub = node.create_publisher(MultiDOFJointTrajectory, TRAJECTORY_TOPIC, 1)
    path_pub = TrajectoryPathPublisher(node, reference_frame=REFERENCE_FRAME)

    node.get_logger().info(f"[{node_name}] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"[{node_name}] could not get a TF pose")
        return
    pos, quat, _ = tf_client.get_pose()
    node.get_logger().info(f"[{node_name}] TF pose acquired: {pos}")

    p0 = np.asarray(pos)
    q0 = quat
    p1 = np.asarray(TARGET_POS)
    q1 = TARGET_QUAT
    node.get_logger().info(
        f"[{node_name}] p0={p0.tolist()} -> p1={p1.tolist()} (above_dock_2) over {DURATION_SEC}s"
    )

    n_samples = int(DURATION_SEC / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = p0 + (p1 - p0) * s
        samples.append((p.tolist(), q0))
    path_pub.publish(samples)
    node.get_logger().info(
        f"[{node_name}] path sample bounds: min={samples[0][0]}, max_end={samples[-1][0]}"
    )

    dt = 1.0 / RATE_HZ
    t_start = time.monotonic()
    try:
        while rclpy.ok():
            elapsed = time.monotonic() - t_start
            tau = min(1.0, elapsed / DURATION_SEC)
            s, ds, dds = quintic(tau)
            p_des = p0 + (p1 - p0) * s
            v_des = (p1 - p0) * ds / DURATION_SEC
            a_des = (p1 - p0) * dds / (DURATION_SEC ** 2)

            msg = MultiDOFJointTrajectory()
            msg.header.frame_id = REFERENCE_FRAME
            msg.header.stamp = node.get_clock().now().to_msg()
            point = MultiDOFJointTrajectoryPoint()
            transform = Transform()
            transform.translation.x, transform.translation.y, transform.translation.z = p_des.tolist()
            transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = q1
            point.transforms.append(transform)
            vel = Twist()
            vel.linear.x, vel.linear.y, vel.linear.z = v_des.tolist()
            point.velocities.append(vel)
            acc = Twist()
            acc.linear.x, acc.linear.y, acc.linear.z = a_des.tolist()
            point.accelerations.append(acc)
            msg.points.append(point)
            pub.publish(msg)

            if tau >= 1.0:
                node.get_logger().info(f"[{node_name}] trajectory complete, still publishing final point")
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
