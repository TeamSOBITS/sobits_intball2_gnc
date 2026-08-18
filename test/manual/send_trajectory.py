#!/usr/bin/env python3
"""Phase 3a manual validation: publish a smooth minimum-jerk trajectory to
/gnc/trajectory_setpoint at 50Hz, standing in for the not-yet-implemented
Guidance node (owned by another developer, openspec/changes/
add-trajectory-following).

Quintic (minimum-jerk) 1-axis profile: p(tau) = p0 + (p1-p0)*(10tau^3 -
15tau^4 + 6tau^5), tau=t/T clamped to [0,1]. Zero velocity/acceleration at
both endpoints -- a reasonable stand-in for "Guidance starts the trajectory
from the vehicle's current position/velocity".

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
PATH_SAMPLE_DT = 0.1  # seconds between path-visualization waypoints
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 50.0
DURATION_SEC = 15.0
OFFSET = [-0.20, -0.15, 0.0]  # meters, x/y/z offset from the current position


def quintic(tau):
    """Minimum-jerk s-curve and its 1st/2nd derivative w.r.t. tau, tau in [0,1]."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def main():
    rclpy.init(args=sys.argv)
    node = Node("send_trajectory")
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    pub = node.create_publisher(MultiDOFJointTrajectory, TRAJECTORY_TOPIC, 1)
    path_pub = TrajectoryPathPublisher(node, reference_frame=REFERENCE_FRAME)

    node.get_logger().info("[send_trajectory] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=5.0):
        node.get_logger().error("[send_trajectory] could not get current TF pose")
        return
    pos, quat, _ = tf_client.get_pose()

    p0 = np.asarray(pos)
    q0 = quat
    p1 = p0 + np.asarray(OFFSET)
    node.get_logger().info(f"[send_trajectory] p0={p0.tolist()} -> p1={p1.tolist()} over {DURATION_SEC}s")

    # One-shot visualization path: sample the whole planned quintic profile.
    n_samples = int(DURATION_SEC / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = p0 + (p1 - p0) * s
        samples.append((p.tolist(), q0))
    path_pub.publish(samples)

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
            transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = q0
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
                node.get_logger().info("[send_trajectory] trajectory complete, still publishing final point")
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
