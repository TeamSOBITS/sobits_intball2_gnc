#!/usr/bin/env python3
"""Send a CURVED (quadratic Bezier) trajectory to /gnc/trajectory_setpoint
targeting the absolute 'above_dock_2' named location, with quintic
(min-jerk) timing along the curve parameter. Publishes the visualization
path via TrajectoryPathPublisher.

Curve: B(s) = (1-s)^2*P0 + 2(1-s)s*C + s^2*P1, s in [0,1], where C is the
straight-line midpoint offset perpendicular to the P0->P1 direction to
produce a visible arc. Position/velocity/acceleration in time use the same
quintic s(tau) timing law as send_trajectory.py, combined via the chain
rule (v = dB/ds * ds/dt, a = d2B/ds2*(ds/dt)^2 + dB/ds*d2s/dt2).

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

# above_dock_2, from maps/iss_location.yaml (iss_body frame)
TARGET_POS = [11.3, -3.636, 5.5]
TARGET_QUAT = [-0.707106, 0.707106, 0.0, 0.0]  # x,y,z,w
BOW_OFFSET_M = 0.6  # perpendicular bulge of the curve's midpoint control point


def quintic(tau):
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def make_control_point(p0, p1, bow_offset):
    mid = (p0 + p1) / 2.0
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return mid
    direction = direction / norm
    # Pick an arbitrary vector not parallel to `direction` to build a
    # perpendicular via cross product.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    perp = np.cross(direction, helper)
    perp = perp / np.linalg.norm(perp)
    return mid + perp * bow_offset


def bezier(p0, c, p1, s):
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s ** 2 * p1


def bezier_d1(p0, c, p1, s):
    return 2 * (1 - s) * (c - p0) + 2 * s * (p1 - c)


def bezier_d2(p0, c, p1):
    return 2 * (p1 - 2 * c + p0)


def main():
    rclpy.init(args=sys.argv)
    node_name = f"send_curved_to_above_dock2_{int(time.monotonic() * 1000) % 1000000}"
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
    c = make_control_point(p0, p1, BOW_OFFSET_M)
    node.get_logger().info(
        f"[{node_name}] curved path p0={p0.tolist()} -> control={c.tolist()} "
        f"-> p1={p1.tolist()} (above_dock_2) over {DURATION_SEC}s"
    )

    n_samples = int(DURATION_SEC / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = bezier(p0, c, p1, s)
        samples.append((p.tolist(), q0))
    path_pub.publish(samples)
    node.get_logger().info(f"[{node_name}] published curved path with {len(samples)} points")

    dt = 1.0 / RATE_HZ
    t_start = time.monotonic()
    try:
        while rclpy.ok():
            elapsed = time.monotonic() - t_start
            tau = min(1.0, elapsed / DURATION_SEC)
            s, ds_dtau, dds_dtau = quintic(tau)
            ds_dt = ds_dtau / DURATION_SEC
            dds_dt = dds_dtau / (DURATION_SEC ** 2)

            p_des = bezier(p0, c, p1, s)
            b1 = bezier_d1(p0, c, p1, s)
            b2 = bezier_d2(p0, c, p1)
            v_des = b1 * ds_dt
            a_des = b2 * (ds_dt ** 2) + b1 * dds_dt

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
