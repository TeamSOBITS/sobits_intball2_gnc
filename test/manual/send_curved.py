#!/usr/bin/env python3
"""Send a CURVED (quadratic Bezier) trajectory to /gnc/trajectory_setpoint
targeting a named location, with quintic (min-jerk) timing along the curve
parameter. Publishes the visualization path via PathPublisher.

Consolidates the former send_curved_to_above_dock2.py (fixed target
attitude) and send_curved_facing_direction_of_travel.py (q_des(t) faces the
direction of travel) into one script, the same
--facing-direction split send_curve_via_naventry.py uses
(docs/guidance_node_implementation_plan.md's test/manual/ cleanup). The
target is resolved live via TF by name instead of a hardcoded coordinate.

Curve: B(s) = (1-s)^2*P0 + 2(1-s)s*C + s^2*P1, s in [0,1], where C is the
straight-line midpoint offset perpendicular to the P0->P1 direction to
produce a visible arc (unlike send_curve_via_naventry.py, there is no
explicit intermediate waypoint here -- just a fixed bulge amount). Position/
velocity/acceleration in time use the quintic s(tau) timing law, combined via
the chain rule (v = dB/ds * ds/dt, a = d2B/ds2*(ds/dt)^2 + dB/ds*d2s/dt2).

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Pose, PoseArray, Transform, Twist

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.path_publisher import PathPublisher
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

TRAJECTORY_TOPIC = "/gnc/trajectory_setpoint"
TRAJECTORY_PATH_TOPIC = "/gnc/trajectory_path"
CHECKPOINT_TOPIC = "/gnc/checkpoints"
PATH_SAMPLE_DT = 0.1
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 50.0
BOW_OFFSET_M = 0.6  # perpendicular bulge of the curve's midpoint control point
SPEED_THRESHOLD = 0.02  # m/s, matches Trajectory's default (attitude_reference.py)
# See attitude_reference.compute_q_des's docstring and
# docs/trajectory_force_duration_investigation.md 6-3: an unlimited q_des
# jump produced a large, slow-to-clear tracking error. 20 deg/s is a first,
# untuned attempt, only used in --facing-direction mode.
MAX_ANGULAR_RATE = np.radians(20.0)  # rad/s
ALIGN_TOLERANCE_DEG = 3.0
ALIGN_TIMEOUT_SEC = 60.0


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
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default="above_dock_2",
                        help="named target TF frame (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=None, metavar="SEC",
                        help="curve duration (default: 15.0, or 30.0 with "
                             "--facing-direction)")
    parser.add_argument("--bow-offset", type=float, default=BOW_OFFSET_M,
                        metavar="M", help="curve bulge amount (default: %(default)s)")
    parser.add_argument(
        "--facing-direction", action="store_true",
        help="face the direction of travel instead of holding a fixed "
             "target attitude; pre-aligns to the initial tangent first.",
    )
    ns = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    duration_sec = ns.duration if ns.duration is not None else (
        30.0 if ns.facing_direction else 15.0
    )

    rclpy.init(args=sys.argv)
    node_name = f"send_curved_{int(time.monotonic() * 1000) % 1000000}"
    node = Node(
        node_name,
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    pub = node.create_publisher(MultiDOFJointTrajectory, TRAJECTORY_TOPIC, 1)
    path_pub = PathPublisher(node, TRAJECTORY_PATH_TOPIC, reference_frame=REFERENCE_FRAME)

    node.get_logger().info(f"[{node_name}] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"[{node_name}] could not get a TF pose")
        return
    pos, quat, _ = tf_client.get_pose()
    node.get_logger().info(f"[{node_name}] TF pose acquired: {pos}")

    target_tf = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=ns.target)
    if not target_tf.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"[{node_name}] could not resolve target '{ns.target}'")
        node.destroy_node()
        rclpy.shutdown()
        return
    target_pos, target_quat, _ = target_tf.get_pose()

    p0 = np.asarray(pos)
    q0 = np.asarray(quat, dtype=float)
    p1 = np.asarray(target_pos)
    q1 = target_quat
    c = make_control_point(p0, p1, ns.bow_offset)
    node.get_logger().info(
        f"[{node_name}] curved path p0={p0.tolist()} -> control={c.tolist()} "
        f"-> p1={p1.tolist()} ({ns.target}) over {duration_sec}s"
        + (" facing direction of travel" if ns.facing_direction else "")
    )

    n_samples = int(duration_sec / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = bezier(p0, c, p1, s)
        samples.append((p.tolist(), q0.tolist()))
    path_pub.publish(samples)
    node.get_logger().info(f"[{node_name}] published curved path with {len(samples)} points")

    prev_q_des = q0
    if ns.facing_direction:
        # Pre-align to the curve's initial tangent direction via a static
        # checkpoint before streaming /gnc/trajectory_setpoint, so the
        # measured tracking error reflects curve-following only, not the
        # startup reorientation transient.
        b1_0 = bezier_d1(p0, c, p1, 0.0)
        q_align = compute_q_des(b1_0, None, SPEED_THRESHOLD)
        chk_pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 1)
        chk_msg = PoseArray()
        chk_msg.header.frame_id = REFERENCE_FRAME
        chk_msg.header.stamp = node.get_clock().now().to_msg()
        chk_pose = Pose()
        chk_pose.position.x, chk_pose.position.y, chk_pose.position.z = p0.tolist()
        (chk_pose.orientation.x, chk_pose.orientation.y,
         chk_pose.orientation.z, chk_pose.orientation.w) = q_align.tolist()
        chk_msg.poses.append(chk_pose)
        match_deadline = time.monotonic() + 5.0
        while (rclpy.ok() and chk_pub.get_subscription_count() < 1
               and time.monotonic() < match_deadline):
            rclpy.spin_once(node, timeout_sec=0.05)
        if chk_pub.get_subscription_count() < 1:
            node.get_logger().warn(
                f"[{node_name}] no /gnc/checkpoints subscriber matched after 5s -- "
                "publishing anyway, but pre-align will likely fail"
            )
        chk_pub.publish(chk_msg)
        node.get_logger().info(
            f"[{node_name}] pre-aligning to initial tangent q={q_align.tolist()}, "
            f"polling for convergence (within {ALIGN_TOLERANCE_DEG} deg, "
            f"timeout {ALIGN_TIMEOUT_SEC}s)..."
        )
        align_deadline = time.monotonic() + ALIGN_TIMEOUT_SEC
        align_tolerance_rad = np.radians(ALIGN_TOLERANCE_DEG)
        while rclpy.ok() and time.monotonic() < align_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            current_pose = tf_client.get_pose()
            if current_pose is None:
                continue
            _cur_pos, cur_quat, _stamp = current_pose
            qe = np.dot(np.asarray(cur_quat, dtype=float), q_align)
            angle = 2.0 * np.arccos(np.clip(abs(qe), 0.0, 1.0))
            if angle <= align_tolerance_rad:
                node.get_logger().info(
                    f"[{node_name}] pre-align converged "
                    f"({np.degrees(angle):.2f} deg from target)"
                )
                break
        else:
            node.get_logger().warn(
                f"[{node_name}] pre-align did not converge within "
                f"{ALIGN_TIMEOUT_SEC}s -- proceeding anyway"
            )
        prev_q_des = q_align

    dt = 1.0 / RATE_HZ
    # sim clock (not time.monotonic()) so tau tracks simulated time, not
    # wall-clock time -- under RTF<1 a wall-clock tau races ahead of what the
    # vehicle can actually achieve in sim time.
    t_start = node.get_clock().now()
    try:
        while rclpy.ok():
            elapsed = (node.get_clock().now() - t_start).nanoseconds * 1e-9
            tau = min(1.0, elapsed / duration_sec)
            s, ds_dtau, dds_dtau = quintic(tau)
            ds_dt = ds_dtau / duration_sec
            dds_dt = dds_dtau / (duration_sec ** 2)

            p_des = bezier(p0, c, p1, s)
            b1 = bezier_d1(p0, c, p1, s)
            b2 = bezier_d2(p0, c, p1)
            v_des = b1 * ds_dt
            a_des = b2 * (ds_dt ** 2) + b1 * dds_dt

            if ns.facing_direction:
                q_des = compute_q_des(v_des, prev_q_des, SPEED_THRESHOLD, dt=dt,
                                       max_angular_rate=MAX_ANGULAR_RATE)
                prev_q_des = q_des
            else:
                q_des = np.asarray(q1)

            msg = MultiDOFJointTrajectory()
            msg.header.frame_id = REFERENCE_FRAME
            msg.header.stamp = node.get_clock().now().to_msg()
            point = MultiDOFJointTrajectoryPoint()
            transform = Transform()
            transform.translation.x, transform.translation.y, transform.translation.z = p_des.tolist()
            transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = q_des.tolist()
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
