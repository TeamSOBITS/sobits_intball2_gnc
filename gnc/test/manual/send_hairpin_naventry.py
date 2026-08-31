#!/usr/bin/env python3
"""Sharper hairpin stress test: near_dock <-> above_dock via a scaled-up
nav_entry bulge (~144.7 deg turn at --bulge-scale 1.5, vs. the ~89 deg
near_dock<->above_dock_2 cobra maneuver used earlier), with q_des(t) facing
the direction of travel.

Consolidates the former send_hairpin_naventry_facing_direction.py and
preview_hairpin_naventry.py into one script (docs/
guidance_node_implementation_plan.md's test/manual/ cleanup): pass
--path-only to preview the geometry (turn angle, leg lengths, RViz path)
without ever sending /gnc/trajectory_setpoint, same flag name
send_curve_via_naventry.py uses for the same purpose.

Purpose: docs/trajectory_force_duration_investigation.md 6 section found
that the theoretical kp_att recommendation derived from a trajectory's peak
angular acceleration is specific to that trajectory's geometry (sharper turn
-> higher required kp_att), so the near_dock<->above_dock_2 cobra maneuver's
~9-11 recommendation isn't necessarily a true worst case. This script tests
a sharper turn using existing, already-flown-safely endpoints (near_dock,
above_dock) plus an extended nav_entry-direction waypoint, chosen to be much
closer than the far inspection_entry/capture_point locations while still
meaningfully sharper.

Run WITH test/manual/measure_attitude_tracking_error.py in a separate
process to measure the resulting tracking error.

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
PATH_SAMPLE_DT = 0.1
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 50.0
# 6-16 (docs/trajectory_force_duration_investigation.md): a cleaned-up
# (noise-free, analytic) recalculation found the fan hardware's physical
# torque ceiling corresponds to ~100-129 deg turns at 40s (axis-dependent),
# so the 144.7 deg hairpin needs ~1.4-2.4x longer to bring its peak demanded
# angular accel back under that ceiling -- default bumped to 100s (comfortable
# margin over the ~95s the worst-case axis needs). Override with --duration.
DURATION_SEC = 100.0
SPEED_THRESHOLD = 0.02  # m/s, matches Trajectory's default (attitude_reference.py)

CHECKPOINT_TOPIC = "/gnc/checkpoints"
ALIGN_TOLERANCE_DEG = 3.0
ALIGN_TIMEOUT_SEC = 60.0

# near_dock / above_dock / nav_entry, from maps/iss_location.yaml (iss_body
# frame). Using above_dock (not above_dock_2) here -- combined with
# near_dock and nav_entry it forms a naturally sharper (~127 deg) triangle
# than the near_dock<->above_dock_2 pair (~89 deg) uses, with comparable leg
# lengths (no need for a brand-new, unverified endpoint).
NEAR_DOCK = np.array([10.936, -3.636, 4.121])
ABOVE_DOCK = np.array([10.936, -3.636, 5.0])
NAV_ENTRY = np.array([11.0, -4.3, 5.0])
BULGE_SCALE = 1.5  # ~144.7 deg turn


def quintic(tau):
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def bezier(p0, c, p1, s):
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s ** 2 * p1


def bezier_d1(p0, c, p1, s):
    return 2 * (1 - s) * (c - p0) + 2 * s * (p1 - c)


def bezier_d2(p0, c, p1):
    return 2 * (p1 - 2 * c + p0)


def turn_angle_deg(p0, w, p1):
    leg1 = w - p0
    leg2 = p1 - w
    cos_a = np.dot(leg1, leg2) / (np.linalg.norm(leg1) * np.linalg.norm(leg2))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def main():
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reverse", action="store_true",
        help="above_dock -> near_dock instead of near_dock -> above_dock.",
    )
    parser.add_argument("--bulge-scale", type=float, default=BULGE_SCALE)
    parser.add_argument("--duration", type=float, default=DURATION_SEC)
    parser.add_argument(
        "--path-only", action="store_true",
        help="Publish the RViz preview path (with turn-angle/leg-length "
             "geometry logged) only; never send /gnc/trajectory_setpoint "
             "(no vehicle motion). Ctrl-C to exit after previewing.",
    )
    ns = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    duration_sec = ns.duration

    rclpy.init(args=sys.argv)
    node_name = f"send_hairpin_naventry_{int(time.monotonic() * 1000) % 1000000}"
    node = Node(
        node_name,
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    tf_client = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)
    path_pub = PathPublisher(node, TRAJECTORY_PATH_TOPIC, reference_frame=REFERENCE_FRAME)

    node.get_logger().info(f"[{node_name}] waiting for TF...")
    if not tf_client.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"[{node_name}] could not get a TF pose")
        return
    _pos, quat0, _ = tf_client.get_pose()

    p0, p1 = (ABOVE_DOCK, NEAR_DOCK) if ns.reverse else (NEAR_DOCK, ABOVE_DOCK)
    midpoint = 0.5 * (p0 + p1)
    bulge = NAV_ENTRY - midpoint
    w = midpoint + ns.bulge_scale * bulge
    c = 2.0 * w - 0.5 * p0 - 0.5 * p1

    angle = turn_angle_deg(p0, w, p1)
    leg1_len = np.linalg.norm(w - p0)
    leg2_len = np.linalg.norm(p1 - w)
    node.get_logger().info(
        f"[{node_name}] hairpin path (bulge_scale={ns.bulge_scale}) "
        f"p0={p0.tolist()} -> waypoint={w.tolist()} -> p1={p1.tolist()} "
        f"turn_angle={angle:.1f}deg leg_lengths=[{leg1_len:.3f}, {leg2_len:.3f}]m "
        f"over {duration_sec}s, facing direction of travel"
    )

    n_samples = int(duration_sec / PATH_SAMPLE_DT) + 1
    preview = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = bezier(p0, c, p1, s)
        preview.append((p.tolist(), quat0))
    path_pub.publish(preview)
    node.get_logger().info(f"[{node_name}] published curved path with {len(preview)} points")

    if ns.path_only:
        node.get_logger().info(
            f"[{node_name}] --path-only: not sending /gnc/trajectory_setpoint. "
            "Path is latched for RViz; Ctrl-C when done previewing."
        )
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return

    pub = node.create_publisher(MultiDOFJointTrajectory, TRAJECTORY_TOPIC, 1)

    # Pre-align to the curve's initial tangent direction, so the measured
    # tracking error reflects curve-following only, not the unavoidable
    # startup-reorientation transient.
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
        angle_now = 2.0 * np.arccos(np.clip(abs(qe), 0.0, 1.0))
        if angle_now <= align_tolerance_rad:
            node.get_logger().info(
                f"[{node_name}] pre-align converged ({np.degrees(angle_now):.2f} deg from target)"
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

            q_des = compute_q_des(v_des, prev_q_des, SPEED_THRESHOLD)
            prev_q_des = q_des

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
