#!/usr/bin/env python3
"""Send a CURVED (quadratic Bezier) trajectory to /gnc/trajectory_setpoint
from the current position, through a named waypoint (default: 'nav_entry'),
to a named target location -- with quintic (min-jerk) timing along the curve
parameter. Publishes the visualization path via PathPublisher.

Consolidates the former per-destination/per-mode scripts
(send_curve_via_naventry_to_{near_dock,above_dock2}[_facing_direction].py,
docs/guidance_node_implementation_plan.md's test/manual/ cleanup): waypoint
and target are resolved live via TF (same TfClient lookup
guidance/ros/move_to_client.py uses) instead of hardcoded per-file
coordinates, and --facing-direction toggles between a fixed target quaternion
and facing the direction of travel (guidance.utils.attitude_reference), so
one file now serves every destination/mode combination.

Curve: B(s) = (1-s)^2*P0 + 2(1-s)s*C + s^2*P1, s in [0,1]. The control point
C is solved so the curve passes exactly through the waypoint at s=0.5:
    B(0.5) = 0.25*P0 + 0.5*C + 0.25*P1 = WAYPOINT
    => C = 2*WAYPOINT - 0.5*P0 - 0.5*P1
Since the quintic timing law also satisfies s(tau=0.5)=0.5 by symmetry, the
vehicle passes through the waypoint at the midpoint of --duration. Position/
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
DURATION_SEC = 40.0  # 1/T^2 peak-force scaling; see former per-file notes
SPEED_THRESHOLD = 0.02  # m/s, matches Trajectory's default (attitude_reference.py)
ERROR_LOG_PERIOD_SEC = 1.0
ALIGN_TOLERANCE_DEG = 3.0
ALIGN_TIMEOUT_SEC = 60.0


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


def _resolve(node, name):
    """Look up a named TF frame's (pos, quat) in REFERENCE_FRAME, or exit."""
    tf = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=name)
    if not tf.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error(f"could not resolve TF frame '{name}'")
        sys.exit(1)
    pos, quat, _stamp = tf.get_pose()
    return np.asarray(pos), quat


def main():
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="named target TF frame, e.g. near_dock")
    parser.add_argument("--waypoint", default="nav_entry",
                        help="named mid-curve waypoint (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=DURATION_SEC,
                        metavar="SEC", help="curve duration (default: %(default)s)")
    parser.add_argument(
        "--facing-direction", action="store_true",
        help="face the direction of travel instead of holding a fixed "
             "target attitude; pre-aligns to the initial tangent first.",
    )
    parser.add_argument(
        "--path-only", action="store_true",
        help="Publish the RViz preview path only; never send /gnc/trajectory_setpoint "
             "(no vehicle motion). Keeps the node alive so the latched path stays "
             "visible until Ctrl-C.",
    )
    ns = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    duration_sec = ns.duration

    rclpy.init(args=sys.argv)
    node_name = f"send_curve_via_naventry_{int(time.monotonic() * 1000) % 1000000}"
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

    p0 = np.asarray(pos)
    q0 = quat
    w, _w_quat = _resolve(node, ns.waypoint)
    p1, q1 = _resolve(node, ns.target)
    c = 2.0 * w - 0.5 * p0 - 0.5 * p1
    node.get_logger().info(
        f"[{node_name}] curved path p0={p0.tolist()} -> control={c.tolist()} "
        f"(passes through {ns.waypoint}={w.tolist()} at t={duration_sec / 2:.1f}s) "
        f"-> p1={p1.tolist()} ({ns.target}) over {duration_sec}s"
        + (" facing direction of travel" if ns.facing_direction else "")
    )

    n_samples = int(duration_sec / PATH_SAMPLE_DT) + 1
    samples = []
    for i in range(n_samples):
        tau = i / (n_samples - 1)
        s, _, _ = quintic(tau)
        p = bezier(p0, c, p1, s)
        samples.append((p.tolist(), q0))
    path_pub.publish(samples)
    node.get_logger().info(f"[{node_name}] published curved path with {len(samples)} points")

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

    prev_q_des = q0
    if ns.facing_direction:
        # Pre-align to the curve's initial tangent direction via a static
        # checkpoint, so the tracking-error measurement below reflects only
        # curve-following (not the startup reorientation transient).
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
    last_error_log = time.monotonic()
    max_error = 0.0
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
                q_des = compute_q_des(v_des, prev_q_des, SPEED_THRESHOLD)
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

            now = time.monotonic()
            if now - last_error_log >= ERROR_LOG_PERIOD_SEC:
                last_error_log = now
                actual = tf_client.get_pose()
                if actual is not None:
                    p_now = np.asarray(actual[0])
                    error = float(np.linalg.norm(p_now - p_des))
                    max_error = max(max_error, error)
                    node.get_logger().info(
                        f"[{node_name}] t={elapsed:5.1f}s tau={tau:.2f} "
                        f"p_des={np.round(p_des, 3).tolist()} "
                        f"p_now={np.round(p_now, 3).tolist()} "
                        f"tracking_error={error * 1000:.1f}mm (max so far: {max_error * 1000:.1f}mm)"
                    )
                else:
                    node.get_logger().warn(f"[{node_name}] no TF pose available for error check")

            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
