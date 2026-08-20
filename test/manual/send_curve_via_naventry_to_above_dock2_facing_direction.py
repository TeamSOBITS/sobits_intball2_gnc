#!/usr/bin/env python3
"""Phase 3b manual check: same cobra-maneuver curve as
send_curve_via_naventry_to_above_dock2.py (current pose -> nav_entry -> above_dock_2,
quadratic Bezier passing exactly through nav_entry at the midpoint), but with
q_des(t) facing the direction of travel (Guidance's
attitude_reference.compute_q_des) instead of a fixed target quaternion.

This is the same stress-test curve used in
docs/trajectory_force_duration_investigation.md (peak translation force ~0.1N
at DURATION_SEC=40s) -- reusing it here so the attitude-tracking check is done
under a realistic, well-conditioned cobra maneuver rather than an ad-hoc short
path (see docs/trajectory_force_duration_investigation.md 6-1's note about the
degenerate near-identical-endpoints failure of an earlier improvised test).

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
DURATION_SEC = 40.0  # see send_curve_via_naventry_to_above_dock2.py: 1/T^2 force scaling
SPEED_THRESHOLD = 0.02  # m/s, matches Trajectory's default (attitude_reference.py)
# 6-7: re-testing lookahead on top of 6-6's pre-aligned baseline (6-5 tested
# it confounded with rate-limiting; this isolates its effect against the
# now-boosted kp_att). Face the direction from the current path point
# toward a point this far ahead in time, instead of the instantaneous
# tangent -- a smooth curve's lookahead-point direction changes more
# gradually than its raw instantaneous tangent, which should reduce the
# demanded attitude-rate right at nav_entry's peak-curvature crossing.
LOOKAHEAD_TIME = 3.0  # s

CHECKPOINT_TOPIC = "/gnc/checkpoints"
# 6-6 (docs/trajectory_force_duration_investigation.md): pre-align the
# vehicle to the curve's initial tangent direction via a static checkpoint
# before streaming /gnc/trajectory_setpoint, so the measured tracking error
# reflects only curve-following (not the startup reorientation transient
# from 6-3/6-4/6-5, which is an unavoidable one-off and not what's being
# tuned here).
ALIGN_TOLERANCE_DEG = 3.0  # converged once within this geodesic angle of q_align
ALIGN_TIMEOUT_SEC = 60.0  # safety cap: PoseCorrector's checkpoint-hold gain
# (tf_correction.kp_att) is much weaker than trajectory_controller's, so a
# large reorientation can take a while -- poll for actual convergence
# instead of a fixed sleep (a fixed 8s undershot this on one leg of the
# round trip: docs/trajectory_force_duration_investigation.md 6-6).

# nav_entry / near_dock, from maps/iss_location.yaml (iss_body frame)
WAYPOINT_POS = [11.000, -4.300, 5.000]  # nav_entry
TARGET_POS = [11.3, -3.636, 5.5]  # above_dock_2


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


def main():
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-only", action="store_true",
        help="Publish the RViz preview path only; never send /gnc/trajectory_setpoint.",
    )
    ns = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node_name = f"send_curve_naventry_above2_facing_{int(time.monotonic() * 1000) % 1000000}"
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
    q0 = quat  # used only for the preview path's fixed-quat field
    w = np.asarray(WAYPOINT_POS)
    p1 = np.asarray(TARGET_POS)
    c = 2.0 * w - 0.5 * p0 - 0.5 * p1
    node.get_logger().info(
        f"[{node_name}] curved path p0={p0.tolist()} -> control={c.tolist()} "
        f"(passes through nav_entry={w.tolist()} at t={DURATION_SEC / 2:.1f}s) "
        f"-> p1={p1.tolist()} (above_dock_2) over {DURATION_SEC}s, facing direction of travel"
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

    # Pre-align to the curve's initial tangent direction via a static
    # checkpoint (see ALIGN_WAIT_SEC above), so the trajectory-following
    # phase below starts from (approximately) the correct attitude instead
    # of an unrelated resting one.
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
    # Wait for an actual subscriber match instead of a fixed sleep -- a fixed
    # 1.0s isn't guaranteed to win the discovery race (this dropped the
    # checkpoint silently on one run: docs/trajectory_force_duration_investigation.md
    # 6-7, torque_corr stayed ~0 for the entire align window instead of
    # converging or even trying).
    match_deadline = time.monotonic() + 5.0
    while rclpy.ok() and chk_pub.get_subscription_count() < 1 and time.monotonic() < match_deadline:
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
                f"[{node_name}] pre-align converged ("
                f"{np.degrees(angle):.2f} deg from target)"
            )
            break
    else:
        node.get_logger().warn(
            f"[{node_name}] pre-align did not converge within "
            f"{ALIGN_TIMEOUT_SEC}s -- proceeding anyway"
        )

    prev_q_des = q_align
    dt = 1.0 / RATE_HZ
    # Sim-clock (not time.monotonic()) so the reference advances at the rate
    # the vehicle can actually move in simulation time, not wall-clock time:
    # under CPU load, Gazebo's real-time factor can drop well below 1, and a
    # wall-clock-paced tau races the reference ahead of what the vehicle can
    # physically achieve -- confirmed as the root cause of the sustained
    # position-tracking failure in docs/recording_cpu_load_control_degradation.md.
    # use_sim_time is forced True via parameter_overrides above (requires
    # /clock bridged from the simulator).
    t_start = node.get_clock().now().nanoseconds * 1e-9
    try:
        while rclpy.ok():
            elapsed = node.get_clock().now().nanoseconds * 1e-9 - t_start
            tau = min(1.0, elapsed / DURATION_SEC)
            s, ds_dtau, dds_dtau = quintic(tau)
            ds_dt = ds_dtau / DURATION_SEC
            dds_dt = dds_dtau / (DURATION_SEC ** 2)

            p_des = bezier(p0, c, p1, s)
            b1 = bezier_d1(p0, c, p1, s)
            b2 = bezier_d2(p0, c, p1)
            v_des = b1 * ds_dt
            a_des = b2 * (ds_dt ** 2) + b1 * dds_dt

            # 6-6: pre-aligned start -> plain instantaneous v_des, no
            # rate-limit/lookahead needed (those were mitigating the startup
            # jump, which pre-alignment now avoids at the source). Isolates
            # the curve-body tracking error to the controller gains alone.
            # 6-8: lookahead confirmed (6-5, 6-7) not to help the curve-body
            # error -- back to plain instantaneous v_des for a clean read on
            # this gain step's effect alone.
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

            # TEMPORARY debug instrumentation (docs/recording_cpu_load_control_degradation.md):
            # confirming the "reference races ahead of sim-time capacity"
            # hypothesis -- dumping wall-clock time.monotonic() alongside this
            # node's sim-clock-based elapsed/tau (use_sim_time is now forced
            # True above) to cross-check against trajectory_controller's own
            # log. Remove after concluding.
            try:
                with open("/tmp/trajectory_reference_race_sender.log", "a") as _dbgf3:
                    _dbgf3.write(f"{time.monotonic():.6f},{elapsed:.6f},{tau:.6f}\n")
            except Exception:
                pass

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
