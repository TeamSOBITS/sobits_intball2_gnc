#!/usr/bin/env python3
"""Experiment: compare a static-checkpoint ("step") align against a SLERP +
trapezoidal-velocity-profile moving-checkpoint ("trajectory") align, to check
whether feeding tf_correction's checkpoint loop a smoothly moving target
removes the composite-axis overshoot documented in
docs/2026-08-27_composite_axis_overshoot_summary_and_plan.md.

Does NOT modify tf_correction/control_node gains or config -- only publishes
checkpoints to /gnc/checkpoints (same mechanism as diagnose_align_gains.py)
and polls TF/gyro, logging angle-to-target over time to CSV. Run control_node
with whatever gains you want to test beforehand.

Before commanding the offset, waits for the vehicle to be at rest (gyro below
--rest-gyro-threshold-deg-s, held for --rest-hold-time) at its CURRENT pose --
this matters because the trapezoidal profile assumes a rest-to-rest move;
starting from nonzero angular velocity (e.g. left over from a still-converging
previous run) invalidates it. After commanding the offset, runs until the
angle-to-target stays within --convergence-tolerance-deg for
--convergence-hold-time (or --convergence-timeout elapses), and reports
time-to-converge and final accuracy.

All timing is on the node clock (use_sim_time=True), no wall-clock sleep.

Usage:
    # step (today's behavior): single static checkpoint
    python3 test/manual/experiment_align_slerp_trapezoid.py \\
        --mode step --axis xyz --offset-deg 180 \\
        --out-csv /tmp/align_step_180.csv

    # trajectory: SLERP + trapezoidal profile, checkpoint moved every tick
    python3 test/manual/experiment_align_slerp_trapezoid.py \\
        --mode trajectory --axis xyz --offset-deg 180 \\
        --cruise-rate-deg 20 --accel-deg 10 \\
        --out-csv /tmp/align_traj_180.csv
"""
import argparse
import csv
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ib2_msgs.msg import IMU

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.quat_math import quat_mul, quat_conj, geodesic_angle

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
CHECKPOINT_TOPIC = "/gnc/checkpoints"
GYRO_TOPIC = "/imu/imu"
POLL_HZ = 20.0
PUBLISH_HZ_DEFAULT = 20.0
OUT_CSV_DEFAULT = "/tmp/experiment_align_slerp_trapezoid_log.csv"

# Mirrors tf_correction.align_tolerance_deg / align_settle_time in
# config/gnc_params.yaml, so "converged" here means the same thing align does.
CONVERGENCE_TOLERANCE_DEG_DEFAULT = 3.0
CONVERGENCE_HOLD_TIME_DEFAULT = 0.5
CONVERGENCE_TIMEOUT_DEFAULT = 60.0

REST_GYRO_THRESHOLD_DEG_S_DEFAULT = 1.0
REST_HOLD_TIME_DEFAULT = 2.0
REST_TIMEOUT_DEFAULT = 60.0

OVERSHOOT_ENTRY_DEG = 10.0


def slerp(q0, q1, t):
    """SLERP between q0 and q1 (both [x,y,z,w]) at t in [0, 1].

    Resolves the double-cover sign so the interpolation takes the short arc.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def trapezoid_duration(theta_total, v_cap, a_max):
    """Rest-to-rest trapezoidal (or triangular, if too short) duration [s]
    for covering angle ``theta_total`` [rad] with cruise rate ``v_cap``
    [rad/s] and acceleration ``a_max`` [rad/s^2]."""
    if theta_total <= 0.0:
        return 0.0
    d_accel = v_cap * v_cap / (2.0 * a_max)
    if theta_total >= 2.0 * d_accel:
        t_accel = v_cap / a_max
        return 2.0 * t_accel + (theta_total - 2.0 * d_accel) / v_cap
    v_peak = np.sqrt(a_max * theta_total)
    return 2.0 * v_peak / a_max


def trapezoid_fraction(t, theta_total, v_cap, a_max, duration):
    """Progress fraction u(t) in [0, 1] along the trapezoid profile."""
    if theta_total <= 0.0 or duration <= 0.0:
        return 1.0
    t = float(np.clip(t, 0.0, duration))
    d_accel = v_cap * v_cap / (2.0 * a_max)
    if theta_total >= 2.0 * d_accel:
        t_accel = v_cap / a_max
        t_decel_start = duration - t_accel
        if t <= t_accel:
            s = 0.5 * a_max * t * t
        elif t <= t_decel_start:
            s = d_accel + v_cap * (t - t_accel)
        else:
            s = theta_total - 0.5 * a_max * (duration - t) ** 2
    else:
        t_accel = duration / 2.0
        if t <= t_accel:
            s = 0.5 * a_max * t * t
        else:
            s = theta_total - 0.5 * a_max * (duration - t) ** 2
    return float(np.clip(s / theta_total, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["step", "trajectory"], required=True)
    ap.add_argument("--axis", default="xyz", choices=["x", "y", "z", "xyz"],
                     help="body-local rotation axis for the offset (default: xyz)")
    ap.add_argument("--offset-deg", type=float, default=180.0)
    ap.add_argument("--cruise-rate-deg", type=float, default=20.0,
                     help="trajectory mode only: cruise angular rate [deg/s]")
    ap.add_argument("--accel-deg", type=float, default=10.0,
                     help="trajectory mode only: angular acceleration cap [deg/s^2]")
    ap.add_argument("--publish-hz", type=float, default=PUBLISH_HZ_DEFAULT,
                     help="trajectory mode only: checkpoint publish rate")
    ap.add_argument("--convergence-tolerance-deg", type=float,
                     default=CONVERGENCE_TOLERANCE_DEG_DEFAULT)
    ap.add_argument("--convergence-hold-time", type=float,
                     default=CONVERGENCE_HOLD_TIME_DEFAULT)
    ap.add_argument("--convergence-timeout", type=float,
                     default=CONVERGENCE_TIMEOUT_DEFAULT,
                     help="give up waiting for convergence after this many "
                          "sim-seconds and log whatever was reached")
    ap.add_argument("--rest-gyro-threshold-deg-s", type=float,
                     default=REST_GYRO_THRESHOLD_DEG_S_DEFAULT,
                     help="pre-run: |gyro| must drop below this before we "
                          "capture the rest-to-rest starting pose")
    ap.add_argument("--rest-hold-time", type=float, default=REST_HOLD_TIME_DEFAULT)
    ap.add_argument("--rest-timeout", type=float, default=REST_TIMEOUT_DEFAULT)
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    args = ap.parse_args()

    axis_vec = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "xyz": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }[args.axis]

    rclpy.init()
    node = Node("experiment_align_slerp_trapezoid")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])

    tf_client = TfClient(node, REFERENCE_FRAME, TARGET_FRAME)
    if not tf_client.wait_for_frame(timeout_sec=10.0):
        print("TF unavailable, aborting")
        return 1

    latest_gyro_deg_s = {"mag": None}

    def on_imu(msg: IMU):
        latest_gyro_deg_s["mag"] = float(np.degrees(
            np.linalg.norm([msg.gyro_x, msg.gyro_y, msg.gyro_z])))

    node.create_subscription(IMU, GYRO_TOPIC, on_imu, 10)

    checkpoint_pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 5)

    def publish_checkpoint(pos, quat):
        msg = PoseArray()
        msg.header.frame_id = REFERENCE_FRAME
        p = Pose()
        p.position.x, p.position.y, p.position.z = pos
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
        msg.poses = [p]
        checkpoint_pub.publish(msg)

    def clear_checkpoints():
        msg = PoseArray()
        msg.header.frame_id = REFERENCE_FRAME
        checkpoint_pub.publish(msg)

    def wait_until_at_rest(threshold_deg_s, hold_time_s, timeout_s):
        pos_hold, quat_hold, _ = tf_client.get_pose()
        publish_checkpoint(pos_hold, quat_hold)
        start = node.get_clock().now()
        below_since = None
        while rclpy.ok():
            now = node.get_clock().now()
            elapsed = (now - start).nanoseconds * 1e-9
            if elapsed > timeout_s:
                print(f"WARNING: rest-wait timed out after {timeout_s}s "
                      f"(last |gyro|={latest_gyro_deg_s['mag']})")
                return False
            mag = latest_gyro_deg_s["mag"]
            if mag is not None and mag <= threshold_deg_s:
                if below_since is None:
                    below_since = now
                elif (now - below_since).nanoseconds * 1e-9 >= hold_time_s:
                    return True
            else:
                below_since = None
            rclpy.spin_once(node, timeout_sec=0.05)
        return False

    print(f"waiting for vehicle to settle (|gyro| <= {args.rest_gyro_threshold_deg_s}deg/s "
          f"for {args.rest_hold_time}s) before starting ...")
    settled = wait_until_at_rest(args.rest_gyro_threshold_deg_s, args.rest_hold_time,
                                  args.rest_timeout)
    print("settled, capturing rest-to-rest start pose" if settled else
          "proceeding anyway after rest-wait timeout")

    pos0, quat0, _ = tf_client.get_pose()
    quat0 = np.asarray(quat0, dtype=float)

    half = np.radians(args.offset_deg) / 2.0
    offset_q = np.array([*(np.sin(half) * axis_vec), np.cos(half)])
    target_quat = quat_mul(quat0, offset_q)
    theta_total = geodesic_angle(quat0, target_quat)

    rows = []
    converge_state = {"below_since": None, "converged_t": None}

    def log_tick(t_since_start):
        pose = tf_client.get_pose()
        if pose is None:
            return
        _, quat, stamp = pose
        qe = quat_mul(quat_conj(target_quat), np.asarray(quat, dtype=float))
        raw_w = qe[3]
        angle_deg = float(np.degrees(2.0 * np.arccos(min(1.0, abs(raw_w)))))
        rows.append({"stamp": stamp, "t": t_since_start, "angle_to_target_deg": angle_deg})

        if converge_state["converged_t"] is None:
            now = node.get_clock().now()
            if angle_deg <= args.convergence_tolerance_deg:
                if converge_state["below_since"] is None:
                    converge_state["below_since"] = now
                elif (now - converge_state["below_since"]).nanoseconds * 1e-9 >= args.convergence_hold_time:
                    converge_state["converged_t"] = t_since_start
            else:
                converge_state["below_since"] = None

    log_timer = node.create_timer(1.0 / POLL_HZ, lambda: log_tick(
        (node.get_clock().now() - t0).nanoseconds * 1e-9))

    if args.mode == "step":
        print(f"[step] publishing single {args.offset_deg}deg {args.axis}-axis checkpoint ...")
        t0 = node.get_clock().now()
        publish_checkpoint(pos0, target_quat)
    else:
        v_cap = np.radians(args.cruise_rate_deg)
        a_max = np.radians(args.accel_deg)
        traj_duration = trapezoid_duration(theta_total, v_cap, a_max)
        print(f"[trajectory] {args.offset_deg}deg {args.axis}-axis, "
              f"cruise={args.cruise_rate_deg}deg/s accel={args.accel_deg}deg/s^2 "
              f"-> ramp duration {traj_duration:.2f}s")
        t0 = node.get_clock().now()
        traj_timer_period = 1.0 / args.publish_hz

        def traj_tick():
            t = (node.get_clock().now() - t0).nanoseconds * 1e-9
            if t >= traj_duration:
                publish_checkpoint(pos0, target_quat)
                traj_timer.cancel()
                return
            u = trapezoid_fraction(t, theta_total, v_cap, a_max, traj_duration)
            q_des = slerp(quat0, target_quat, u)
            publish_checkpoint(pos0, q_des)

        traj_timer = node.create_timer(traj_timer_period, traj_tick)

    print(f"running until convergence (<= {args.convergence_tolerance_deg}deg held "
          f"{args.convergence_hold_time}s) or {args.convergence_timeout}s timeout ...")
    while rclpy.ok():
        if converge_state["converged_t"] is not None:
            break
        elapsed = (node.get_clock().now() - t0).nanoseconds * 1e-9
        if elapsed > args.convergence_timeout:
            print(f"WARNING: did not converge within {args.convergence_timeout}s")
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    log_timer.cancel()

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if rows:
        angles = [r["angle_to_target_deg"] for r in rows]
        final_angle = angles[-1]
        print(f"logged {len(rows)} ticks -> {args.out_csv}")
        if converge_state["converged_t"] is not None:
            print(f"CONVERGED at t={converge_state['converged_t']:.2f}s "
                  f"(<= {args.convergence_tolerance_deg}deg held "
                  f"{args.convergence_hold_time}s), final angle-to-target: "
                  f"{final_angle:.2f}deg")
        else:
            print(f"did NOT converge within timeout, final angle-to-target: "
                  f"{final_angle:.2f}deg")
        first_close_idx = next((i for i, a in enumerate(angles) if a <= OVERSHOOT_ENTRY_DEG), None)
        if first_close_idx is not None:
            entry_angle = angles[first_close_idx]
            rebound_peak = max(angles[first_close_idx:])
            print(f"first reached <={OVERSHOOT_ENTRY_DEG}deg at t={rows[first_close_idx]['t']:.2f}s "
                  f"(angle={entry_angle:.2f}deg), peak angle afterward: {rebound_peak:.2f}deg "
                  f"(overshoot rebound: {rebound_peak - entry_angle:.2f}deg)")
        else:
            print(f"never reached <={OVERSHOOT_ENTRY_DEG}deg within the logged duration")
    else:
        print("no TF samples logged")

    print("clearing checkpoints ...")
    clear_checkpoints()

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
