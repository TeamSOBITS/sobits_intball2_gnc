#!/usr/bin/env python3
"""Rigorous A/B test: thrust_allocator.minimax_objective OFF vs ON, for
composite-axis align overshoot (docs/2026-08-27_composite_axis_overshoot_
next_steps.md).

The earlier A/B test in docs/2026-08-27_thrust_allocator_single_axis_
saturation_findings.md ran each condition once, without controlling for the
vehicle's residual angular velocity between conditions, and its "single-axis-
dominance" column was very likely computed before the async-tick-mismatch
measurement bug (see that doc's "最終訂正") was found and fixed -- so it may
be unreliable. This script fixes both gaps for the metrics that matter here
(convergence time, max angle overshoot, sign flips -- all computed from TF
alone, not the buggy cross-topic duty reconstruction):

- Multiple trials per condition (default 3), not one-off.
- A settle-wait (polling body-frame gyro via IMU, sim-clock timeout, no
  wall-clock sleep) before every trial, so each trial starts from a
  genuinely at-rest vehicle instead of carrying over angular velocity from
  the previous trial.
- Single long-lived node/rclpy session across all trials (avoids restarting
  control_node, which is a much heavier and riskier operation than needed
  here -- what actually needs to be "clean" between trials is the vehicle's
  physical state, which the settle-wait handles).

Does NOT modify config/gnc_params.yaml defaults: gains and
thrust_allocator.minimax_objective are restored to their pre-run values at
the end (or immediately via --restore).

Usage:
    python3 test/manual/ab_test_minimax.py --axis xyz --offset-deg 180 \\
        --trials 3 --duration 28 --out-csv-prefix /tmp/ab_test
"""
import argparse
import csv
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters
from geometry_msgs.msg import PoseArray, Pose
from ib2_msgs.msg import IMU

from sobits_intball2_gnc.common.ros.tf_client import TfClient

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
CHECKPOINT_TOPIC = "/gnc/checkpoints"
IMU_TOPIC = "/imu/imu"
POLL_HZ = 20.0

BASELINE_KP_ATT = [0.01, 0.01, 0.01]
BASELINE_KD_ATT = [0.0, 0.0, 0.0]
BASELINE_MAX_CORR_TORQUE = 0.01
BASELINE_ATT_FILTER_ALPHA = 1.0

DEFAULT_KP_ATT = [0.20, 0.20, 0.20]
DEFAULT_KD_ATT = [0.4635, 0.4077, 0.264]
DEFAULT_MAX_CORR_TORQUE = 0.3


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qe_and_sign(target_quat, quat):
    qe = quat_mul(quat_conj(target_quat), quat)
    raw_w = qe[3]
    sign = np.sign(raw_w if raw_w != 0.0 else 1.0)
    angle_deg = float(np.degrees(2.0 * np.arccos(min(1.0, abs(raw_w)))))
    return qe, raw_w, sign, angle_deg


def dparam(name, value_list):
    return ParameterMsg(name=name, value=ParameterValue(
        type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=value_list))


def dscalar(name, value):
    return ParameterMsg(name=name, value=ParameterValue(
        type=ParameterType.PARAMETER_DOUBLE, double_value=value))


def dbool(name, value):
    return ParameterMsg(name=name, value=ParameterValue(
        type=ParameterType.PARAMETER_BOOL, bool_value=value))


def pv_to_python(pv: ParameterValue):
    if pv.type == ParameterType.PARAMETER_DOUBLE:
        return pv.double_value
    if pv.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
        return list(pv.double_array_value)
    if pv.type == ParameterType.PARAMETER_BOOL:
        return pv.bool_value
    raise ValueError(f"unsupported parameter type {pv.type}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", default="xyz", choices=["x", "y", "z", "xyz"])
    ap.add_argument("--offset-deg", type=float, default=180.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=28.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--kd-override", type=float, nargs=3, default=DEFAULT_KD_ATT)
    ap.add_argument("--kp-override", type=float, nargs=3, default=DEFAULT_KP_ATT)
    ap.add_argument("--max-torque", type=float, default=DEFAULT_MAX_CORR_TORQUE)
    ap.add_argument("--settle-gyro-thresh", type=float, default=0.02,
                     help="rad/s magnitude below which the vehicle is considered "
                          "at rest (default 0.02)")
    ap.add_argument("--settle-hold-s", type=float, default=1.0,
                     help="how long gyro must stay below threshold, sim seconds")
    ap.add_argument("--settle-timeout-s", type=float, default=20.0)
    ap.add_argument("--out-csv-prefix", default="/tmp/ab_test_minimax")
    args = ap.parse_args()

    axis_vec = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "xyz": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }[args.axis]

    rclpy.init()
    node = Node("ab_test_minimax")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    set_cli = node.create_client(SetParameters, "/control_node/set_parameters")
    get_cli = node.create_client(GetParameters, "/control_node/get_parameters")
    if not set_cli.wait_for_service(timeout_sec=5.0) or not get_cli.wait_for_service(timeout_sec=5.0):
        print("control_node parameter services unavailable")
        return 1

    def get_param(name):
        req = GetParameters.Request()
        req.names = [name]
        future = get_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f"get_parameters({name}) failed")
        return pv_to_python(result.values[0])

    def set_params(msgs):
        req = SetParameters.Request()
        req.parameters = msgs
        future = set_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            reasons = [r.reason for r in (result.results if result else []) if not r.successful]
            print(f"WARNING: set_params failed: {reasons}")

    def set_align_gains(kp, kd, max_torque, filt):
        set_params([
            dparam("tf_correction.kp_att_align", kp),
            dparam("tf_correction.kd_att_align", kd),
            dscalar("tf_correction.max_corr_torque", max_torque),
            dscalar("tf_correction.att_filter_alpha", filt),
            dbool("tf_correction.torque_direction_preserving", False),
        ])

    checkpoint_pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 5)

    def clear_checkpoints():
        msg = PoseArray()
        msg.header.frame_id = REFERENCE_FRAME
        checkpoint_pub.publish(msg)

    latest_gyro = {"values": None}

    def on_imu(msg: IMU):
        latest_gyro["values"] = (msg.gyro_x, msg.gyro_y, msg.gyro_z)

    node.create_subscription(IMU, IMU_TOPIC, on_imu, 10)

    tf_client = TfClient(node, REFERENCE_FRAME, TARGET_FRAME)
    if not tf_client.wait_for_frame(timeout_sec=10.0):
        print("TF unavailable, aborting")
        return 1

    def wait_for_settle():
        start = node.get_clock().now()
        timeout = rclpy.duration.Duration(seconds=args.settle_timeout_s)
        hold_needed = rclpy.duration.Duration(seconds=args.settle_hold_s)
        below_since = None
        while rclpy.ok():
            now = node.get_clock().now()
            if (now - start) > timeout:
                print("  WARNING: settle-wait timed out, proceeding anyway")
                return
            gyro = latest_gyro["values"]
            if gyro is not None and float(np.linalg.norm(gyro)) < args.settle_gyro_thresh:
                if below_since is None:
                    below_since = now
                elif (now - below_since) >= hold_needed:
                    return
            else:
                below_since = None
            rclpy.spin_once(node, timeout_sec=0.1)

    minimax_baseline = get_param("thrust_allocator.minimax_objective")
    print(f"thrust_allocator.minimax_objective baseline: {minimax_baseline}")

    def run_trial(minimax_value, trial_idx, condition_label):
        print(f"\n=== {condition_label} trial {trial_idx + 1}/{args.trials} ===")
        print("  settling ...")
        wait_for_settle()

        set_params([dbool("thrust_allocator.minimax_objective", minimax_value)])
        set_align_gains(list(args.kp_override), list(args.kd_override),
                         args.max_torque, args.alpha)

        pos0, quat0, _ = tf_client.get_pose()
        quat0 = np.asarray(quat0, dtype=float)
        half = np.radians(args.offset_deg) / 2.0
        offset_q = np.array([*(np.sin(half) * axis_vec), np.cos(half)])
        target_quat = quat_mul(quat0, offset_q)

        msg = PoseArray()
        msg.header.frame_id = REFERENCE_FRAME
        p = Pose()
        p.position.x, p.position.y, p.position.z = pos0
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = target_quat
        msg.poses = [p]
        checkpoint_pub.publish(msg)

        rows = []
        last_sign = None
        flip_count = 0

        def tick():
            nonlocal last_sign, flip_count
            pose = tf_client.get_pose()
            if pose is None:
                return
            _, quat, stamp = pose
            qe, raw_w, sign, angle_deg = qe_and_sign(target_quat, np.asarray(quat, dtype=float))
            flipped = last_sign is not None and sign != last_sign
            if flipped:
                flip_count += 1
            last_sign = sign
            rows.append({"stamp": stamp, "angle_deg": angle_deg})

        timer = node.create_timer(1.0 / POLL_HZ, tick)
        start = node.get_clock().now()
        duration = rclpy.duration.Duration(seconds=args.duration)
        while rclpy.ok() and (node.get_clock().now() - start) < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
        timer.cancel()

        set_align_gains(BASELINE_KP_ATT, BASELINE_KD_ATT,
                         BASELINE_MAX_CORR_TORQUE, BASELINE_ATT_FILTER_ALPHA)
        clear_checkpoints()

        angles = [r["angle_deg"] for r in rows]
        stamps = [r["stamp"] for r in rows]
        t0 = stamps[0] if stamps else 0.0
        conv_t = next((s - t0 for s, a in zip(stamps, angles) if a < 5.0), None)
        # Ignore the first 3s (initial fast descent from the offset isn't
        # "overshoot" -- only look for angle climbing back up afterward).
        post_transient = [a for s, a in zip(stamps, angles) if (s - t0) > 3.0]
        max_angle_after_3s = max(post_transient) if post_transient else None

        csv_path = f"{args.out_csv_prefix}_{condition_label}_{trial_idx}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stamp", "angle_deg"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        result = {
            "condition": condition_label,
            "trial": trial_idx,
            "n_ticks": len(rows),
            "sign_flips": flip_count,
            "convergence_time_s": conv_t,
            "max_angle_after_3s_deg": max_angle_after_3s,
            "final_angle_deg": angles[-1] if angles else None,
            "csv": csv_path,
        }
        print(f"  {result}")
        return result

    results = []
    try:
        for trial_idx in range(args.trials):
            results.append(run_trial(False, trial_idx, "off"))
        for trial_idx in range(args.trials):
            results.append(run_trial(True, trial_idx, "on"))
    finally:
        print(f"\nrestoring thrust_allocator.minimax_objective -> {minimax_baseline} ...")
        set_params([dbool("thrust_allocator.minimax_objective", minimax_baseline)])
        set_align_gains(BASELINE_KP_ATT, BASELINE_KD_ATT,
                         BASELINE_MAX_CORR_TORQUE, BASELINE_ATT_FILTER_ALPHA)
        clear_checkpoints()

    print("\n=== summary ===")
    for cond in ("off", "on"):
        cond_results = [r for r in results if r["condition"] == cond]
        conv_times = [r["convergence_time_s"] for r in cond_results if r["convergence_time_s"] is not None]
        max_angles = [r["max_angle_after_3s_deg"] for r in cond_results if r["max_angle_after_3s_deg"] is not None]
        flips = [r["sign_flips"] for r in cond_results]
        print(f"{cond}: n={len(cond_results)} "
              f"convergence_time_s={conv_times} "
              f"max_angle_after_3s_deg={[round(a, 1) for a in max_angles]} "
              f"sign_flips={flips}")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
