#!/usr/bin/env python3
"""Diagnostic: drive tf_correction's checkpoint attitude loop through a
controlled angle offset, logging the raw error/sign/duty-saturation/gyro
trace for gain-tuning investigation (see
docs/archive/achieved/2026-08-21_tf_correction_align_optimization.md).

Does NOT modify tf_correction's persisted config. Only:
  - sets tf_correction gains on control_node via SetParameters (runtime;
    restored to the current baseline afterward, or use --restore to just
    restore and exit)
  - publishes a single offset checkpoint to /gnc/checkpoints
  - polls TF (iss_body <- body), /ctl/duty, /imu/imu and /ctl/wrench,
    recomputing the quaternion error/sign locally with the same formula as
    pose_control_law.attitude_error_to_torque
  - logs every tick to CSV (including per-axis requested torque from
    /ctl/wrench vs. torque actually realized by /ctl/duty, reconstructed via
    the live thrust_allocator's wrench matrix) and flags sign-flip ticks

All timing is on the node clock (use_sim_time=True), no wall-clock sleep.

Usage:
    python3 test/manual/diagnose_align_gains.py --axis z --offset-deg 180 \\
        --kd-override 0.4635 0.4077 0.264 --max-torque 0.3 \\
        --out-csv /tmp/sign_flip_log.csv
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
from geometry_msgs.msg import PoseArray, Pose, WrenchStamped
from std_msgs.msg import Float64MultiArray
from ib2_msgs.msg import IMU

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
CHECKPOINT_TOPIC = "/gnc/checkpoints"
DUTY_TOPIC = "/ctl/duty"
IMU_TOPIC = "/imu/imu"
WRENCH_TOPIC = "/ctl/wrench"
WRENCH_TOTAL_TOPIC = "/ctl/wrench_total"
WRENCH_ACHIEVED_TOPIC = "/ctl/wrench_achieved"
POLL_HZ = 20.0
THRUST_ALLOCATOR_PARAM_NAMES = [
    "thrust_allocator.kj",
    "thrust_allocator.fj_max",
    "thrust_allocator.cg",
    "thrust_allocator.fan_positions",
    "thrust_allocator.fan_vectors",
    "thrust_allocator.force_weight_ref",
    "thrust_allocator.torque_weight_ref",
]
DURATION_S_DEFAULT = 25.0
OUT_CSV_DEFAULT = "/tmp/diagnose_align_gains_log.csv"

# tf_correction's current baseline (config/gnc_params.yaml), restored on exit.
BASELINE_KP_ATT = [0.01, 0.01, 0.01]
BASELINE_KD_ATT = [0.0, 0.0, 0.0]
BASELINE_MAX_CORR_TORQUE = 0.01
BASELINE_ATT_FILTER_ALPHA = 1.0

# Mid-angle gains verified in the 2026-08-21 investigation (theoretical
# zeta=0.9 value, times 3 -- see the doc above).
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
    pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=value_list)
    return ParameterMsg(name=name, value=pv)


def dscalar(name, value):
    pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
    return ParameterMsg(name=name, value=pv)


def dbool(name, value):
    pv = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
    return ParameterMsg(name=name, value=pv)


def _parameter_value_to_python(pv: ParameterValue):
    if pv.type == ParameterType.PARAMETER_DOUBLE:
        return pv.double_value
    if pv.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
        return list(pv.double_array_value)
    if pv.type == ParameterType.PARAMETER_BOOL:
        return pv.bool_value
    raise ValueError(f"unsupported parameter type {pv.type}")


def fetch_thrust_allocator(node: Node) -> ThrustAllocator:
    """Build a ThrustAllocator mirroring control_node's live thrust_allocator.*
    parameters (fetched via GetParameters, not the module's DEFAULT_* values),
    so the achieved-torque reconstruction below matches whatever weighting
    control_node is actually running with -- including force_weight_ref/
    torque_weight_ref, which are dynamically reconfigurable and may not match
    config/gnc_params.yaml's on-disk defaults if tuned live."""
    cli = node.create_client(GetParameters, "/control_node/get_parameters")
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("control_node get_parameters service unavailable")
    req = GetParameters.Request()
    req.names = THRUST_ALLOCATOR_PARAM_NAMES
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    if result is None:
        raise RuntimeError("get_parameters call to control_node failed")
    values = dict(zip(THRUST_ALLOCATOR_PARAM_NAMES, result.values))
    return ThrustAllocator(
        kj=_parameter_value_to_python(values["thrust_allocator.kj"]),
        fj_max=_parameter_value_to_python(values["thrust_allocator.fj_max"]),
        cg=_parameter_value_to_python(values["thrust_allocator.cg"]),
        fan_positions=_parameter_value_to_python(values["thrust_allocator.fan_positions"]),
        fan_vectors=_parameter_value_to_python(values["thrust_allocator.fan_vectors"]),
        force_weight_ref=_parameter_value_to_python(values["thrust_allocator.force_weight_ref"]),
        torque_weight_ref=_parameter_value_to_python(values["thrust_allocator.torque_weight_ref"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restore", action="store_true",
                     help="only restore baseline gains + clear checkpoints, then exit")
    ap.add_argument("--alpha", type=float, default=0.3,
                     help="tf_correction.att_filter_alpha to use (default: 0.3)")
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--kd-z", type=float, default=None,
                     help="override kd_att[2] (z axis) only, keeping the rest of --kd-override")
    ap.add_argument("--kd-override", type=float, nargs=3, default=DEFAULT_KD_ATT,
                     help="kd_att [x y z] for this run (default: 3x theoretical zeta=0.9 value)")
    ap.add_argument("--kp-override", type=float, nargs=3, default=DEFAULT_KP_ATT,
                     help="kp_att [x y z] for this run")
    ap.add_argument("--duration", type=float, default=DURATION_S_DEFAULT)
    ap.add_argument("--max-torque", type=float, default=DEFAULT_MAX_CORR_TORQUE)
    ap.add_argument("--offset-deg", type=float, default=180.0,
                     help="offset angle for the checkpoint (default: 180)")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z", "xyz"],
                     help="body-local rotation axis for the offset (default: z)")
    ap.add_argument("--preserve-direction", action="store_true",
                     help="set tf_correction.torque_direction_preserving=true "
                          "(scale all torque axes uniformly instead of clamping "
                          "independently -- see docs/2026-08-21_align_hold_gain_"
                          "oscillation_investigation.md)")
    ap.add_argument("--hover-max-torque", type=float, default=None,
                     help="temporarily override hover_control.max_torque (the "
                          "OUTER clamp applied after tf_correction's own clamp, "
                          "config default 0.02 Nm, independent-axis np.clip with "
                          "no preserve_direction option -- see docs/2026-08-27_"
                          "composite_axis_overshoot_next_steps.md) for this run "
                          "only; restored to its current live value afterward")
    ap.add_argument("--minimax", action="store_true",
                     help="temporarily set thrust_allocator.minimax_objective=true "
                          "(config default false -- L-infinity fit instead of the "
                          "default weighted L2 lsq_linear fit, meant to avoid "
                          "winner-takes-all single-axis-dominant allocation under "
                          "saturation, see docs/2026-08-27_thrust_allocator_single_"
                          "axis_saturation_findings.md) for this run only; "
                          "restored to its current live value afterward")
    args = ap.parse_args()
    out_csv = args.out_csv
    att_filter_alpha = args.alpha
    kd_att = list(args.kd_override)
    if args.kd_z is not None:
        kd_att[2] = args.kd_z
    max_corr_torque = args.max_torque
    kp_att = list(args.kp_override)
    duration_s = args.duration
    axis_vec = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "xyz": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }[args.axis]

    rclpy.init()
    node = Node("diagnose_align_gains")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    cli = node.create_client(SetParameters, "/control_node/set_parameters")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("control_node set_parameters service unavailable")
        return 1
    get_cli = node.create_client(GetParameters, "/control_node/get_parameters")
    if not get_cli.wait_for_service(timeout_sec=5.0):
        print("control_node get_parameters service unavailable")
        return 1

    def get_hover_max_torque():
        req = GetParameters.Request()
        req.names = ["hover_control.max_torque"]
        future = get_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError("get_parameters(hover_control.max_torque) failed")
        return _parameter_value_to_python(result.values[0])

    def set_hover_max_torque(value):
        req = SetParameters.Request()
        req.parameters = [dscalar("hover_control.max_torque", value)]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            reasons = [r.reason for r in (result.results if result else []) if not r.successful]
            print(f"WARNING: set_hover_max_torque failed: {reasons}")

    def get_minimax_objective():
        req = GetParameters.Request()
        req.names = ["thrust_allocator.minimax_objective"]
        future = get_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError("get_parameters(thrust_allocator.minimax_objective) failed")
        return _parameter_value_to_python(result.values[0])

    def set_minimax_objective(value):
        req = SetParameters.Request()
        req.parameters = [dbool("thrust_allocator.minimax_objective", value)]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            reasons = [r.reason for r in (result.results if result else []) if not r.successful]
            print(f"WARNING: set_minimax_objective failed: {reasons}")

    def set_gains(kp, kd, max_torque, filt, preserve_direction=False):
        # NOTE: tf_correction.kp_att/kd_att do NOT exist since the 2026-08-21
        # align/hold gain split (docs/archive/achieved/
        # 2026-08-21_tf_correction_align_hold_gain_split_design.md) --
        # setting them here used to silently no-op (SetParameters returns
        # successful=False for an undeclared parameter, which this function's
        # callers never checked). Target the _align variant: this script's
        # checkpoints are is_align=True the whole time (no align->hold
        # transition happens mid-offset for the angles this script tests).
        req = SetParameters.Request()
        req.parameters = [
            dparam("tf_correction.kp_att_align", kp),
            dparam("tf_correction.kd_att_align", kd),
            dscalar("tf_correction.max_corr_torque", max_torque),
            dscalar("tf_correction.att_filter_alpha", filt),
            dbool("tf_correction.torque_direction_preserving", preserve_direction),
        ]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            reasons = [r.reason for r in (result.results if result else []) if not r.successful]
            print(f"WARNING: set_gains partially/fully failed: {reasons}")
        return result

    checkpoint_pub = node.create_publisher(PoseArray, CHECKPOINT_TOPIC, 5)

    def clear_checkpoints():
        msg = PoseArray()
        msg.header.frame_id = REFERENCE_FRAME
        checkpoint_pub.publish(msg)

    if args.restore:
        set_gains(BASELINE_KP_ATT, BASELINE_KD_ATT, BASELINE_MAX_CORR_TORQUE,
                   BASELINE_ATT_FILTER_ALPHA)
        clear_checkpoints()
        print("restored baseline gains and cleared checkpoints")
        rclpy.shutdown()
        return 0

    tf_client = TfClient(node, REFERENCE_FRAME, TARGET_FRAME)
    if not tf_client.wait_for_frame(timeout_sec=10.0):
        print("TF unavailable, aborting")
        return 1
    pos0, quat0, _ = tf_client.get_pose()
    quat0 = np.asarray(quat0, dtype=float)

    allocator = fetch_thrust_allocator(node)

    # Offset about the body's local axis.
    half = np.radians(args.offset_deg) / 2.0
    offset_q = np.array([*(np.sin(half) * axis_vec), np.cos(half)])
    target_quat = quat_mul(quat0, offset_q)

    latest_duty = {"values": None}

    def on_duty(msg: Float64MultiArray):
        latest_duty["values"] = tuple(msg.data)

    node.create_subscription(Float64MultiArray, DUTY_TOPIC, on_duty, 10)

    latest_gyro = {"values": None}

    def on_imu(msg: IMU):
        latest_gyro["values"] = (msg.gyro_x, msg.gyro_y, msg.gyro_z)

    node.create_subscription(IMU, IMU_TOPIC, on_imu, 10)

    latest_req_torque = {"values": None}
    latest_req_force = {"values": None}

    def on_wrench(msg: WrenchStamped):
        latest_req_torque["values"] = (
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        )
        latest_req_force["values"] = (
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
        )

    node.create_subscription(WrenchStamped, WRENCH_TOPIC, on_wrench, 10)

    latest_req_torque_total = {"values": None}
    latest_req_force_total = {"values": None}

    def on_wrench_total(msg: WrenchStamped):
        latest_req_torque_total["values"] = (
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        )
        latest_req_force_total["values"] = (
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
        )

    node.create_subscription(WrenchStamped, WRENCH_TOTAL_TOPIC, on_wrench_total, 10)

    latest_ach_torque_sync = {"values": None}
    latest_ach_force_sync = {"values": None}

    def on_wrench_achieved(msg: WrenchStamped):
        latest_ach_torque_sync["values"] = (
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        )
        latest_ach_force_sync["values"] = (
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
        )

    node.create_subscription(WrenchStamped, WRENCH_ACHIEVED_TOPIC, on_wrench_achieved, 10)

    hover_max_torque_baseline = None
    if args.hover_max_torque is not None:
        hover_max_torque_baseline = get_hover_max_torque()
        print(f"overriding hover_control.max_torque {hover_max_torque_baseline} "
              f"-> {args.hover_max_torque} (restored after this run) ...")
        set_hover_max_torque(args.hover_max_torque)

    minimax_baseline = None
    if args.minimax:
        minimax_baseline = get_minimax_objective()
        print(f"overriding thrust_allocator.minimax_objective {minimax_baseline} "
              f"-> True (restored after this run) ...")
        set_minimax_objective(True)

    print("setting align gains via SetParameters ...")
    set_gains(kp_att, kd_att, max_corr_torque, att_filter_alpha)

    print(f"publishing {args.offset_deg}deg {args.axis}-axis offset checkpoint ...")
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
        pos, quat, stamp = pose
        qe, raw_w, sign, angle_deg = qe_and_sign(target_quat, np.asarray(quat, dtype=float))
        flipped = last_sign is not None and sign != last_sign
        if flipped:
            flip_count += 1
        last_sign = sign
        duty = latest_duty["values"]
        n_saturated = sum(1 for d in duty if d >= 0.999) if duty else None
        gyro = latest_gyro["values"]
        req_torque = latest_req_torque["values"]
        req_force = latest_req_force["values"]
        req_torque_total = latest_req_torque_total["values"]
        req_force_total = latest_req_force_total["values"]
        ach_torque_sync = latest_ach_torque_sync["values"]
        ach_force_sync = latest_ach_force_sync["values"]
        if duty is not None:
            # Invert _force_to_duty (duty = kj*sqrt(f)) to recover per-fan
            # thrust, then re-apply the allocator's wrench matrix to get the
            # force/torque actually realized by that duty -- as opposed to what
            # was requested (req_force/req_torque, pre-allocation/pre-clamp,
            # from /ctl/wrench).
            thrust = [(d / allocator.kj) ** 2 for d in duty]
            achieved = allocator.A @ np.asarray(thrust)
            ach_force = tuple(achieved[0:3])
            ach_torque = tuple(achieved[3:6])
        else:
            ach_force = None
            ach_torque = None
        rows.append({
            "stamp": stamp,
            "angle_deg": angle_deg,
            "qe_w": raw_w,
            "sign": sign,
            "sign_flip": flipped,
            "qe_x": qe[0], "qe_y": qe[1], "qe_z": qe[2],
            "gyro_x": gyro[0] if gyro else None,
            "gyro_y": gyro[1] if gyro else None,
            "gyro_z": gyro[2] if gyro else None,
            "n_saturated": n_saturated,
            **{f"duty_{i}": (duty[i] if duty else None) for i in range(8)},
            "req_fx": req_force[0] if req_force else None,
            "req_fy": req_force[1] if req_force else None,
            "req_fz": req_force[2] if req_force else None,
            "req_tx": req_torque[0] if req_torque else None,
            "req_ty": req_torque[1] if req_torque else None,
            "req_tz": req_torque[2] if req_torque else None,
            "req_total_tx": req_torque_total[0] if req_torque_total else None,
            "req_total_ty": req_torque_total[1] if req_torque_total else None,
            "req_total_tz": req_torque_total[2] if req_torque_total else None,
            "req_total_fx": req_force_total[0] if req_force_total else None,
            "req_total_fy": req_force_total[1] if req_force_total else None,
            "req_total_fz": req_force_total[2] if req_force_total else None,
            "ach_fx": ach_force[0] if ach_force else None,
            "ach_fy": ach_force[1] if ach_force else None,
            "ach_fz": ach_force[2] if ach_force else None,
            "ach_tx": ach_torque[0] if ach_torque else None,
            "ach_ty": ach_torque[1] if ach_torque else None,
            "ach_tz": ach_torque[2] if ach_torque else None,
            "ach_sync_fx": ach_force_sync[0] if ach_force_sync else None,
            "ach_sync_fy": ach_force_sync[1] if ach_force_sync else None,
            "ach_sync_fz": ach_force_sync[2] if ach_force_sync else None,
            "ach_sync_tx": ach_torque_sync[0] if ach_torque_sync else None,
            "ach_sync_ty": ach_torque_sync[1] if ach_torque_sync else None,
            "ach_sync_tz": ach_torque_sync[2] if ach_torque_sync else None,
        })

    timer = node.create_timer(1.0 / POLL_HZ, tick)
    start = node.get_clock().now()
    duration = rclpy.duration.Duration(seconds=duration_s)
    while rclpy.ok() and (node.get_clock().now() - start) < duration:
        rclpy.spin_once(node, timeout_sec=0.1)
    timer.cancel()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"logged {len(rows)} ticks, {flip_count} sign flips -> {out_csv}")

    print("\n--- sign-flip events (angle before -> after, n_saturated before -> after) ---")
    for idx, r in enumerate(rows):
        if r["sign_flip"] and idx > 0:
            prev = rows[idx - 1]
            print(
                f"t={r['stamp']:.2f}s  angle {prev['angle_deg']:.1f}->{r['angle_deg']:.1f} deg  "
                f"qe_w {prev['qe_w']:.4f}->{r['qe_w']:.4f}  "
                f"n_saturated {prev['n_saturated']}->{r['n_saturated']}"
            )

    print("\nrestoring baseline gains and clearing checkpoints ...")
    set_gains(BASELINE_KP_ATT, BASELINE_KD_ATT, BASELINE_MAX_CORR_TORQUE,
              BASELINE_ATT_FILTER_ALPHA)
    if hover_max_torque_baseline is not None:
        print(f"restoring hover_control.max_torque -> {hover_max_torque_baseline} ...")
        set_hover_max_torque(hover_max_torque_baseline)
    if minimax_baseline is not None:
        print(f"restoring thrust_allocator.minimax_objective -> {minimax_baseline} ...")
        set_minimax_objective(minimax_baseline)
    clear_checkpoints()

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
