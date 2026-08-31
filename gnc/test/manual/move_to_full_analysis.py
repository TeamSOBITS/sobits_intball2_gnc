#!/usr/bin/env python3
"""One-shot move_to verification: drives a real move_to goal (MoveToClient)
while recording everything relevant, event-driven (TF, `/gnc/trajectory_setpoint`,
`/ctl/wrench`, `/ctl/wrench_achieved`, `/ctl/duty`), then computes and prints a
single tracking-quality report -- position/attitude tracking error, fan-duty
saturation, wrench desired-vs-achieved, and final arrival accuracy -- without
needing a separate script per metric or manual CSV post-processing.

Supersedes running `move_to_wrench_duty_trace.py` +
`measure_position_tracking_error.py` + `measure_attitude_tracking_error.py`
separately: those each recorded/measured one slice of this (see
test/manual/README.md); this merges the event-driven multi-topic recording
approach from `log_replanning_attitude_trace.py` (no dropped samples under
CPU load) with all four topics from `move_to_wrench_duty_trace.py`
(`/ctl/wrench_achieved` included) plus the actual error computation
(position/attitude vs setpoint) that neither of those scripts did.

Works with any `guidance.trajectory_tracking_mode` value (``static``,
``replanning``, ``replanning_minco``, ``static_minco``, ...) -- pass
`--set-mode` to set it first, or leave whatever is already configured.

Usage:
    python3 test/manual/move_to_full_analysis.py nav_entry
    python3 test/manual/move_to_full_analysis.py nav_entry --set-mode static_minco --out-dir /tmp/trace --tag run1

Raw per-topic CSVs are still written to `--out-dir` (tf/setpoint/wrench/
wrench_achieved/duty/tracking_error) for deeper inspection, but are written
even on Ctrl-C/exception (not only on clean completion) -- unlike
`log_replanning_attitude_trace.py`, which loses everything if killed before
its single end-of-run `write_csv` call.
"""
import argparse
import csv
import os
import sys

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from trajectory_msgs.msg import MultiDOFJointTrajectory

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul
from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
GUIDANCE_NODE = "guidance_node"
DUTY_SATURATION_THRESHOLD = 0.99

_NO_WAIT = rclpy.duration.Duration(seconds=0)

# Matches the actual publishers' QoS -- see log_replanning_attitude_trace.py's
# equivalent comments (a RELIABLE reader on a BEST_EFFORT /tf publisher would
# silently receive nothing at all).
TF_QOS = QoSProfile(
    depth=200, durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
)
RELIABLE_QOS = QoSProfile(
    depth=200, durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
)


def geodesic_angle_deg(q_a, q_b):
    """Quaternion geodesic angle [deg] between two [x,y,z,w] orientations."""
    qe = quat_mul(quat_conj(np.asarray(q_a)), np.asarray(q_b))
    w = float(np.clip(abs(qe[3]), 0.0, 1.0))
    return np.degrees(2.0 * np.arccos(w))


class TfRawRecorder:
    """Records every distinct ``iss_body <- body`` TF sample, event-driven.

    Owns its own ``tf2_ros.Buffer`` (not ``TfClient``'s -- same reasoning as
    ``log_replanning_attitude_trace.py``'s ``TfRawRecorder``: sharing a buffer
    via a second subscription would race the read against the write for the
    same incoming message).
    """

    def __init__(self, node):
        self._buffer = tf2_ros.Buffer()
        self._last_stamp = None
        self.rows = []  # (t_sim, px, py, pz, qx, qy, qz, qw)
        node.create_subscription(TFMessage, "/tf", self._on_tf, TF_QOS)

    def _on_tf(self, msg):
        for transform in msg.transforms:
            self._buffer.set_transform(transform, "move_to_full_analysis")
        try:
            t = self._buffer.lookup_transform(
                REFERENCE_FRAME, TARGET_FRAME, rclpy.time.Time(), timeout=_NO_WAIT
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if stamp == self._last_stamp:
            return
        self._last_stamp = stamp
        tr, q = t.transform.translation, t.transform.rotation
        self.rows.append((stamp, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))


class SetpointRawRecorder:
    """Records every ``/gnc/trajectory_setpoint`` message, in arrival order."""

    def __init__(self, node):
        self.rows = []  # (t_sim, px,py,pz, qx,qy,qz,qw)
        node.create_subscription(
            MultiDOFJointTrajectory, "/gnc/trajectory_setpoint", self._on_msg, RELIABLE_QOS
        )

    def _on_msg(self, msg):
        if not msg.points:
            return
        point = msg.points[0]
        tr = point.transforms[0].translation
        q = point.transforms[0].rotation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rows.append((stamp, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))


class WrenchRecorder:
    """Records ``/ctl/wrench`` (desired) or ``/ctl/wrench_achieved``
    (event-driven, whichever topic it's pointed at)."""

    def __init__(self, node, topic):
        self.rows = []  # (t_sim, fx, fy, fz, tx, ty, tz)
        node.create_subscription(WrenchStamped, topic, self._on_msg, RELIABLE_QOS)

    def _on_msg(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        f, t = msg.wrench.force, msg.wrench.torque
        self.rows.append((stamp, f.x, f.y, f.z, t.x, t.y, t.z))


class DutyRecorder:
    """Records ``/ctl/duty`` (per-fan duty, ``[0, 1]``) -- no header/stamp on
    ``Float64MultiArray``, so timestamped on reception with the node's own
    (sim-time) clock instead."""

    def __init__(self, node):
        self._node = node
        self.rows = []  # (t_sim, duty0, ..., dutyN)
        node.create_subscription(Float64MultiArray, "/ctl/duty", self._on_msg, RELIABLE_QOS)

    def _on_msg(self, msg):
        t = self._node.get_clock().now().nanoseconds * 1e-9
        self.rows.append((t,) + tuple(msg.data))


def set_tracking_mode(node, mode):
    client = node.create_client(SetParameters, f"/{GUIDANCE_NODE}/set_parameters")
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(
            "[move_to_full_analysis] /guidance_node/set_parameters unavailable "
            "-- is guidance_node running?"
        )
        return False
    value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
    req = SetParameters.Request(
        parameters=[ParameterMsg(name="guidance.trajectory_tracking_mode", value=value)]
    )
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    result = future.result()
    ok = result is not None and result.results and result.results[0].successful
    if not ok:
        reason = result.results[0].reason if result and result.results else "no response"
        node.get_logger().error(
            "[move_to_full_analysis] failed to set trajectory_tracking_mode=%s (%s)"
            % (mode, reason)
        )
    return ok


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def nearest_match_errors(tf_rows, sp_rows):
    """For each TF sample, find the nearest-in-time setpoint sample (both
    lists are already time-ordered, event-driven) and compute position/
    attitude tracking error against it. Two-pointer merge, O(n)."""
    rows = []
    j = 0
    n = len(sp_rows)
    for tf_row in tf_rows:
        t = tf_row[0]
        while j + 1 < n and abs(sp_rows[j + 1][0] - t) <= abs(sp_rows[j][0] - t):
            j += 1
        if n == 0:
            continue
        sp_row = sp_rows[j]
        pos_now = np.array(tf_row[1:4])
        quat_now = np.array(tf_row[4:8])
        p_des = np.array(sp_row[1:4])
        q_des = np.array(sp_row[4:8])
        pos_err_m = float(np.linalg.norm(pos_now - p_des))
        att_err_deg = geodesic_angle_deg(q_des, quat_now)
        rows.append((t, pos_err_m, att_err_deg))
    return rows


def summarize(label, values, unit, fmt="%.3f"):
    if not values:
        print("  %s: no samples" % label)
        return
    print(
        ("  %s: max=" + fmt + "%s mean=" + fmt + "%s final=" + fmt + "%s (n=%d)")
        % (label, max(values), unit, sum(values) / len(values), unit, values[-1], unit, len(values))
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("location_name", help="destination TF frame name (maps/iss_location.yaml)")
    ap.add_argument("--set-mode", default=None,
                     help="set guidance.trajectory_tracking_mode before sending the goal "
                          "(any value the node accepts, e.g. static/replanning/"
                          "replanning_minco/static_minco); default: leave as-is")
    ap.add_argument("--out-dir", default="/tmp/move_to_full_analysis")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--timeout-sec", type=float, default=90.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    node = Node("move_to_full_analysis")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    tf_rec = TfRawRecorder(node)
    sp_rec = SetpointRawRecorder(node)
    wrench_rec = WrenchRecorder(node, "/ctl/wrench")
    wrench_achieved_rec = WrenchRecorder(node, "/ctl/wrench_achieved")
    duty_rec = DutyRecorder(node)
    move_client = MoveToClient(node)

    if args.set_mode is not None:
        if not set_tracking_mode(node, args.set_mode):
            node.destroy_node()
            rclpy.shutdown()
            return 1

    resolved = move_client.resolve_location(args.location_name)
    if resolved is None:
        print("could not resolve '%s' via TF" % args.location_name)
        node.destroy_node()
        rclpy.shutdown()
        return 1
    target_pos, target_quat = resolved
    print("resolved %s -> pos=%s quat=%s, sending goal..." % (args.location_name, target_pos, target_quat))

    result = None
    try:
        result = move_client.send_goal(target_pos, target_quat, timeout_sec=args.timeout_sec)
    except KeyboardInterrupt:
        print("interrupted -- analyzing whatever was recorded so far")
    except Exception as exc:  # noqa: BLE001 -- still want the analysis below
        print("send_goal raised %r -- analyzing whatever was recorded so far" % exc)

    # --- write raw per-topic CSVs (even on interrupt/exception above) ---
    write_csv(os.path.join(args.out_dir, "%s_tf.csv" % args.tag),
              ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw"], tf_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_setpoint.csv" % args.tag),
              ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw"], sp_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_wrench.csv" % args.tag),
              ["t_sim", "fx", "fy", "fz", "tx", "ty", "tz"], wrench_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_wrench_achieved.csv" % args.tag),
              ["t_sim", "fx", "fy", "fz", "tx", "ty", "tz"], wrench_achieved_rec.rows)
    n_fans = (len(duty_rec.rows[0]) - 1) if duty_rec.rows else 8
    write_csv(os.path.join(args.out_dir, "%s_duty.csv" % args.tag),
              ["t_sim"] + ["duty%d" % i for i in range(n_fans)], duty_rec.rows)

    tracking_rows = nearest_match_errors(tf_rec.rows, sp_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_tracking_error.csv" % args.tag),
              ["t_sim", "pos_error_m", "attitude_error_deg"], tracking_rows)

    # --- report ---
    print()
    print("=== move_to_full_analysis report: %s -> %s ===" % (args.location_name, args.tag))
    print("goal result: %s" % ("None (no result)" if result is None else "type=%d" % result.type))
    print()
    print("tracking error (TF vs /gnc/trajectory_setpoint, nearest-time matched):")
    summarize("position error", [r[1] * 1000 for r in tracking_rows], "mm", fmt="%.1f")
    summarize("attitude error", [r[2] for r in tracking_rows], "deg")
    print()

    if duty_rec.rows:
        duty_values = [row[1:] for row in duty_rec.rows]
        max_duty_per_sample = [max(row) for row in duty_values]
        n_saturated = sum(1 for m in max_duty_per_sample if m >= DUTY_SATURATION_THRESHOLD)
        print("fan duty (%d fans, saturation threshold=%.2f):" % (n_fans, DUTY_SATURATION_THRESHOLD))
        print("  max duty overall=%.3f, samples saturated=%d/%d (%.1f%%)"
              % (max(max_duty_per_sample), n_saturated, len(max_duty_per_sample),
                 100.0 * n_saturated / len(max_duty_per_sample)))
    else:
        print("fan duty: no samples")
    print()

    if wrench_rec.rows:
        force_norms = [float(np.linalg.norm(r[1:4])) for r in wrench_rec.rows]
        torque_norms = [float(np.linalg.norm(r[4:7])) for r in wrench_rec.rows]
        print("commanded wrench (desired, before fan allocation):")
        summarize("|force|", force_norms, "N")
        summarize("|torque|", torque_norms, "Nm")
    else:
        print("commanded wrench: no samples")
    if wrench_achieved_rec.rows:
        force_norms_ach = [float(np.linalg.norm(r[1:4])) for r in wrench_achieved_rec.rows]
        torque_norms_ach = [float(np.linalg.norm(r[4:7])) for r in wrench_achieved_rec.rows]
        summarize("|force| achieved", force_norms_ach, "N")
        summarize("|torque| achieved", torque_norms_ach, "Nm")
    print()

    if tf_rec.rows:
        last = tf_rec.rows[-1]
        final_pos_err_m = float(np.linalg.norm(np.array(last[1:4]) - np.array(target_pos)))
        final_att_err_deg = geodesic_angle_deg(target_quat, last[4:8])
        print("final arrival accuracy (last TF sample vs resolved goal):")
        print("  position error=%.1fmm, attitude error=%.3fdeg" % (final_pos_err_m * 1000, final_att_err_deg))
    print()
    print("raw CSVs written to %s/%s_*.csv" % (args.out_dir, args.tag))

    node.destroy_node()
    rclpy.shutdown()
    return 0 if (result is not None and result.type == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
