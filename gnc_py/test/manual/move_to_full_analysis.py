#!/usr/bin/env python3
"""One-shot move_to verification: drives a real move_to goal (CtlCommandActionClient)
while recording everything relevant, event-driven (TF, `/gnc/trajectory_setpoint`,
`/gnc/checkpoints`, `/ctl/wrench_correction`, `/ctl/wrench_achieved`, `/ctl/duty`, and every published
global/local path preview `/gnc/trajectory_path_speed(_local)`), then computes and prints a
single tracking-quality report -- position/attitude tracking error, fan-duty
saturation, wrench desired-vs-achieved, and final arrival accuracy -- without
needing a separate script per metric or manual CSV post-processing.

Supersedes running `move_to_wrench_duty_trace.py` +
`measure_position_tracking_error.py` + `measure_attitude_tracking_error.py`
separately: those each recorded/measured one slice of this (see
test/manual/README.md); this merges event-driven multi-topic recording
(no dropped samples under CPU load) with all four topics from `move_to_wrench_duty_trace.py`
(`/ctl/wrench_achieved` included) plus the actual error computation
(position/attitude vs setpoint) that neither of those scripts did.

Works with any `guidance.trajectory_tracking_mode` value (``static_toppra``,
``static_minco``, ``static_min_snap``, ``replan_minco``) -- pass
`--set-mode` to set it first, or leave whatever is already configured.

Also works against the JAXA move function (``/ctl/command_ros2``, goal frame
``dock_body``): pass ``--action-name /ctl/command_ros2 --reference-frame dock_body``
(run ``location_broadcaster`` so the location frames exist). The JAXA side publishes
no reference trajectory, so the setpoint-based tracking error reads "no samples";
the straight-line section below is what compares both stacks. All recorded poses
and the CSVs are then expressed in ``--reference-frame``.

Straight-line section (printed for every run): the line runs from the position at
goal send to the target. Lateral deviation is the distance from that line (and
from the line to the *measured* final position, which removes a constant offset
between the controller's pose and the TF ``body``). Motion time runs from the
first move beyond ``--move-threshold-mm`` to the last time the position was
farther than ``--settle-tol-mm`` from its final value; "reached" is the final
position error against the target within ``--reach-tol-mm``.

Usage:
    python3 test/manual/move_to_full_analysis.py nav_entry
    python3 test/manual/move_to_full_analysis.py nav_entry --set-mode static_minco --out-dir /tmp/trace --tag run1
    python3 test/manual/move_to_full_analysis.py jaxa_paper_goal --action-name /ctl/command_ros2 --reference-frame dock_body

Tracking error is split by phase: ``trajectory`` samples are compared with the
nearest-in-time `/gnc/trajectory_setpoint`; ``align`` samples (the latest
reference received is a `/gnc/checkpoints` hold, i.e. pre_align/
align_at_arrival) are compared with that checkpoint instead -- the aligner
never updates the setpoint topic, so comparing against it would report the
stale trajectory attitude.

`--timeout-sec` bounds the wait for the goal result in sim time; on expiry
the goal is canceled and whatever was recorded is analyzed.

Raw per-topic CSVs are still written to `--out-dir` (tf/setpoint/checkpoint/
wrench/wrench_achieved/duty/tracking_error/global_path/local_path) for deeper inspection, but are
written even on Ctrl-C/exception (not only on clean completion).
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
from geometry_msgs.msg import PoseArray, WrenchStamped
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
from visualization_msgs.msg import Marker

from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_mul
from sobits_intball2_gnc.guidance.move_to import resolve_location
from sobits_intball2_gnc.guidance.ros.ctl_command_action_client import (
    ACTION_NAME,
    CtlCommandActionClient,
)

REFERENCE_FRAME = "iss_body"
STATIC_QOS = QoSProfile(
    depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
)
TARGET_FRAME = "body"
GUIDANCE_NODE = "guidance_node"
DUTY_SATURATION_THRESHOLD = 0.99
PHASE_TRAJECTORY = "trajectory"
PHASE_ALIGN = "align"

_NO_WAIT = rclpy.duration.Duration(seconds=0)

# Matches the actual publishers' QoS (a RELIABLE reader on a BEST_EFFORT /tf
# publisher would silently receive nothing at all).
TF_QOS = QoSProfile(
    depth=200, durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT,
)
RELIABLE_QOS = QoSProfile(
    depth=200, durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
)
# SpeedPathPublisher is transient-local; a volatile reader would still work, but
# depth must cover a burst of replans between spins.
PATH_QOS = QoSProfile(
    depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
)


def geodesic_angle_deg(q_a, q_b):
    """Quaternion geodesic angle [deg] between two [x,y,z,w] orientations."""
    qe = quat_mul(quat_conj(np.asarray(q_a)), np.asarray(q_b))
    w = float(np.clip(abs(qe[3]), 0.0, 1.0))
    return np.degrees(2.0 * np.arccos(w))


class TfRawRecorder:
    """Records every distinct ``reference_frame <- body`` TF sample, event-driven.

    Owns its own ``tf2_ros.Buffer`` (not ``TfClient``'s -- sharing a buffer
    via a second subscription would race the read against the write for the
    same incoming message).

    A reference frame other than ``iss_body`` (JAXA's ``dock_body``) is a
    static transform, so ``/tf_static`` is read too -- but only that frame's
    own transform: ``/tf_static`` also holds a frozen identity for ``iss_body``
    and ``body`` that would shadow the real pose.
    """

    def __init__(self, node, reference_frame=REFERENCE_FRAME):
        self._buffer = tf2_ros.Buffer()
        self._reference_frame = reference_frame
        self._last_stamp = None
        self.rows = []  # (t_sim, px, py, pz, qx, qy, qz, qw)
        node.create_subscription(TFMessage, "/tf", self._on_tf, TF_QOS)
        if reference_frame != REFERENCE_FRAME:
            node.create_subscription(TFMessage, "/tf_static", self._on_tf_static, STATIC_QOS)

    def _on_tf_static(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == self._reference_frame:
                self._buffer.set_transform_static(transform, "move_to_full_analysis")

    def _on_tf(self, msg):
        for transform in msg.transforms:
            self._buffer.set_transform(transform, "move_to_full_analysis")
        try:
            t = self._buffer.lookup_transform(
                self._reference_frame, TARGET_FRAME, rclpy.time.Time(), timeout=_NO_WAIT
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


class CheckpointRecorder:
    """Records every ``/gnc/checkpoints`` hold target (first pose only --
    Guidance always publishes a single-pose array)."""

    def __init__(self, node):
        self.rows = []  # (t_sim, px,py,pz, qx,qy,qz,qw)
        node.create_subscription(PoseArray, "/gnc/checkpoints", self._on_msg, RELIABLE_QOS)

    def _on_msg(self, msg):
        if not msg.poses:
            return
        pose = msg.poses[0]
        tr, q = pose.position, pose.orientation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rows.append((stamp, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))


class PathRecorder:
    """Records every speed-path preview (a ``LINE_STRIP`` marker) as numbered paths,
    so the planned shape (not just the sampled setpoint) can be inspected afterwards."""

    def __init__(self, node, topic):
        self.rows = []  # (t_sim, path_id, point_idx, x, y, z)
        self._count = 0
        node.create_subscription(Marker, topic, self._on_msg, PATH_QOS)

    def _on_msg(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for i, pt in enumerate(msg.points):
            self.rows.append((stamp, self._count, i, pt.x, pt.y, pt.z))
        self._count += 1


class WrenchRecorder:
    """Records ``/ctl/wrench_correction`` (desired) or ``/ctl/wrench_achieved``
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


def tracking_errors(tf_rows, sp_rows, cp_rows):
    """For each TF sample, pick the reference the controller is following
    and compute position/attitude error against it: the latest checkpoint if
    it arrived after the latest setpoint (align phase), otherwise the
    nearest-in-time setpoint. TF samples before the first reference are
    skipped. All lists are time-ordered; O(n)."""
    rows = []
    n_sp, n_cp = len(sp_rows), len(cp_rows)
    j = 0            # nearest setpoint
    sp_before = -1   # latest setpoint with stamp <= t
    cp_before = -1   # latest checkpoint with stamp <= t
    for tf_row in tf_rows:
        t = tf_row[0]
        while j + 1 < n_sp and abs(sp_rows[j + 1][0] - t) <= abs(sp_rows[j][0] - t):
            j += 1
        while sp_before + 1 < n_sp and sp_rows[sp_before + 1][0] <= t:
            sp_before += 1
        while cp_before + 1 < n_cp and cp_rows[cp_before + 1][0] <= t:
            cp_before += 1
        # Before Guidance's first reference the vehicle holds a target this
        # script never saw; the nearest future setpoint is not it.
        if sp_before < 0 and cp_before < 0:
            continue
        in_align = cp_before >= 0 and (
            sp_before < 0 or cp_rows[cp_before][0] > sp_rows[sp_before][0]
        )
        if in_align:
            phase, ref_row = PHASE_ALIGN, cp_rows[cp_before]
        elif n_sp > 0:
            phase, ref_row = PHASE_TRAJECTORY, sp_rows[j]
        else:
            continue
        pos_err_m = float(np.linalg.norm(np.array(tf_row[1:4]) - np.array(ref_row[1:4])))
        att_err_deg = geodesic_angle_deg(ref_row[4:8], tf_row[4:8])
        rows.append((t, phase, pos_err_m, att_err_deg))
    return rows


def straight_line_metrics(tf_rows, t_send, target_pos, t_result=None,
                          move_threshold_m=0.005, settle_tol_m=0.010, reach_tol_m=0.050):
    """Return straight-line metrics from the TF samples, or None without samples.

    ``tf_rows``: ``(t, px, py, pz, ...)`` in time order. The start is the last
    sample at or before ``t_send`` (the first sample if none). Lateral deviation
    is the perpendicular distance to the line start->``target_pos`` and to the
    line start->measured final position. Motion time runs from the first sample
    farther than ``move_threshold_m`` from the start to the last sample farther
    than ``settle_tol_m`` from the final position (``None`` if it never moved).
    """
    if len(tf_rows) < 2:
        return None
    t = np.array([r[0] for r in tf_rows])
    p = np.array([r[1:4] for r in tf_rows], dtype=float)
    i0 = max(int(np.searchsorted(t, t_send, side="right")) - 1, 0)
    t, p = t[i0:], p[i0:]
    start, final = p[0], p[-1]
    target = np.asarray(target_pos, dtype=float)

    def lateral(end):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        direction = direction / length
        rel = p - start
        return np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1)

    moved = np.where(np.linalg.norm(p - start, axis=1) > move_threshold_m)[0]
    out = {
        "line_length_m": float(np.linalg.norm(target - start)),
        "travelled_m": float(np.linalg.norm(final - start)),
        "lateral_target_m": lateral(target),
        "lateral_final_m": lateral(final),
        "final_error_m": float(np.linalg.norm(final - target)),
        "reached": bool(np.linalg.norm(final - target) <= reach_tol_m),
        "action_time_s": None if t_result is None else float(t_result - t_send),
        "motion_time_s": None,
    }
    if moved.size:
        t_move = float(t[moved[0]])
        unsettled = np.where(np.linalg.norm(p - final, axis=1) > settle_tol_m)[0]
        t_settle = float(t[unsettled[-1]]) if unsettled.size else t_move
        out["motion_time_s"] = max(t_settle - t_move, 0.0)
    return out


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
                          "(any value the node accepts: static/static_minco/"
                          "replan_minco); default: leave as-is")
    ap.add_argument("--action-name", default=ACTION_NAME,
                     help="action server to send the goal to (default: %(default)s; "
                          "JAXA move function: /ctl/command_ros2)")
    ap.add_argument("--reference-frame", default=REFERENCE_FRAME,
                     help="frame the goal and the recorded poses are expressed in "
                          "(default: %(default)s; JAXA: dock_body)")
    ap.add_argument("--move-threshold-mm", type=float, default=5.0,
                     help="motion starts when the position leaves the start by this much")
    ap.add_argument("--settle-tol-mm", type=float, default=10.0,
                     help="motion ends at the last sample farther than this from the final position")
    ap.add_argument("--reach-tol-mm", type=float, default=50.0,
                     help="'reached' when the final position error to the target is within this")
    ap.add_argument("--out-dir", default="/tmp/move_to_full_analysis")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--timeout-sec", type=float, default=90.0,
                     help="sim-time limit on the goal result; the goal is canceled on expiry")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    node = Node("move_to_full_analysis")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    tf_rec = TfRawRecorder(node, args.reference_frame)
    sp_rec = SetpointRawRecorder(node)
    cp_rec = CheckpointRecorder(node)
    wrench_rec = WrenchRecorder(node, "/ctl/wrench_correction")
    wrench_achieved_rec = WrenchRecorder(node, "/ctl/wrench_achieved")
    duty_rec = DutyRecorder(node)
    global_path_rec = PathRecorder(node, "/gnc/trajectory_path_speed")
    local_path_rec = PathRecorder(node, "/gnc/trajectory_path_speed_local")
    move_client = CtlCommandActionClient(node, args.action_name, args.reference_frame)

    if args.set_mode is not None:
        if not set_tracking_mode(node, args.set_mode):
            node.destroy_node()
            rclpy.shutdown()
            return 1

    resolved = resolve_location(node, args.location_name, args.reference_frame)
    if resolved is None:
        print("could not resolve '%s' via TF" % args.location_name)
        node.destroy_node()
        rclpy.shutdown()
        return 1
    target_pos, target_quat = resolved
    print("resolved %s -> pos=%s quat=%s, sending goal..." % (args.location_name, target_pos, target_quat))

    result = None
    t_result = None
    t_send = node.get_clock().now().nanoseconds * 1e-9
    try:
        result = move_client.send_goal(
            target_pos, target_quat, result_timeout_sec=args.timeout_sec
        )
        t_result = node.get_clock().now().nanoseconds * 1e-9
    except KeyboardInterrupt:
        print("interrupted -- analyzing whatever was recorded so far")
    except Exception as exc:  # noqa: BLE001 -- still want the analysis below
        print("send_goal raised %r -- analyzing whatever was recorded so far" % exc)

    # --- write raw per-topic CSVs (even on interrupt/exception above) ---
    write_csv(os.path.join(args.out_dir, "%s_tf.csv" % args.tag),
              ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw"], tf_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_setpoint.csv" % args.tag),
              ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw"], sp_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_checkpoint.csv" % args.tag),
              ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw"], cp_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_wrench.csv" % args.tag),
              ["t_sim", "fx", "fy", "fz", "tx", "ty", "tz"], wrench_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_wrench_achieved.csv" % args.tag),
              ["t_sim", "fx", "fy", "fz", "tx", "ty", "tz"], wrench_achieved_rec.rows)
    for name, rec in (("global_path", global_path_rec), ("local_path", local_path_rec)):
        write_csv(os.path.join(args.out_dir, "%s_%s.csv" % (args.tag, name)),
                  ["t_sim", "path_id", "point_idx", "px", "py", "pz"], rec.rows)
    n_fans = (len(duty_rec.rows[0]) - 1) if duty_rec.rows else 8
    write_csv(os.path.join(args.out_dir, "%s_duty.csv" % args.tag),
              ["t_sim"] + ["duty%d" % i for i in range(n_fans)], duty_rec.rows)

    tracking_rows = tracking_errors(tf_rec.rows, sp_rec.rows, cp_rec.rows)
    write_csv(os.path.join(args.out_dir, "%s_tracking_error.csv" % args.tag),
              ["t_sim", "phase", "pos_error_m", "attitude_error_deg"], tracking_rows)

    # --- report ---
    print()
    print("=== move_to_full_analysis report: %s -> %s ===" % (args.location_name, args.tag))
    print("goal result: %s" % ("None (no result)" if result is None else "type=%d" % result.type))
    print()
    for phase, title in (
        (PHASE_TRAJECTORY, "trajectory tracking error (TF vs /gnc/trajectory_setpoint, nearest-time matched):"),
        (PHASE_ALIGN, "align hold error (TF vs latest /gnc/checkpoints, pre_align/align_at_arrival):"),
    ):
        phase_rows = [r for r in tracking_rows if r[1] == phase]
        print(title)
        summarize("position error", [r[2] * 1000 for r in phase_rows], "mm", fmt="%.1f")
        summarize("attitude error", [r[3] for r in phase_rows], "deg")
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

    line = straight_line_metrics(
        tf_rec.rows, t_send, target_pos, t_result,
        move_threshold_m=args.move_threshold_mm / 1000.0,
        settle_tol_m=args.settle_tol_mm / 1000.0,
        reach_tol_m=args.reach_tol_mm / 1000.0,
    )
    print("straight-line metrics (poses in %s; line = start at goal send -> target):" % args.reference_frame)
    if line is None:
        print("  no TF samples")
    else:
        print("  line length=%.3fm, travelled=%.3fm" % (line["line_length_m"], line["travelled_m"]))
        for key, label in (("lateral_target_m", "lateral deviation from start->target line"),
                           ("lateral_final_m", "lateral deviation from start->final line")):
            if line[key] is None:
                print("  %s: n/a (no motion)" % label)
            else:
                print("  %s: max=%.1fmm mean=%.1fmm" % (
                    label, 1000.0 * float(np.max(line[key])), 1000.0 * float(np.mean(line[key]))))
        print("  motion time=%s, time to goal result=%s"
              % ("n/a" if line["motion_time_s"] is None else "%.1fs" % line["motion_time_s"],
                 "n/a" if line["action_time_s"] is None else "%.1fs" % line["action_time_s"]))
        print("  reached (final error %.1fmm <= %.0fmm): %s"
              % (1000.0 * line["final_error_m"], args.reach_tol_mm, "yes" if line["reached"] else "NO"))
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
