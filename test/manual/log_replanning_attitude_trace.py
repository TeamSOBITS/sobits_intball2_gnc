#!/usr/bin/env python3
"""Combined TF + `/gnc/trajectory_setpoint` recorder for the replanning-mode
attitude-pointing investigation (`docs/archive/achieved/2026-08-25_guidance_realtime_replanning_sim_verification.md`
6 節). Drives one or more real `move_to` goals (via `MoveToClient`) while
recording every single incoming sample on both streams, event-driven (no
polling timer, no downsampling) -- so no tick is silently dropped the way a
fixed-rate logging timer could drop one under load
(`docs/recording_cpu_load_control_degradation.md`'s bursty-TF finding is
exactly the failure mode this script is designed not to reproduce on the
recording side).

Also records `/ctl/wrench` (commanded force/torque, the attitude
controller's output before fan allocation) and `/ctl/duty` (per-fan duty,
`[0, 1]`) -- for checking whether `trajectory_controller.max_torque` or a
fan's duty actually saturates during a large re-orientation, instead of
guessing from gains alone.

The `move_to` goal (blocking `send_goal`) and all four recorder
subscriptions run on the SAME node/single-threaded default executor -- same
pattern as `move_to_full_trace.py` -- so `spin_until_future_complete` inside
`send_goal` keeps servicing the subscriptions while it blocks on the action
result. No separate process/manual timing is needed to line them up.

Ground truth for "was this setpoint tick a re-plan tick" is derived from
message ORDER, not from any new instrumentation added to `guidance_executor.py`:
`ReplanningTrajectoryTracker.sample()` (`guidance/trajectory_tracking/
replanning_trajectory_tracker.py`) increments its internal tick counter and
attempts a re-plan every `replan_every_n_ticks`-th call, and `_run_trajectory`
calls `tracker.sample()` (hence publishes) exactly once per while-loop
iteration -- so every Nth published setpoint since the translation leg
started IS a re-plan tick, deterministically, regardless of any timing jitter
in when ticks actually occur. This script fetches the live `guidance.rate`/
`guidance.replan_rate_hz` params from `/guidance_node` to compute N, falling
back to the hardcoded default (5) if the service call fails.

Usage (single leg, replanning mode already set via `ros2 param set`):
    python3 test/manual/log_replanning_attitude_trace.py near_dock \
        --out-dir /tmp/replan_trace --tag run1

Usage (N repeated round trips, to investigate the 6-4 節 run-to-run
variance, one command instead of manually re-timing each leg):
    python3 test/manual/log_replanning_attitude_trace.py near_dock \
        --return-to nav_entry --trials 5 --out-dir /tmp/replan_trace \
        --set-mode replanning
"""
import argparse
import csv
import os
import sys

import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from trajectory_msgs.msg import MultiDOFJointTrajectory

from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
SETPOINT_TOPIC = "/gnc/trajectory_setpoint"
GUIDANCE_NODE = "guidance_node"
DEFAULT_REPLAN_EVERY_N_TICKS = 5  # matches replanning_trajectory_tracker.py's default

_NO_WAIT = rclpy.duration.Duration(seconds=0)

# Matches the actual /tf publisher's QoS (best-effort) -- a RELIABLE reader
# would be QoS-incompatible with a BEST_EFFORT writer and silently receive
# nothing at all. Same profile as common/ros/tf_client.py's DEFAULT_QOS.
TF_QOS = QoSProfile(
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)

# Matches the actual /gnc/trajectory_setpoint publisher's QoS (reliable) --
# subscribing RELIABLE here (rather than the package's usual best-effort
# subscriber default) is deliberate: this script's whole point is to not
# drop a single setpoint sample.
SETPOINT_QOS = QoSProfile(
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)

# /ctl/wrench and /ctl/duty are both published RELIABLE too (confirmed live
# via `ros2 topic info --verbose`), same reasoning as SETPOINT_QOS.
WRENCH_TOPIC = "/ctl/wrench"
DUTY_TOPIC = "/ctl/duty"
WRENCH_DUTY_QOS = QoSProfile(
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


class TfRawRecorder:
    """Records every distinct ``iss_body <- body`` TF sample, event-driven.

    Owns its own ``tf2_ros.Buffer`` (deliberately not ``TfClient``'s) fed by
    its own ``/tf`` subscription, so ``set_transform`` always runs before
    this recorder's own lookup for the same incoming message -- sharing
    ``TfClient``'s buffer via a second, independent subscription to the same
    topic would race against ``TfClient``'s own callback for that guarantee.
    Deliberately does not subscribe ``/tf_static`` (see test/manual/README.md
    "Common pattern" 1. and ``docs/archive/achieved/tf_race_investigation.md``).
    """

    def __init__(self, node):
        self._buffer = tf2_ros.Buffer()
        self._last_stamp = None
        self.rows = []  # (t_sim, px, py, pz, qx, qy, qz, qw)
        node.create_subscription(TFMessage, "/tf", self._on_tf, TF_QOS)

    def _on_tf(self, msg):
        for transform in msg.transforms:
            self._buffer.set_transform(transform, "log_replanning_attitude_trace")
        try:
            t = self._buffer.lookup_transform(
                REFERENCE_FRAME, TARGET_FRAME, rclpy.time.Time(), timeout=_NO_WAIT
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if stamp == self._last_stamp:
            return  # not a new sample, e.g. an unrelated edge in this /tf message
        self._last_stamp = stamp
        tr, q = t.transform.translation, t.transform.rotation
        self.rows.append((stamp, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))


class SetpointRawRecorder:
    """Records every ``/gnc/trajectory_setpoint`` message, in arrival order."""

    def __init__(self, node):
        self.rows = []  # (t_sim, px,py,pz, vx,vy,vz, ax,ay,az, qx,qy,qz,qw)
        node.create_subscription(
            MultiDOFJointTrajectory, SETPOINT_TOPIC, self._on_msg, SETPOINT_QOS
        )

    def _on_msg(self, msg):
        if not msg.points:
            return
        point = msg.points[0]
        tr = point.transforms[0].translation
        q = point.transforms[0].rotation
        v = point.velocities[0].linear
        a = point.accelerations[0].linear
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rows.append((
            stamp, tr.x, tr.y, tr.z, v.x, v.y, v.z, a.x, a.y, a.z, q.x, q.y, q.z, q.w
        ))


class WrenchRawRecorder:
    """Records every ``/ctl/wrench`` message (commanded force/torque, the
    input to the fan allocator) -- for checking whether the attitude
    controller's ``max_torque`` clamp is actually being hit during a large
    re-orientation (``docs/archive/achieved/2026-08-25_guidance_realtime_replanning_sim_verification.md``
    9-5 節's open question)."""

    def __init__(self, node):
        self.rows = []  # (t_sim, fx, fy, fz, tx, ty, tz)
        node.create_subscription(WrenchStamped, WRENCH_TOPIC, self._on_msg, WRENCH_DUTY_QOS)

    def _on_msg(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        f, t = msg.wrench.force, msg.wrench.torque
        self.rows.append((stamp, f.x, f.y, f.z, t.x, t.y, t.z))


class DutyRawRecorder:
    """Records every ``/ctl/duty`` message (per-fan duty, ``[0, 1]``) --
    ``Float64MultiArray`` has no header/stamp, so this timestamps on the
    owning node's own (sim-time) clock at reception instead."""

    def __init__(self, node):
        self._node = node
        self.rows = []  # (t_sim, duty0, duty1, ..., dutyN)
        node.create_subscription(Float64MultiArray, DUTY_TOPIC, self._on_msg, WRENCH_DUTY_QOS)

    def _on_msg(self, msg):
        t = self._node.get_clock().now().nanoseconds * 1e-9
        self.rows.append((t,) + tuple(msg.data))


def fetch_replan_every_n_ticks(node):
    """Best-effort fetch of the live ``guidance.rate``/``guidance.replan_rate_hz``
    params from ``/guidance_node``, to compute the real re-plan tick interval
    instead of assuming the hardcoded default still applies."""
    client = node.create_client(GetParameters, f"/{GUIDANCE_NODE}/get_parameters")
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().warn(
            "[log_replanning_attitude_trace] /guidance_node/get_parameters "
            "unavailable, falling back to default replan_every_n_ticks=%d"
            % DEFAULT_REPLAN_EVERY_N_TICKS
        )
        return DEFAULT_REPLAN_EVERY_N_TICKS
    req = GetParameters.Request(names=["guidance.rate", "guidance.replan_rate_hz"])
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    result = future.result()
    if result is None or len(result.values) != 2:
        node.get_logger().warn(
            "[log_replanning_attitude_trace] could not fetch guidance.rate/"
            "replan_rate_hz, falling back to default replan_every_n_ticks=%d"
            % DEFAULT_REPLAN_EVERY_N_TICKS
        )
        return DEFAULT_REPLAN_EVERY_N_TICKS
    rate = result.values[0].double_value
    replan_rate_hz = result.values[1].double_value
    if rate <= 0.0 or replan_rate_hz <= 0.0:
        return DEFAULT_REPLAN_EVERY_N_TICKS
    n_ticks = round(rate / replan_rate_hz)
    node.get_logger().info(
        "[log_replanning_attitude_trace] live guidance.rate=%.1f "
        "replan_rate_hz=%.1f -> replan_every_n_ticks=%d"
        % (rate, replan_rate_hz, n_ticks)
    )
    return n_ticks


def set_tracking_mode(node, mode):
    client = node.create_client(SetParameters, f"/{GUIDANCE_NODE}/set_parameters")
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(
            "[log_replanning_attitude_trace] /guidance_node/set_parameters "
            "unavailable -- is guidance_node running?"
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
        node.get_logger().error(
            "[log_replanning_attitude_trace] failed to set "
            "guidance.trajectory_tracking_mode=%s" % mode
        )
    return ok


def move_leg(node, client, location_name, leg_label):
    """Resolve ``location_name`` via TF and send it as a blocking move_to
    goal. Returns True on SUCCESS-shaped result, False otherwise -- the
    caller decides whether to keep going."""
    resolved = client.resolve_location(location_name)
    if resolved is None:
        node.get_logger().error(
            "[log_replanning_attitude_trace] could not resolve '%s' via TF"
            % location_name
        )
        return False
    pos, quat = resolved
    node.get_logger().info(
        "[log_replanning_attitude_trace] leg=%s -> %s pos=%s, sending goal..."
        % (leg_label, location_name, pos)
    )
    result = client.send_goal(pos, quat, timeout_sec=90.0)
    if result is None:
        node.get_logger().error(
            "[log_replanning_attitude_trace] leg=%s did not complete" % leg_label
        )
        return False
    node.get_logger().info(
        "[log_replanning_attitude_trace] leg=%s finished result.type=%d"
        % (leg_label, result.type)
    )
    return True


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote %d rows -> %s" % (len(rows), path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("location_name", help="first leg's destination (TF frame name)")
    ap.add_argument("--return-to", default=None,
                     help="destination for the return leg; required if --trials > 1")
    ap.add_argument("--trials", type=int, default=1,
                     help="number of round trips (location_name -> return-to) to run "
                          "back-to-back in one process (default: 1, i.e. a single "
                          "one-way leg to location_name if --return-to is omitted)")
    ap.add_argument("--set-mode", choices=("static", "replanning"), default=None,
                     help="set guidance.trajectory_tracking_mode before the first "
                          "leg (default: leave whatever is already set)")
    ap.add_argument("--out-dir", default="/tmp/replan_trace")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    if args.trials > 1 and not args.return_to:
        ap.error("--return-to is required when --trials > 1")

    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    node = Node("log_replanning_attitude_trace")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    tf_rec = TfRawRecorder(node)
    sp_rec = SetpointRawRecorder(node)
    wrench_rec = WrenchRawRecorder(node)
    duty_rec = DutyRawRecorder(node)
    move_client = MoveToClient(node)

    replan_every_n_ticks = fetch_replan_every_n_ticks(node)

    if args.set_mode is not None:
        if not set_tracking_mode(node, args.set_mode):
            node.destroy_node()
            rclpy.shutdown()
            return 1

    # Bookmarks (start/end index) into each ever-growing recorder list, per
    # leg -- so nothing recorded has to be split/filtered live (all raw data
    # stays, legs are just index ranges).
    recorders = {"sp": sp_rec, "tf": tf_rec, "wrench": wrench_rec, "duty": duty_rec}
    legs = []
    ok = True
    for trial in range(args.trials):
        for dest, suffix in (
            (args.location_name, "fwd"),
            (args.return_to, "back") if args.return_to else (None, None),
        ):
            if dest is None:
                continue
            leg_label = "trial%d_%s" % (trial, suffix)
            starts = {name: len(rec.rows) for name, rec in recorders.items()}
            ok = move_leg(node, move_client, dest, leg_label)
            ends = {name: len(rec.rows) for name, rec in recorders.items()}
            legs.append((leg_label, starts, ends))
            if not ok:
                break
        if not ok:
            break

    sp_header = ["t_sim", "px", "py", "pz", "vx", "vy", "vz",
                 "ax", "ay", "az", "qx", "qy", "qz", "qw", "leg",
                 "leg_local_idx", "likely_replan_tick"]
    sp_rows = []
    for leg_label, starts, ends in legs:
        for local_idx, row in enumerate(sp_rec.rows[starts["sp"]:ends["sp"]]):
            likely_replan = (local_idx % replan_every_n_ticks) == 0
            sp_rows.append(tuple(row) + (leg_label, local_idx, likely_replan))

    tf_header = ["t_sim", "px", "py", "pz", "qx", "qy", "qz", "qw", "leg", "leg_local_idx"]
    tf_rows = []
    for leg_label, starts, ends in legs:
        for local_idx, row in enumerate(tf_rec.rows[starts["tf"]:ends["tf"]]):
            tf_rows.append(tuple(row) + (leg_label, local_idx))

    wrench_header = ["t_sim", "fx", "fy", "fz", "tx", "ty", "tz", "leg", "leg_local_idx"]
    wrench_rows = []
    for leg_label, starts, ends in legs:
        for local_idx, row in enumerate(wrench_rec.rows[starts["wrench"]:ends["wrench"]]):
            wrench_rows.append(tuple(row) + (leg_label, local_idx))

    duty_rows = []
    for leg_label, starts, ends in legs:
        for local_idx, row in enumerate(duty_rec.rows[starts["duty"]:ends["duty"]]):
            duty_rows.append(tuple(row) + (leg_label, local_idx))
    n_fans = (len(duty_rows[0]) - 3) if duty_rows else 8  # row = (t_sim, *duties, leg, leg_local_idx)
    duty_header = (["t_sim"] + ["duty%d" % i for i in range(n_fans)] + ["leg", "leg_local_idx"])

    write_csv(os.path.join(args.out_dir, "%s_setpoint.csv" % args.tag), sp_header, sp_rows)
    write_csv(os.path.join(args.out_dir, "%s_tf.csv" % args.tag), tf_header, tf_rows)
    write_csv(os.path.join(args.out_dir, "%s_wrench.csv" % args.tag), wrench_header, wrench_rows)
    write_csv(os.path.join(args.out_dir, "%s_duty.csv" % args.tag), duty_header, duty_rows)
    print("legs: %s" % [l[0] for l in legs])
    print("total setpoint rows=%d, total tf rows=%d, wrench rows=%d, duty rows=%d"
          % (len(sp_rec.rows), len(tf_rec.rows), len(wrench_rec.rows), len(duty_rec.rows)))

    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
