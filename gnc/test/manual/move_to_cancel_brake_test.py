#!/usr/bin/env python3
"""Cancel a move_to mid-route and measure the post-cancel stop
(docs/archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md).

Sends a goal to ``location_name``, cancels it once the vehicle has covered
``--cancel-fraction`` of the straight start->target distance (or, with
``--cancel-near FRAME``, once it is within ``--cancel-radius`` of that TF
frame, e.g. a via waypoint to cancel mid-curve), then keeps
recording TF (10 Hz) and ``/ctl/duty`` for ``--post-sec`` (sim time) and
reports how far past the cancel point it went, whether it came back, when it
settled, and fan-duty saturation. Also reports the cancel result latency and
the stop point guidance published on ``/gnc/checkpoints``.

Usage:
    python3 test/manual/move_to_cancel_brake_test.py inspection_entry_2
    python3 test/manual/move_to_cancel_brake_test.py inspection_entry_2 --cancel-fraction 0.4 --post-sec 60
    python3 test/manual/move_to_cancel_brake_test.py inspection_entry_3 --cancel-near inspection_entry_1 --cancel-radius 0.15
"""
import argparse
import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from ib2_msgs.action import CtlCommand
from ib2_msgs.msg import CtlStatusType
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import MultiDOFJointTrajectory

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle
from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

ACTION_NAME = "/gnc/move_to"
REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RECORD_PERIOD = 0.1
SAT_DUTY = 0.95
SETTLE_SPEED = 0.005
SETTLE_POS = 0.02
RELIABLE = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)


def _now(node):
    return node.get_clock().now().nanoseconds * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("location_name")
    ap.add_argument("--cancel-fraction", type=float, default=0.5)
    ap.add_argument("--cancel-near", default=None,
                    help="TF frame; cancel once within --cancel-radius of it")
    ap.add_argument("--cancel-radius", type=float, default=0.15)
    ap.add_argument("--post-sec", type=float, default=60.0)
    ap.add_argument("--move-timeout-sec", type=float, default=120.0,
                    help="sim-time limit before the cancel point is reached")
    ap.add_argument("--out-dir", default="/tmp/move_to_cancel_brake_test")
    args = ap.parse_args()

    rclpy.init()
    node = Node("move_to_cancel_brake_test",
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    try:
        _run(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run(node, args):
    log = node.get_logger()
    target = MoveToClient(node, reference_frame=REFERENCE_FRAME).resolve_location(
        args.location_name)
    if target is None:
        log.error("location '%s' not found in TF" % args.location_name)
        return
    cancel_near = None
    if args.cancel_near:
        near = MoveToClient(node, reference_frame=REFERENCE_FRAME).resolve_location(
            args.cancel_near)
        if near is None:
            log.error("location '%s' not found in TF" % args.cancel_near)
            return
        cancel_near = np.asarray(near[0])
    tf = TfClient(node, REFERENCE_FRAME, TARGET_FRAME)
    if not tf.wait_for_frame(5.0):
        log.error("TF %s <- %s unavailable" % (REFERENCE_FRAME, TARGET_FRAME))
        return

    state = {"v_des_max": 0.0, "duty": [], "checkpoints": []}

    def on_setpoint(msg):
        if msg.points and msg.points[0].velocities:
            lin = msg.points[0].velocities[0].linear
            state["v_des_max"] = max(state["v_des_max"], float(np.linalg.norm([lin.x, lin.y, lin.z])))

    def on_duty(msg):
        state["duty"].append((_now(node), list(msg.data)))

    def on_checkpoint(msg):
        if msg.poses:
            p = msg.poses[0].position
            state["checkpoints"].append((_now(node), [p.x, p.y, p.z]))

    node.create_subscription(MultiDOFJointTrajectory, "/gnc/trajectory_setpoint",
                             on_setpoint, RELIABLE)
    node.create_subscription(Float64MultiArray, "/ctl/duty", on_duty, RELIABLE)
    node.create_subscription(PoseArray, "/gnc/checkpoints", on_checkpoint, RELIABLE)

    client = ActionClient(node, CtlCommand, ACTION_NAME)
    if not client.wait_for_server(timeout_sec=5.0):
        log.error("action server %s unavailable" % ACTION_NAME)
        return

    p_start = np.asarray(tf.get_pose()[0])
    p_target = np.asarray(target[0])
    route = p_target - p_start
    route_len = float(np.linalg.norm(route))
    route_dir = route / route_len
    log.info("start=%s target=%s dist=%.2fm, cancel %s"
             % (np.round(p_start, 3), np.round(p_target, 3), route_len,
                "within %.2fm of %s" % (args.cancel_radius, args.cancel_near)
                if cancel_near is not None else "at %.0f%%" % (100 * args.cancel_fraction)))

    goal = CtlCommand.Goal()
    goal.target.header.frame_id = REFERENCE_FRAME
    goal.target.header.stamp = node.get_clock().now().to_msg()
    (goal.target.pose.position.x, goal.target.pose.position.y,
     goal.target.pose.position.z) = target[0]
    (goal.target.pose.orientation.x, goal.target.pose.orientation.y,
     goal.target.pose.orientation.z, goal.target.pose.orientation.w) = target[1]
    goal.type.type = CtlStatusType.MOVE_TO_ABSOLUTE_TARGET
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        log.error("goal rejected")
        return
    result_future = handle.get_result_async()

    samples = []
    t_goal = _now(node)
    t_next_record = t_goal
    t_cancel = t_result = None
    while True:
        rclpy.spin_once(node, timeout_sec=0.02)
        t = _now(node)
        if t >= t_next_record:
            pose = tf.get_pose()
            if pose is not None:
                samples.append((t, np.asarray(pose[0]), np.asarray(pose[1])))
            t_next_record = t + RECORD_PERIOD
        if t_result is None and result_future.done():
            t_result = t
        if t_cancel is None:
            if t_result is not None:
                log.error("goal finished before the cancel point")
                return
            if cancel_near is not None:
                reached = bool(samples) and np.linalg.norm(samples[-1][1] - cancel_near) <= args.cancel_radius
            else:
                reached = bool(samples) and (
                    np.dot(samples[-1][1] - p_start, route_dir) >= args.cancel_fraction * route_len)
            if reached:
                handle.cancel_goal_async()
                t_cancel = t
                log.info("cancel sent at t=%.1fs" % (t - t_goal))
            elif t - t_goal > args.move_timeout_sec:
                handle.cancel_goal_async()
                log.error("cancel point not reached within %.0fs" % args.move_timeout_sec)
                return
        elif t - t_cancel >= args.post_sec:
            break

    _report(log, args, samples, state, t_goal, t_cancel, t_result, route_dir)


def _report(log, args, samples, state, t_goal, t_cancel, t_result, route_dir):
    ts = np.array([s[0] for s in samples])
    ps = np.array([s[1] for s in samples])
    qs = [s[2] for s in samples]
    vs = np.vstack([np.zeros(3), np.diff(ps, axis=0) / np.diff(ts)[:, None]])
    speeds = np.linalg.norm(vs, axis=1)

    i_c = int(np.searchsorted(ts, t_cancel))
    v_cancel = vs[i_c]
    d = v_cancel / np.linalg.norm(v_cancel) if np.linalg.norm(v_cancel) > 1e-3 else route_dir
    post_t = ts[i_c:] - t_cancel
    post_p = ps[i_c:]
    along = (post_p - ps[i_c]) @ d
    p_final = post_p[-1]
    final_along = float((p_final - ps[i_c]) @ d)

    settled = (speeds[i_c:] < SETTLE_SPEED) & (np.linalg.norm(post_p - p_final, axis=1) < SETTLE_POS)
    unsettled = np.flatnonzero(~settled)
    settle_t = None if unsettled.size and unsettled[-1] + 1 >= len(post_t) else (
        post_t[0] if not unsettled.size else post_t[unsettled[-1] + 1])
    stop_candidates = np.flatnonzero(speeds[i_c:] < SETTLE_SPEED)
    first_stop_t = post_t[stop_candidates[0]] if stop_candidates.size else None

    duty_post = [dd for t, dd in state["duty"] if t >= t_cancel]
    sat = np.mean([max(dd) >= SAT_DUTY for dd in duty_post]) if duty_post else float("nan")
    att_dev = max(geodesic_angle(q, qs[-1]) for q in qs[i_c:])
    cps = [(t - t_cancel, p) for t, p in state["checkpoints"] if t >= t_cancel]

    print("\n=== move_to cancel/brake report: %s ===" % args.location_name)
    print("cancel sent        : %.1fs after goal, |v| at cancel = %.3f m/s (peak v_des %.3f)"
          % (t_cancel - t_goal, np.linalg.norm(v_cancel), state["v_des_max"]))
    print("result received    : %s" % ("%.2fs after cancel" % (t_result - t_cancel)
                                        if t_result is not None else "NOT received"))
    print("max excursion      : %.3f m past the cancel point (along motion)" % along.max())
    print("final position     : %.3f m past the cancel point" % final_along)
    print("came back          : %.3f m" % (along.max() - final_along))
    print("first |v|<%.0fmm/s  : %s" % (SETTLE_SPEED * 1e3, "-" if first_stop_t is None
                                         else "%.1fs after cancel" % first_stop_t))
    print("settled            : %s" % ("not within %.0fs" % args.post_sec if settle_t is None
                                        else "%.1fs after cancel" % settle_t))
    print("duty>=%.2f         : %.1f%% of post-cancel /ctl/duty samples" % (SAT_DUTY, 100 * sat))
    print("attitude change    : %.1f deg max vs final" % np.degrees(att_dev))
    if cps:
        t_cp, p_cp = cps[0]
        print("stop-point hold    : published %.1fs after cancel, %.3f m past the cancel point, "
              "final pos %.3f m from it" % (t_cp, float((np.asarray(p_cp) - ps[i_c]) @ d),
                                          np.linalg.norm(p_final - np.asarray(p_cp))))
    else:
        print("stop-point hold    : NOT published")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tf_%s.csv" % args.location_name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_since_cancel", "x", "y", "z", "qx", "qy", "qz", "qw", "speed"])
        for t, p, q, s in zip(ts, ps, qs, speeds):
            w.writerow([round(t - t_cancel, 3), *np.round(p, 4), *np.round(q, 5), round(s, 4)])
    print("tf csv             : %s" % path)


if __name__ == "__main__":
    main()
