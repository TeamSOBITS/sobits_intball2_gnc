#!/usr/bin/env python3
"""move_to a named location while injecting one real physical disturbance
mid-flight, to verify docs/main_plan.md's outstanding "擬似衝突からの復帰"
item: does trajectory_tracking_mode="replanning" actually recover from a
disturbance, not just from an undisturbed/monotonic approach (the only kind
exercised by test_execute_replanning_mode_reaches_target's fake TF)?

The disturbance is a genuine Gazebo physics impulse via the ROS1
/gazebo/apply_body_wrench service (body_name "ib2::base", confirmed reachable
and effective 2026-08-25: a 0.3N/0.3s pulse produced a real +4.4mm TF
displacement that then decayed back under hover control) -- NOT a TF/pose
override, which would fake the very signal replanning is supposed to react
to. Called from this ROS2 script via a clean ROS1 subshell (docs/archive
[[ros1_bridge_access]] pattern), since the service isn't bridged into ROS2.

Reuses test/manual/move_to_full_trace.py's pattern (TfClient + MoveToClient
on the same node/executor, so spin_until_future_complete inside send_goal
still services the logging/trigger timer while blocking on the goal result).

Trigger modes (exactly one disturbance, fired once the condition is first
met after --arm-delay seconds have elapsed):
    --trigger-elapsed-sec SEC   fire SEC seconds after the goal was sent
        (use for a mid-flight disturbance).
    --trigger-distance-m M      fire once remaining distance to the target
        first drops below M (use to land the disturbance near arrival, e.g.
        just inside distance_fallback_m, to probe the one-way-latch gap
        documented in test_latched_fallback_does_not_recover_from_post_
        latch_disturbance).

The default disturbance (5N/0.3s = 1.5N*s impulse, dv~0.33m/s at this
vehicle's ~4.5kg mass) is deliberately scaled well above the 0.3N/0.3s
calibration pulse (4.4mm displacement) to a collision-like kick meant to
displace the vehicle past distance_fallback_m (0.3m) -- otherwise the test
never exercises a real re-plan, just noise-scale jitter.

Usage:
    ros2 param set /guidance_node guidance.trajectory_tracking_mode replanning
    python3 test/manual/move_to_with_disturbance.py nav_entry \\
        --trigger-elapsed-sec 3.0 --force 0.0 5.0 0.0 --force-duration 0.3 \\
        --out-csv /tmp/disturbance_nav_entry.csv
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.guidance.ros.move_to_client import MoveToClient

OUT_CSV_DEFAULT = "/tmp/move_to_with_disturbance.csv"
BODY_NAME = "ib2::base"
ROS1_MASTER_URI = "http://172.17.0.1:11311"


def rpy(q):
    x, y, z, w = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sinp))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def apply_body_wrench(force_xyz, duration_sec, logger):
    """Fire one Gazebo physics impulse at ib2::base via a clean ROS1 subshell
    (this rclpy process is ROS2-only; the service isn't bridged). Returns
    (success, status_message)."""
    yaml_req = (
        "{body_name: '%s', reference_frame: '', "
        "reference_point: {x: 0.0, y: 0.0, z: 0.0}, "
        "wrench: {force: {x: %.6f, y: %.6f, z: %.6f}, "
        "torque: {x: 0.0, y: 0.0, z: 0.0}}, "
        "start_time: {secs: 0, nsecs: 0}, "
        "duration: {secs: 0, nsecs: %d}}"
    ) % (BODY_NAME, force_xyz[0], force_xyz[1], force_xyz[2],
         int(duration_sec * 1e9))
    bash_cmd = (
        "source /opt/ros/noetic/setup.bash && "
        "export ROS_MASTER_URI=%s && export ROS_IP=127.0.0.1 && "
        "rosservice call /gazebo/apply_body_wrench \"%s\""
        % (ROS1_MASTER_URI, yaml_req.replace('"', '\\"'))
    )
    cmd = ["env", "-i", "HOME=%s" % os.environ.get("HOME", "/root"),
           "PATH=/usr/bin:/bin", "bash", "-c", bash_cmd]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        logger.error("[move_to_with_disturbance] apply_body_wrench call timed out")
        return False, "timeout"
    ok = "success: True" in result.stdout
    logger.info(
        "[move_to_with_disturbance] apply_body_wrench force=%s duration=%.3fs "
        "-> %s" % (list(force_xyz), duration_sec, result.stdout.strip())
    )
    if not ok:
        logger.error(
            "[move_to_with_disturbance] apply_body_wrench FAILED: stdout=%r stderr=%r"
            % (result.stdout, result.stderr)
        )
    return ok, result.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("location_name")
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    trigger = ap.add_mutually_exclusive_group(required=True)
    trigger.add_argument("--trigger-elapsed-sec", type=float,
                          help="fire the disturbance this many seconds after "
                               "the goal is sent (mid-flight scenario)")
    trigger.add_argument("--trigger-distance-m", type=float,
                          help="fire the disturbance once remaining distance "
                               "to the target first drops below this many "
                               "meters (near-arrival scenario)")
    ap.add_argument("--force", type=float, nargs=3, default=[5.0, 0.0, 0.0],
                     metavar=("FX", "FY", "FZ"),
                     help="disturbance force in N, iss_body-ish axes as seen "
                          "by Gazebo world frame. Default (5.0, 0, 0) with "
                          "the default 0.3s duration is impulse=1.5N*s, "
                          "dv=1.5/4.5kg=0.33m/s -- roughly 15x the 0.3N/0.3s "
                          "calibration pulse (2026-08-25) that only produced "
                          "a 4.4mm displacement, deliberately scaled up to a "
                          "collision-like kick that should displace the "
                          "vehicle well past distance_fallback_m (0.3m) "
                          "before control/replanning reacts. Pick a direction "
                          "roughly perpendicular to the direction of travel "
                          "for a genuine off-path knock rather than a push "
                          "straight along the line to the target "
                          "(default: %(default)s)")
    ap.add_argument("--force-duration", type=float, default=0.3,
                     help="disturbance duration in seconds (default: %(default)s)")
    ap.add_argument("--timeout-sec", type=float, default=90.0,
                     help="move_to goal timeout (default: %(default)s)")
    ap.add_argument("--post-goal-log-sec", type=float, default=30.0,
                     help="keep logging TF pose for this many additional sim "
                          "seconds after send_goal() returns -- execute()'s "
                          "own SUCCESS only means attitude converged "
                          "(align_at_arrival's _align_to checks attitude "
                          "only), while PoseCorrector's checkpoint hold "
                          "keeps correcting position in the background "
                          "afterward; without this window the true "
                          "convergence curve/time is invisible to this "
                          "script's CSV (default: %(default)s)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("move_to_with_disturbance")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    tf_client = TfClient(node, "iss_body", "body")
    if not tf_client.wait_for_frame(timeout_sec=10.0):
        print("TF unavailable, aborting")
        return 1

    client = MoveToClient(node)
    resolved = client.resolve_location(args.location_name)
    if resolved is None:
        print(f"could not resolve '{args.location_name}' via TF")
        return 1
    target_pos, target_quat = resolved
    target_pos_np = np.asarray(target_pos, dtype=float)
    print(f"resolved {args.location_name} -> pos={target_pos} quat={target_quat}")

    rows = []
    state = {"fired": False, "t0": None}

    def tick():
        pose = tf_client.get_pose()
        if pose is None:
            return
        pos, quat, stamp = pose
        if state["t0"] is None:
            state["t0"] = stamp
        elapsed = stamp - state["t0"]
        remaining = float(np.linalg.norm(target_pos_np - np.asarray(pos, dtype=float)))
        r, p, y = rpy(quat)

        fired_this_tick = False
        if not state["fired"]:
            should_fire = (
                (args.trigger_elapsed_sec is not None and elapsed >= args.trigger_elapsed_sec)
                or (args.trigger_distance_m is not None and remaining <= args.trigger_distance_m)
            )
            if should_fire:
                state["fired"] = True
                fired_this_tick = True
                node.get_logger().info(
                    "[move_to_with_disturbance] TRIGGER at elapsed=%.2fs "
                    "remaining=%.3fm pos=%s" % (elapsed, remaining, pos)
                )
                apply_body_wrench(args.force, args.force_duration, node.get_logger())

        rows.append((stamp, elapsed, pos[0], pos[1], pos[2], r, p, y,
                     quat[0], quat[1], quat[2], quat[3], remaining,
                     int(fired_this_tick)))

    timer = node.create_timer(1.0 / args.rate_hz, tick)

    def on_feedback(fb):
        pass  # position/attitude already captured by the tick() timer

    result = client.send_goal(target_pos, target_quat, feedback_cb=on_feedback,
                               timeout_sec=args.timeout_sec)

    if args.post_goal_log_sec > 0.0:
        node.get_logger().info(
            "[move_to_with_disturbance] goal returned (result=%s) -- "
            "continuing to log for %.1fs to capture any post-arrival "
            "checkpoint-hold convergence" % (result, args.post_goal_log_sec)
        )
        post_goal_start = node.get_clock().now()
        deadline_sec = args.post_goal_log_sec
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed_since_goal = (
                (node.get_clock().now() - post_goal_start).nanoseconds * 1e-9
            )
            if elapsed_since_goal >= deadline_sec:
                break

    timer.cancel()

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tf_stamp", "elapsed", "x", "y", "z", "roll", "pitch", "yaw",
                     "qx", "qy", "qz", "qw", "remaining_dist_m", "disturbance_fired"])
        w.writerows(rows)

    final_pos = np.array(rows[-1][2:5]) if rows else None
    final_error = (
        float(np.linalg.norm(target_pos_np - final_pos)) if final_pos is not None else None
    )

    if result is None:
        print(f"goal did not complete, logged {len(rows)} ticks -> {args.out_csv}")
    else:
        print(f"finished result.type={result.type}, logged {len(rows)} ticks -> {args.out_csv}")
    print(f"disturbance fired: {state['fired']}, final position error: {final_error} m")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
