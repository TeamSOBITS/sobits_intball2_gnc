#!/usr/bin/env python3
"""Measure self-position-estimate stability: per-tick pose jump and stale gap.

Written for docs/tf_bridge_stability_investigation.md's stability criteria:
GNC cares whether the *output* of self-position estimation is well-behaved
(no discontinuous jumps, no long freezes), not which transport delivered it.
So the pose source is a small swappable adapter (currently TfClient) behind a
`get_pose() -> (pos, quat, stamp) | None` interface -- the jump/gap evaluation
below is written against that interface only, and stays valid if the source
is later replaced (e.g. a direct gz topic reader, bypassing the ROS1<->ROS2
bridge investigated in that doc) instead of being embedded in PoseCorrector /
TrajectoryController, which would tie it to today's bridge+TF path.

Two metrics, sampled every tick at RATE_HZ:

- Position/attitude jump (m / deg) between the current pose and the last
  *distinct* stamp seen -- the primary criterion. A real jump here means
  something GNC actually consumes went wrong, regardless of root cause.
- Stale gap (sec): how long the pose source has been repeating the same
  stamp -- a secondary/diagnostic signal for *why* a jump might be coming
  (bridge stall), measured on the sim clock so it is comparable to the pose
  source's own timestamps.

Manual verification script (test/manual/): requires a running sim + gnc
launch, not collected by pytest. See test/manual/README.md.
"""
import csv
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient

REFERENCE_FRAME = "iss_body"
TARGET_FRAME = "body"
RATE_HZ = 50.0
OUT_CSV = "/tmp/pose_stability.csv"

# Tentative -- not yet calibrated against a confirmed "safe" baseline. Purpose
# here is to flag candidate anomalies in the log for later threshold tuning,
# not to make a pass/fail call.
POS_JUMP_WARN_M = 0.05
ATT_JUMP_WARN_DEG = 10.0
GAP_WARN_SEC = 0.2


def _quat_angle_deg(q1, q2) -> float:
    """Angle (deg) between two unit quaternions, sign/double-cover agnostic."""
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, dot)
    return float(np.degrees(2.0 * np.arccos(dot)))


class PoseStabilityEvaluator:
    """Turns a stream of (pos, quat, stamp) samples into jump/gap metrics.

    Deliberately knows nothing about TF, the bridge, or rclpy -- feed it
    whatever a pose source's get_pose() returns.
    """

    def __init__(self):
        self._last_pos = None
        self._last_quat = None
        self._last_stamp = None
        self._last_change_wall_sec = None
        self.max_pos_jump_m = 0.0
        self.max_att_jump_deg = 0.0
        self.max_gap_sec = 0.0
        self.anomaly_count = 0

    def update(self, pose, wall_now_sec: float):
        """Process one sample. Returns a dict of this tick's metrics.

        `wall_now_sec` is the caller's clock (sim time in this script) used
        only to time stale gaps -- never compared against `stamp`, which may
        be on a different clock (see TfClient.get_pose docstring).
        """
        is_new_sample = False
        pos_jump_m = 0.0
        att_jump_deg = 0.0
        gap_sec = 0.0

        if pose is None:
            if self._last_change_wall_sec is not None:
                gap_sec = wall_now_sec - self._last_change_wall_sec
                self.max_gap_sec = max(self.max_gap_sec, gap_sec)
            return {
                "is_new_sample": is_new_sample,
                "pos_jump_m": pos_jump_m,
                "att_jump_deg": att_jump_deg,
                "gap_sec": gap_sec,
                "anomaly": False,
            }

        pos, quat, stamp = pose
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)

        if self._last_stamp is not None and stamp == self._last_stamp:
            # No new data since last tick: still the same sample, growing gap.
            gap_sec = wall_now_sec - self._last_change_wall_sec
            self.max_gap_sec = max(self.max_gap_sec, gap_sec)
        else:
            is_new_sample = True
            if self._last_pos is not None:
                pos_jump_m = float(np.linalg.norm(pos - self._last_pos))
                att_jump_deg = _quat_angle_deg(quat, self._last_quat)
                self.max_pos_jump_m = max(self.max_pos_jump_m, pos_jump_m)
                self.max_att_jump_deg = max(self.max_att_jump_deg, att_jump_deg)
            self._last_pos = pos
            self._last_quat = quat
            self._last_stamp = stamp
            self._last_change_wall_sec = wall_now_sec

        anomaly = (
            pos_jump_m > POS_JUMP_WARN_M
            or att_jump_deg > ATT_JUMP_WARN_DEG
            or gap_sec > GAP_WARN_SEC
        )
        if anomaly:
            self.anomaly_count += 1

        return {
            "is_new_sample": is_new_sample,
            "pos_jump_m": pos_jump_m,
            "att_jump_deg": att_jump_deg,
            "gap_sec": gap_sec,
            "anomaly": anomaly,
        }


def main():
    rclpy.init(args=sys.argv)
    node = Node(
        "measure_pose_stability",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
    )
    # Swappable pose source: only get_pose()/ready/wait_for_frame are used
    # below. A future gz-topic-direct source just needs the same interface.
    pose_source = TfClient(node, reference_frame=REFERENCE_FRAME, target_frame=TARGET_FRAME)

    if not pose_source.wait_for_frame(timeout_sec=8.0):
        node.get_logger().error("[measure_pose_stability] could not get a pose")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info(
        f"[measure_pose_stability] logging to {OUT_CSV} at {RATE_HZ} Hz. Ctrl-C to stop."
    )

    evaluator = PoseStabilityEvaluator()
    rows = []
    t_start = node.get_clock().now()
    tick_count = 0

    def on_tick():
        nonlocal tick_count
        tick_count += 1
        elapsed = (node.get_clock().now() - t_start).nanoseconds * 1e-9
        pose = pose_source.get_pose()
        metrics = evaluator.update(pose, elapsed)
        rows.append((
            elapsed,
            metrics["is_new_sample"],
            metrics["pos_jump_m"],
            metrics["att_jump_deg"],
            metrics["gap_sec"],
            metrics["anomaly"],
        ))
        if metrics["anomaly"]:
            node.get_logger().warn(
                "[measure_pose_stability] t=%.2fs anomaly: pos_jump=%.1fmm "
                "att_jump=%.1fdeg gap=%.3fs"
                % (elapsed, metrics["pos_jump_m"] * 1000, metrics["att_jump_deg"], metrics["gap_sec"])
            )
        if tick_count % int(RATE_HZ * 2) == 0:  # every ~2s
            node.get_logger().info(
                "[measure_pose_stability] t=%.1fs max_pos_jump=%.1fmm "
                "max_att_jump=%.1fdeg max_gap=%.3fs anomalies=%d"
                % (
                    elapsed,
                    evaluator.max_pos_jump_m * 1000,
                    evaluator.max_att_jump_deg,
                    evaluator.max_gap_sec,
                    evaluator.anomaly_count,
                )
            )

    node.create_timer(1.0 / RATE_HZ, on_tick)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "is_new_sample", "pos_jump_m", "att_jump_deg",
                "gap_sec", "anomaly",
            ])
            writer.writerows(rows)
        node.get_logger().info(
            "[measure_pose_stability] %d samples written to %s. "
            "max_pos_jump=%.1fmm max_att_jump=%.1fdeg max_gap=%.3fs anomalies=%d"
            % (
                len(rows), OUT_CSV,
                evaluator.max_pos_jump_m * 1000,
                evaluator.max_att_jump_deg,
                evaluator.max_gap_sec,
                evaluator.anomaly_count,
            )
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
