#!/usr/bin/env python3
"""Save stereo_image_proc depth frames with the vehicle pose, for ground-truth comparison.

Records ``/stereo/disparity`` frames while the vehicle holds still, converts them
to depth, and saves them with the body pose (TF, sampled before and after) to an
``.npz`` for ``compare_stereo_depth_to_octomap.py``. Aborts if the vehicle moved
during capture, since the pose is not interpolated to each frame's stamp.

Needs ``ros2 launch intball2_programs stereo_pointcloud.launch.py`` running.

Usage:
    python3 test/manual/capture_stereo_depth.py --out /tmp/stereo_depth.npz [--frames 5] [--timeout 60]
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from virtual_obstacle import spin_for_sim

DISPARITY_TOPIC = "/stereo/disparity"
CAMERA_INFO_TOPIC = "/camera_left/camera_info_fixed"
MAX_MOVE_M = 0.005
MAX_TURN_DEG = 0.2


def disparity_to_depth(msg):
    d = np.frombuffer(bytes(msg.image.data), np.float32).reshape(msg.image.height, msg.image.width)
    valid = np.isfinite(d) & (d >= msg.min_disparity) & (d <= msg.max_disparity) & (d != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid, msg.f * msg.t / d, np.nan).astype(np.float32)


def turn_deg(q0, q1):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0, help="sim seconds")
    args = parser.parse_args()

    rclpy.init()
    node = Node("capture_stereo_depth")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    tf_client = TfClient(node, "iss_body", "body")
    frames, stamps, info = [], [], {}
    meta = {}

    def on_disparity(msg):
        if len(frames) >= args.frames:
            return
        frames.append(disparity_to_depth(msg))
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        meta.update(f=msg.f, t=msg.t, frame_id=msg.header.frame_id)

    def on_info(msg):
        info.setdefault("k", np.array(msg.k, dtype=float).reshape(3, 3))

    try:
        if not tf_client.wait_for_frame(timeout_sec=5.0):
            print("TF unavailable: iss_body <- body")
            return
        pos0, quat0, stamp0 = tf_client.get_pose()
        node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, on_info, qos_profile_sensor_data)
        node.create_subscription(DisparityImage, DISPARITY_TOPIC, on_disparity, 10)
        spin_for_sim(node, args.timeout, lambda: len(frames) >= args.frames and "k" in info)
        pos1, quat1, stamp1 = tf_client.get_pose()

        if not frames or "k" not in info:
            print(f"got {len(frames)} disparity frame(s), camera_info={'k' in info}; is stereo_pointcloud.launch.py running?")
            return
        moved = float(np.linalg.norm(np.subtract(pos1, pos0)))
        turned = turn_deg(np.asarray(quat0), np.asarray(quat1))
        if moved > MAX_MOVE_M or turned > MAX_TURN_DEG:
            print(f"vehicle moved during capture ({moved * 1e3:.1f}mm, {turned:.2f}deg); not saved")
            return

        np.savez_compressed(
            args.out, depth=np.stack(frames), stamps=np.array(stamps), k=info["k"],
            body_pos=np.asarray(pos0, dtype=float), body_quat=np.asarray(quat0, dtype=float),
            pose_stamps=np.array([stamp0, stamp1]), f=meta["f"], t=meta["t"], frame_id=meta["frame_id"])
        print(f"saved {len(frames)} frame(s) {frames[0].shape} to {args.out}; "
              f"frames {stamps[0]:.2f}-{stamps[-1]:.2f}s, pose {stamp0:.2f}/{stamp1:.2f}s, "
              f"drift {moved * 1e3:.1f}mm {turned:.2f}deg")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
