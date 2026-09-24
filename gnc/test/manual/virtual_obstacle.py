#!/usr/bin/env python3
"""Add, remove or clear a virtual obstacle box in guidance_node's obstacle map
(``/guidance/virtual_obstacles``, see guidance/ros/marker_array_subscriber.py).

The box is axis-aligned in iss_body. Default size is a standing person across
JEM's +-y travel: 0.5 (x) x 0.3 (y) x 1.7 (z) m.

Usage:
    python3 test/manual/virtual_obstacle.py add --ahead 1.0 [--floor-z 4.05] [--id 0]
    python3 test/manual/virtual_obstacle.py add --at 10.95 -6.6 4.9 [--size 0.5 0.3 1.7]
    python3 test/manual/virtual_obstacle.py delete --id 0
    python3 test/manual/virtual_obstacle.py clear

``--ahead D`` centers the box D m (plus half its depth) ahead of the vehicle
along the body forward axis (the travel direction under face travel), at the
vehicle's height unless ``--floor-z`` stands it on the floor.
Needs guidance_node running (obstacle avoidance is enabled per goal with
``ros2 param set /guidance_node guidance.minco_obstacle_avoidance true``).
"""
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.ros.marker_array_subscriber import (
    LATCHED_QOS,
    VIRTUAL_OBSTACLES_TOPIC,
)
from sobits_intball2_gnc.guidance.utils.guidance_executor import DEFAULT_CAMERA_FORWARD_AXIS

REFERENCE_FRAME = "iss_body"
NAMESPACE = "manual"
MATCH_WAIT_SIM_S = 5.0
FLUSH_SIM_S = 1.0
# /clock and /tf wake spin_once early, so the waits are bounded by sim time; this only
# stops an endless loop when /clock never arrives.
MAX_SPINS_WITHOUT_CLOCK = 200


def box_marker(action, marker_id, center=None, size=None):
    marker = Marker()
    marker.header.frame_id = REFERENCE_FRAME
    marker.ns = NAMESPACE
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = action
    marker.pose.orientation.w = 1.0
    if center is not None:
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, center)
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, size)
    return marker


def box_ahead(tf_client, ahead, size, floor_z):
    pos, quat, _stamp = tf_client.get_pose()
    forward = np.asarray(quat_rotate(quat, DEFAULT_CAMERA_FORWARD_AXIS["main"]))
    forward /= np.linalg.norm(forward)
    half_depth = 0.5 * float(np.abs(np.asarray(size)) @ np.abs(forward))
    center = np.asarray(pos) + forward * (ahead + half_depth)
    if floor_z is not None:
        center[2] = floor_z + size[2] / 2.0
    return center, pos


def spin_for_sim(node, seconds, done=lambda: False):
    """Spin until ``done()`` or ``seconds`` of sim time pass; returns ``done()``."""
    clock = node.get_clock()
    start = None
    spins_without_clock = 0
    while not done():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = clock.now()
        if now.nanoseconds == 0:
            spins_without_clock += 1
            if spins_without_clock >= MAX_SPINS_WITHOUT_CLOCK:
                break
            continue
        if start is None:
            start = now
        if (now - start).nanoseconds * 1e-9 >= seconds:
            break
    return done()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["add", "delete", "clear"])
    parser.add_argument("--id", type=int, default=0)
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--ahead", type=float, help="distance ahead of the vehicle [m]")
    where.add_argument("--at", type=float, nargs=3, metavar=("X", "Y", "Z"), help="center in iss_body [m]")
    parser.add_argument("--size", type=float, nargs=3, default=[0.5, 0.3, 1.7], metavar=("SX", "SY", "SZ"))
    parser.add_argument("--floor-z", type=float, default=None, help="stand the box on this floor height [m]")
    args = parser.parse_args()
    if args.command == "add" and args.ahead is None and args.at is None:
        parser.error("add needs --ahead or --at")

    rclpy.init()
    node = Node("virtual_obstacle_once")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    pub = node.create_publisher(MarkerArray, VIRTUAL_OBSTACLES_TOPIC, LATCHED_QOS)
    try:
        msg = MarkerArray()
        if args.command == "clear":
            msg.markers.append(box_marker(Marker.DELETEALL, 0))
        elif args.command == "delete":
            msg.markers.append(box_marker(Marker.DELETE, args.id))
        else:
            if args.at is not None:
                center = np.asarray(args.at)
            else:
                tf_client = TfClient(node, REFERENCE_FRAME, "body")
                if not tf_client.wait_for_frame(timeout_sec=5.0):
                    print("TF unavailable: %s <- body" % REFERENCE_FRAME)
                    return
                center, pos = box_ahead(tf_client, args.ahead, args.size, args.floor_z)
                print("vehicle at %s" % np.round(pos, 3))
            msg.markers.append(box_marker(Marker.ADD, args.id, center, args.size))
            print("box id=%d center=%s size=%s" % (args.id, np.round(center, 3), args.size))

        if not spin_for_sim(node, MATCH_WAIT_SIM_S, lambda: pub.get_subscription_count() > 0):
            print("no subscriber on %s (is guidance_node running?)" % VIRTUAL_OBSTACLES_TOPIC)
            return
        pub.publish(msg)
        spin_for_sim(node, FLUSH_SIM_S)
        print("sent %s to %s" % (args.command, VIRTUAL_OBSTACLES_TOPIC))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
