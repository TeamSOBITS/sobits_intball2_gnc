#!/usr/bin/env python3
"""Spawn, list or delete visual-only obstacles in Gazebo for the virtual camera.

Boxes are named ``vbox_<sx>x<sy>x<sz>_<id>`` (full size [m]) so
virtual_camera_node can read their size from ``/gazebo/model_states``, which
carries no geometry; mesh models (intball2_programs ``SIM_MODELS``, e.g.
``float_blue``) are named ``<model>_<id>``. Nothing gets a collision, so the
vehicle passes through them.

The ISS model slowly rotates in Gazebo's world (~1.5 cm/min drift 12 m from its
origin), so ``add`` keeps running and re-pins the obstacle to its iss_body
pose at 10 Hz (sim time) until Ctrl-C or the obstacle is deleted (``clear``),
like ``spawn_model``; run it in the background. ``--no-hold`` spawns once and exits (the obstacle then drifts).

Usage:
    python3 test/manual/spawn_obstacle.py add --ahead 1.5 [--model box --size 0.5 0.3 1.7] [--camera main] [--id 0]
    python3 test/manual/spawn_obstacle.py add --at 10.9 -6.6 4.9 --model float_blue
    python3 test/manual/spawn_obstacle.py list
    python3 test/manual/spawn_obstacle.py delete --name vbox_0.500x0.300x1.700_0
    python3 test/manual/spawn_obstacle.py clear
"""
import argparse
import re
import sys

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose
from rclpy.node import Node

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from virtual_obstacle import spin_for_sim

CAMERA_AXES = {"stereo": [0.0, 1.0, 0.0], "main": [1.0, 0.0, 0.0]}
BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""
SPAWNED = re.compile(r"^vbox_|^(float2?_[a-z]+|ctb_\d+)_")
WAIT_SIM_S = 10.0
HOLD_PERIOD_SIM_S = 0.1


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(r):
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class ObstacleSpawner:
    def __init__(self, node):
        self._node = node
        self._states = None
        node.create_subscription(ModelStates, "/gazebo/model_states", self._on_states, 1)
        self._spawn = node.create_client(SpawnModel, "/gazebo/spawn_sdf_model")
        self._delete = node.create_client(DeleteModel, "/gazebo/delete_model")
        self._set_state = node.create_publisher(ModelState, "/gazebo/set_model_state", 10)

    def _on_states(self, msg):
        self._states = msg

    def model_states(self):
        self._states = None
        if not spin_for_sim(self._node, WAIT_SIM_S, lambda: self._states is not None):
            raise RuntimeError("no /gazebo/model_states")
        return self._states

    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"service {client.srv_name} unavailable")
        future = client.call_async(request)
        if not spin_for_sim(self._node, WAIT_SIM_S, future.done):
            raise RuntimeError(f"{client.srv_name} timed out")
        return future.result()

    @staticmethod
    def world_pose(states, center_iss):
        iss = states.pose[states.name.index("iss")]
        world_r_iss = quat_to_matrix([iss.orientation.x, iss.orientation.y, iss.orientation.z, iss.orientation.w])
        world_p = world_r_iss @ np.asarray(center_iss, dtype=float) + [iss.position.x, iss.position.y, iss.position.z]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, world_p)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, matrix_to_quat(world_r_iss))
        return pose

    def spawn(self, name, xml, center_iss):
        request = SpawnModel.Request(model_name=name, model_xml=xml,
                                     initial_pose=self.world_pose(self.model_states(), center_iss), reference_frame="world")
        return self._call(self._spawn, request)

    def hold(self, name, center_iss):
        """Re-pin ``name`` to ``center_iss`` every HOLD_PERIOD_SIM_S until interrupted or the model is deleted."""
        seen = False
        while rclpy.ok():
            if self._states is not None:
                if name in self._states.name:
                    seen = True
                    self._set_state.publish(ModelState(model_name=name, pose=self.world_pose(self._states, center_iss),
                                                       reference_frame="world"))
                elif seen:
                    print(f"{name} was deleted; stop holding", flush=True)
                    return
            spin_for_sim(self._node, HOLD_PERIOD_SIM_S)

    def delete(self, name):
        return self._call(self._delete, DeleteModel.Request(model_name=name))


def mesh_xml(model, name):
    from intball2_programs.spawn import sim_models
    return sim_models.build_sim_model_xml(name, sim_models.SIM_MODELS[model], collision_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    where = add.add_mutually_exclusive_group(required=True)
    where.add_argument("--ahead", type=float, help="distance [m] from the vehicle to the obstacle center along --camera")
    where.add_argument("--at", type=float, nargs=3, metavar=("X", "Y", "Z"), help="center in iss_body")
    add.add_argument("--model", default="box", help="box, or a SIM_MODELS mesh model (float_blue, ctb_1023, ...)")
    add.add_argument("--size", type=float, nargs=3, default=[0.5, 0.3, 1.7], help="box full size [m]")
    add.add_argument("--camera", choices=sorted(CAMERA_AXES), default="main")
    add.add_argument("--id", default="0")
    add.add_argument("--no-hold", action="store_true", help="spawn and exit without pinning to the ISS")
    sub.add_parser("list")
    delete = sub.add_parser("delete")
    delete.add_argument("--name", required=True)
    sub.add_parser("clear")
    args = parser.parse_args()

    rclpy.init()
    node = Node("spawn_obstacle")
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
    spawner = ObstacleSpawner(node)
    try:
        if args.command == "add":
            if args.ahead is not None:
                tf_client = TfClient(node, "iss_body", "body")
                if not tf_client.wait_for_frame(timeout_sec=5.0):
                    raise RuntimeError("TF unavailable: iss_body <- body")
                pos, quat, _ = tf_client.get_pose()
                center = np.asarray(pos) + np.asarray(quat_rotate(quat, CAMERA_AXES[args.camera])) * args.ahead
            else:
                center = np.asarray(args.at)
            if args.model == "box":
                name = "vbox_{:.3f}x{:.3f}x{:.3f}_{}".format(*args.size, args.id)
                xml = BOX_SDF.format(name=name, sx=args.size[0], sy=args.size[1], sz=args.size[2])
            else:
                name = f"{args.model}_{args.id}"
                xml = mesh_xml(args.model, name)
            result = spawner.spawn(name, xml, center)
            print(f"{'spawned' if result.success else 'FAILED'} {name} at iss_body {np.round(center, 3)}: {result.status_message}",
                  flush=True)
            if not result.success:
                return 1
            if not args.no_hold:
                print("holding it to the ISS (Ctrl-C to stop; the obstacle stays)", flush=True)
                spawner.hold(name, center)
            return 0
        names = [n for n in spawner.model_states().name if SPAWNED.match(n)]
        if args.command == "list":
            print("\n".join(names) or "(none)")
            return 0
        targets = [args.name] if args.command == "delete" else names
        ok = True
        for name in targets:
            result = spawner.delete(name)
            ok &= result.success
            print(f"{'deleted' if result.success else 'FAILED'} {name}: {result.status_message}")
        return 0 if ok else 1
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
