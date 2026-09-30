#!/usr/bin/env python3
"""Static-person depth test of the production global-corridor tracker path.

Uses the shipped avoidance settings and a rendered ``float_blue`` mesh.  The
vehicle follows the actual ``ReplanMincoTracker`` reference; controller error
and ROS transport latency are intentionally outside this offline test.
"""
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import sobits_intball2_gnc_cpp as cpp
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.trajectory_tracking.corridor_session import CorridorSession
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
P = yaml.safe_load(open(os.path.join(ROOT, "config", "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
G, VC = P["guidance"], yaml.safe_load(open(os.path.join(ROOT, "config", "virtual_camera.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]["virtual_camera"]
LOC = yaml.safe_load(open(os.path.join(ROOT, "maps", "iss_location.yaml"), encoding="utf-8"))["location_pose"]
MAP = os.path.join(ROOT, "maps", G["obstacle_map_file"])
MESH = "/home/space_project/colcon_ws/install/intball2_programs/share/intball2_programs/media/meshes/human_obstacles/float_blue.dae"
FWD = np.array([1., 0., 0.])
LINK_R_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def pose(name):
    value = LOC[name]
    return (np.array([value["translation"][axis] for axis in ("x", "y", "z")], float),
            np.array([value["rotation"][axis] for axis in ("x", "y", "z", "w")], float))


def main():
    start, qstart = pose("inspection_entry_1")
    goal, _qgoal = pose("nav_entry")
    qface = compute_q_des(goal - start, qstart, 1e-9, FWD)
    depth = {key: G["depth"][key] for key in (
        "p_hit", "p_miss", "p_min", "p_max", "p_occ", "min_range", "max_range", "skip_pixel")}
    obstacle_map = ObstacleMap(G["obstacle_grid_resolution"], G["obstacle_grid_inflation"], MAP, depth)
    resolution, static = cpp.load_octomap_points(MAP)
    renderer = cpp.DepthRenderer(); renderer.set_static(static, resolution)
    shape = renderer.add_shape(cpp.load_mesh_surface_voxels(MESH, VC["mesh_resolution"]), VC["mesh_resolution"])
    person = np.array([10.95, -6.60, 4.90])
    camera = VC["cameras"]["main"]
    width, height = camera["width"] // VC["downsample"], camera["height"] // VC["downsample"]
    focal = .5 * camera["width"] / np.tan(.5 * camera["horizontal_fov"]) / VC["downsample"]
    center = (.5 * (camera["width"] + 1) + .5) / VC["downsample"] - .5
    mesh_r = Rotation.from_euler("x", np.pi).as_matrix().ravel().tolist()
    state = {"p": start.copy(), "q": qface.copy(), "t": 0.0}

    def integrate_depth():
        body = Rotation.from_quat(state["q"]).as_matrix()
        origin = state["p"] + body @ np.array([.14, 0., 0.])
        optical = body @ LINK_R_OPTICAL
        image = renderer.render(origin.tolist(), optical.ravel().tolist(), focal, focal, center, center,
                                width, height, instances=[(shape, mesh_r, person.tolist())],
                                max_range=VC["max_range"], threads=VC["threads"])
        obstacle_map.integrate_depth(np.where(np.isnan(image), np.inf, image).astype(np.float32),
                                     focal, focal, center, center, optical, origin, state["t"])

    # Match the configured 10 Hz depth integration before accepting the goal.
    for _ in range(10):
        integrate_depth()
    session = CorridorSession(obstacle_map, goal, qface, FWD,
                              G["minco_planning_horizon_m"], G["wrench_envelope_safety_margin"],
                              G["minco_obstacle_clearance_soft"])
    tracker = ReplanMincoTracker(
        start, goal, lambda: (state["p"], state["q"], state["t"]), lambda _stamp: True, qface,
        G["target_speed"], min(P["trajectory_controller"]["max_force"]) / P["trajectory_controller"]["mass"],
        local_replan_period=G["minco_local_replan_period"], planning_horizon_m=G["minco_planning_horizon_m"],
        face_travel=True, forward_axis=FWD, local_max_vel=G["minco_local_max_vel"],
        async_replan=False, local_piece_length_m=G["minco_local_piece_length_m"],
        obstacle_grid=obstacle_map.grid, obstacle_clearance_soft=G["minco_obstacle_clearance_soft"],
        corridor_session=session)
    updates, minimum = 0, float("inf")
    while state["t"] < 180.0 and np.linalg.norm(goal - state["p"]) > .10:
        state["t"] += .1
        integrate_depth()
        state["p"], _v, _a, state["q"] = tracker.sample(state["t"])
        updates += int(tracker.last_replan_occurred)
        minimum = min(minimum, float(np.linalg.norm(state["p"] - person)))
    print("static_person,arrived=%s,arrival_s=%.3f,goal_m=%.3f,updates=%d,min_person_center_m=%.3f" % (
        np.linalg.norm(goal - state["p"]) <= .10, state["t"], np.linalg.norm(goal - state["p"]),
        updates, minimum))


if __name__ == "__main__":
    main()
