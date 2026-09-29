"""JEM obstacle scenarios seen only through the front depth camera, against the oracle
that puts the boxes straight into the grid
(docs/archive/2026-09-28_virtual_camera_depth_mapping_plan.md, offline step).

Usage: python3 experiment_obstacle_depth_camera.py [scenario ids...] [--mode=oracle|depth|both] [--t-max=200]
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

_here = os.path.dirname(__file__)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage4 = _load("stage4", "experiment_obstacle_stage4_tracker.py")
jem = _load("jem", "experiment_obstacle_jem_route.py")
stage5 = _load("stage5", "experiment_obstacle_stage5_emergency_stop.py")

# gnc_params/virtual_camera.yaml and ib2.urdf (cameraF_joint): main camera, 800 px / downsample 4.
CAMERA_OFFSET_BODY = np.array([0.14, 0.0, 0.0])
LINK_R_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
WIDTH = HEIGHT = 200
FX = 0.5 * 800 / np.tan(0.5 * 1.396263) / 4
CX = CY = (0.5 * 801 + 0.5) / 4 - 0.5
MIN_RANGE, MAX_RANGE = 0.25, 3.0
CAMERA_PERIOD = 0.1
MARGIN = 0.3
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# (name, box center or None for "appears ahead", half extents, appear ahead [m])
SCENARIOS = [
    ("walls only", None, None, None),
    ("person centered", [10.95, -6.6, jem.FLOOR_Z + 0.85], jem.PERSON_ACROSS_Y, None),
    ("person x+0.15", [11.10, -6.6, jem.FLOOR_Z + 0.85], jem.PERSON_ACROSS_Y, None),
    ("appears 1.0m ahead", None, jem.PERSON_ACROSS_Y, 1.0),
    ("blocking wall appears 1.0m ahead", None, np.array([1.5, 0.15, 1.5]), 1.0),
]


def camera_pose(p, q):
    r_body = Rotation.from_quat(q).as_matrix()
    return p + r_body @ CAMERA_OFFSET_BODY, r_body @ LINK_R_OPTICAL


def run(mode, scenario, wall_points, wall_tree, renderer, stop_fn, t_max=200.0):
    name, center, half, ahead = scenario
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(stage4.GRID_RES, stage4.GRID_INFLATION)
    grid.add_points(wall_points)
    pts = np.asarray(wall_points).reshape(-1, 3)
    if mode == "depth":
        grid.enable_depth_layer(list(pts.min(0) - MARGIN), list(pts.max(0) + MARGIN),
                                min_range=MIN_RANGE, max_range=MAX_RANGE)
    start, goal = jem.location("inspection_entry_1"), jem.location("nav_entry")
    route_dir = (goal - start) / np.linalg.norm(goal - start)
    q0 = jem.facing_quat(goal - start)
    box = None if center is None else np.asarray(center, float)
    if box is not None and mode == "oracle":
        grid.add_box(list(box), list(half))
    state = {"p": start.copy(), "q": list(q0), "s": 0.0}
    tr = ReplanMincoTracker(
        start, goal, lambda: (state["p"], state["q"], state["s"]), lambda s: True, q0,
        stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
        attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON,
        face_travel=True, forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE,
        local_piece_length_m=stage4.PIECE_LENGTH, obstacle_grid=grid,
        obstacle_clearance_soft=stage4.CLEARANCE_SOFT, stop_profile_fn=stop_fn)
    t, next_frame = 0.0, 0.0
    clr, wall_clr, detect_dist, stop_t, frames = np.inf, np.inf, None, None, 0
    replans = 0
    while t <= tr.total_duration and t < t_max:
        t += stage4.DT
        p, v, _a, q = tr.sample(t)
        state["p"], state["q"], state["s"] = p, list(q), t
        replans += int(tr.last_replan_occurred)
        if box is None and ahead is not None and (p - start) @ route_dir >= 2.5:
            depth_along = abs(half @ np.abs(route_dir))
            box = start + route_dir * ((p - start) @ route_dir + ahead + depth_along)
            box[2] = jem.FLOOR_Z + half[2] if half[2] < 1.0 else box[2]
            if mode == "oracle":
                grid.add_box(list(box), list(half))
        if mode == "depth" and t >= next_frame:
            next_frame += CAMERA_PERIOD
            origin, r_opt = camera_pose(p, q)
            boxes = [] if box is None else [(list(box), list(half), IDENTITY)]
            depth = renderer.render(list(origin), r_opt.ravel().tolist(), FX, FX, CX, CY, WIDTH, HEIGHT,
                                    boxes=boxes, max_range=MAX_RANGE)
            depth = np.where(np.isnan(depth), np.inf, depth).astype(np.float32)
            grid.integrate_depth(depth, FX, FX, CX, CY, r_opt.ravel().tolist(), list(origin))
            frames += 1
            if box is not None and detect_dist is None:
                cells = grid.depth_occupied_cells()
                if len(cells) and np.any(np.all(np.abs(cells - box) <= half + stage4.GRID_RES, axis=1)):
                    detect_dist = stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS
        if box is not None:
            clr = min(clr, stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS)
        wall_clr = min(wall_clr, wall_tree.query(p)[0] - 0.025 - stage4.ROBOT_RADIUS)
        if tr.emergency_stops and stop_t is None:
            stop_t = t
        # Once halted, only the (slow, failing) rest replans remain; nothing more to measure.
        if stop_t is not None and np.linalg.norm(v) < 0.01:
            break
    return dict(t=t, clr=clr, wall_clr=wall_clr, detect=detect_dist, stops=tr.emergency_stops,
                stop_t=stop_t, replans=replans, frames=frames, end=np.linalg.norm(state["p"] - goal),
                still_stopped=tr._stop_profile is not None)


def main():
    mode_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mode=")), "both")
    modes = ["oracle", "depth"] if mode_arg == "both" else [mode_arg]
    t_max = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--t-max=")), 200.0)
    picked = [int(a) for a in sys.argv[1:] if a.isdigit()] or range(len(SCENARIOS))
    wall_points = sobits_intball2_gnc_cpp.load_octomap_points(jem.MAP)[1]
    wall_tree = cKDTree(np.asarray(wall_points).reshape(-1, 3))
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    renderer.set_static(wall_points, 0.05)
    stop_fn = stage5.stop_profile_factory()
    print("inspection_entry_1 -> nav_entry, cruise %.2f, camera %dx%d @ %.0f Hz, range %.2f-%.1f m; "
          "clearances = robot surface [m]" % (stage4.CRUISE, WIDTH, HEIGHT, 1 / CAMERA_PERIOD, MIN_RANGE, MAX_RANGE))
    for k in picked:
        for mode in modes:
            r = run(mode, SCENARIOS[k], wall_points, wall_tree, renderer, stop_fn, t_max)
            print("%-34s %-6s t=%6.1f goal_err=%.3f still_stopped=%s stops=%d stop_at=%s clr=%7.3f wall=%6.3f "
                  "detected_at=%s replans=%d frames=%d" % (
                      SCENARIOS[k][0], mode, r["t"], r["end"], r["still_stopped"], r["stops"],
                      None if r["stop_t"] is None else round(r["stop_t"], 1), r["clr"], r["wall_clr"],
                      None if r["detect"] is None else round(r["detect"], 2), r["replans"], r["frames"]),
                  flush=True)


if __name__ == "__main__":
    main()
