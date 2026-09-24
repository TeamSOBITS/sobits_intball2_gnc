"""Obstacle avoidance stage 5 (docs/2026-09-24_obstacle_avoidance_jem_map_check.md):
EGO-Planner v2 checkCollisionCallback in ReplanningMincoV3Tracker, with obstacles that
appear suddenly ahead on the straight 6 m route, stopping along the real StoppingProfile.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import minco_native_py
from sobits_intball2_gnc.common.utils.stopping_profile import StoppingProfile
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import (
    ReplanningMincoV3Tracker,
)

_here = os.path.dirname(__file__)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage4 = _load("stage4", "experiment_obstacle_stage4_tracker.py")
cancel = _load("cancel", "experiment_cancel_stopping_profile.py")
jem = _load("jem", "experiment_obstacle_jem_route.py")

# (name, JEM walls, box offset ahead of the setpoint when it appears [m], half extents)
# The box appears once the setpoint has covered 2.5 m, centered on the straight route.
SCENARIOS = [
    ("open: appears 1.0m ahead", False, 1.0, stage4.PERSON),
    ("open: appears 0.6m ahead", False, 0.6, stage4.PERSON),
    ("JEM: appears 1.0m ahead", True, 1.0, jem.PERSON_ACROSS_Y),
    ("JEM: blocking wall 1.0m ahead", True, 1.0, np.array([1.5, 0.15, 1.5])),
]


def stop_profile_factory():
    node = cancel._YamlNode(cancel.PARAMS)
    allocator = ThrustAllocator.from_node(node)
    mass = float(node.get_parameter("trajectory_controller.mass").value)
    inertia = float(node.get_parameter("trajectory_controller.inertia").value)
    eta = float(node.get_parameter("guidance.wrench_envelope_safety_margin").value)
    max_axis_force = float(node.get_parameter("hover_control.max_force").value)
    return lambda p, v, q, w: StoppingProfile(p, v, q, w, allocator, mass, inertia, eta,
                                              max_axis_force=max_axis_force)


def run(use_jem, ahead, half, stop_fn, t_max=60.0, progress=None, appear_after=2.5):
    grid = minco_native_py.OccupancyGrid(stage4.GRID_RES, stage4.GRID_INFLATION)
    wall_tree = None
    if use_jem:
        wall_points = minco_native_py.load_octomap_points(jem.MAP)[1]
        grid.add_points(wall_points)
        wall_tree = cKDTree(np.asarray(wall_points).reshape(-1, 3))
        start, goal = jem.location("inspection_entry_1"), jem.location("nav_entry")
        q0 = jem.facing_quat(goal - start)
    else:
        start, goal, q0 = np.zeros(3), stage4.GOAL, stage4.Q0
    route_dir = (goal - start) / np.linalg.norm(goal - start)
    state = {"p": start.copy(), "s": 0.0}
    tr = ReplanningMincoV3Tracker(
        start, goal, lambda: (state["p"], list(q0), state["s"]),
        lambda s: True, q0, stage4.TS, stage4.MA, via_half_width=0.0,
        wrench_safety_margin=stage4.MARGIN, attitude_resample_spacing_m=stage4.SPACING,
        planning_horizon_m=stage4.HORIZON, face_travel=True, forward_axis=stage4.FWD,
        local_max_vel=stage4.CRUISE, local_piece_length_m=stage4.PIECE_LENGTH,
        obstacle_grid=grid, obstacle_clearance_soft=stage4.CLEARANCE_SOFT,
        stop_profile_fn=stop_fn)
    box, t, clr, stop_t, v_appear, resumed = None, 0.0, np.inf, None, None, False
    wall_clr = np.inf
    solves, last_solve = [], None
    while t <= tr.total_duration and t < t_max:
        t += stage4.DT
        p, v, _a, _q = tr.sample(t)
        state["p"], state["s"] = p, t
        if tr.last_replan_solve_seconds is not None and tr.last_replan_solve_seconds is not last_solve:
            last_solve = tr.last_replan_solve_seconds
            solves.append(last_solve)
        if box is None and (p - start) @ route_dir >= appear_after:
            depth = abs(half @ np.abs(route_dir))
            box = start + route_dir * ((p - start) @ route_dir + ahead + depth)
            if use_jem:
                box[2] = jem.FLOOR_Z + half[2] if half[2] < 1.0 else box[2]
            grid.add_box(list(box), list(half))
            v_appear = float(np.linalg.norm(v))
        if wall_tree is not None:
            wall_clr = min(wall_clr, wall_tree.query(p)[0] - 0.025 - stage4.ROBOT_RADIUS)
        if box is not None:
            clr = min(clr, stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS)
        if tr.emergency_stops and stop_t is None:
            stop_t = t
        if stop_t is not None and tr._stop_profile is None:
            resumed = True
        if progress and int(t / 5.0) != int((t - stage4.DT) / 5.0):
            print("  [%s] t=%5.1f p=%s stopped=%s rest_failures=%d" % (
                progress, t, np.round(p, 2), tr._stop_profile is not None, tr._rest_replan_failures),
                flush=True)
    return dict(t=t, clr=clr, wall_clr=wall_clr, solves=solves, stops=tr.emergency_stops, stop_t=stop_t, v=v_appear, resumed=resumed,
                end=np.linalg.norm(state["p"] - goal), stopped_now=tr._stop_profile is not None)


def main():
    for a in sys.argv[1:]:
        if a.startswith("--inflation="):
            stage4.GRID_INFLATION = float(a.split("=", 1)[1])
    stop_fn = stop_profile_factory()
    print("grid inflation %.2f m" % stage4.GRID_INFLATION)
    print("cruise %.2f; box appears after 2.5 m; clearance = robot surface to box [m]" % stage4.CRUISE)
    picked = [int(a) for a in sys.argv[1:] if a.isdigit()] or range(len(SCENARIOS))
    for k in picked:
        name, use_jem, ahead, half = SCENARIOS[k]
        ahead = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--ahead=")), ahead)
        name = "%s (ahead %.1fm)" % (name, ahead)
        appear_after = next((float(a.split("=", 1)[1]) for a in sys.argv[1:]
                             if a.startswith("--appear-after=")), 2.5)
        r = run(use_jem, ahead, half, stop_fn, progress=name, appear_after=appear_after)
        print("%-30s v_at_appear=%.3f emergency_stops=%d stop_at=%s resumed=%s still_stopped=%s "
              "clearance=%.3f wall_clearance=%.3f t=%.1f goal_err=%.3f solve_n=%d solve_mean=%.3f solve_max=%.3f" % (
                  name, r["v"], r["stops"], None if r["stop_t"] is None else round(r["stop_t"], 1),
                  r["resumed"], r["stopped_now"], r["clr"], r["wall_clr"], r["t"], r["end"],
                  len(r["solves"]), np.mean(r["solves"]) if r["solves"] else np.nan, max(r["solves"], default=np.nan)), flush=True)
    if "--regression" not in sys.argv:
        return
    print("regression (known obstacles, with stop_profile_fn):")
    for name, boxes in stage4.SCENARIOS:
        grid_boxes = boxes
        r = stage4_run_with_stop(grid_boxes, stop_fn)
        print("  %-24s emergency_stops=%d clearance=%.3f t=%.1f end=%s" % (
            name, r[0], r[1], r[2], np.round(r[3], 3)), flush=True)


def stage4_run_with_stop(boxes, stop_fn):
    orig = ReplanningMincoV3Tracker.__init__
    holder = {}

    def init(self, *a, **kw):
        holder["tr"] = self
        orig(self, *a, stop_profile_fn=stop_fn, **kw)

    stage4.ReplanningMincoV3Tracker.__init__ = init
    try:
        r = stage4.run(boxes)
    finally:
        stage4.ReplanningMincoV3Tracker.__init__ = orig
    return holder["tr"].emergency_stops, r["clr"], r["t"], r["end"]


if __name__ == "__main__":
    main()
