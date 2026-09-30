#!/usr/bin/env python3
"""Test-only FIRI half-space -> Int-Ball MINCO -> initial-local gate, nine cases."""
import os
import subprocess
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXE = os.path.join(WORKSPACE, "build", "sobits_intball2_gnc_cpp", "experiment_firi_jem_corridor")
Q0 = np.array([0., 0., 0., 1.])
FWD = np.array([1., 0., 0.])
C = sobits_intball2_gnc_cpp.CONSTRAINT_POINTS_PER_PIECE


def corridor_pairs(output, segments):
    planes = [[] for _ in range(segments)]
    for line in output.splitlines():
        if line.startswith("PLANE "):
            item = dict(value.split("=", 1) for value in line.split()[1:])
            planes[int(item["segment"])].append(np.array(
                [float(item["nx"]), float(item["ny"]), float(item["nz"]), float(item["b"])]) )
    if any(not part for part in planes):
        raise RuntimeError("missing FIRI planes")
    flat = []
    for point_id in range(segments * C + 1):
        segment = min(point_id // C, segments - 1)
        for plane in planes[segment]:
            normal, bias = plane[:3], plane[3]
            norm = np.linalg.norm(normal)
            base_on_plane = -bias * normal / (norm * norm)
            # FIRI inside: n dot q + b <= 0. obstacle_pairs require
            # (q-base) dot direction >= 0, hence direction=-normal/norm.
            flat.extend([point_id, *base_on_plane, *(-normal / norm)])
    return flat


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,firi_minco_global_free,initial_local_solve,n_pairs,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
            route = base.shortcut_path(base.AStarPlanner(
                base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS, connectivity=6).plan(start, goal), grid, base.BOUNDS)
            encoded = ";".join(",".join(f"{value:.6f}" for value in point) for point in route)
            run = subprocess.run([EXE, base.MAP, str(x), str(y), encoded], capture_output=True,
                                 text=True, timeout=30, check=True)
            try:
                pairs = corridor_pairs(run.stdout, len(route) - 1)
                global_trajectory = MincoTrajectory(
                    route, Q0, face_travel=True, forward_axis=FWD, body_frame_wrench=True,
                    via_half_width=0., target_speed=base.TARGET_SPEED, max_accel=base.MAX_ACCEL,
                    obstacle_pairs=pairs, obstacle_touch_goal=True)
                global_ok = curve_is_free(global_trajectory, grid, base.BOUNDS)["ok"]
                state = {"p": start.copy(), "t": 0.0}
                ReplanMincoTracker(
                    start, goal, lambda: (state["p"], list(Q0), state["t"]), lambda _s: True, Q0,
                    base.TARGET_SPEED, base.MAX_ACCEL, route_waypoints=route[1:-1], via_half_width=0.,
                    face_travel=True, forward_axis=FWD, local_max_vel=.15, local_piece_length_m=1.5,
                    obstacle_grid=grid, obstacle_clearance_soft=.2, local_replan_period=1.0,
                    planning_horizon_m=2.0)
                print(f"x={x:.2f} y={y:.2f},{global_ok},True,{len(pairs)//7},", flush=True)
            except Exception as error:
                print(f"x={x:.2f} y={y:.2f},False,False,0,{type(error).__name__}", flush=True)


if __name__ == "__main__":
    main()
