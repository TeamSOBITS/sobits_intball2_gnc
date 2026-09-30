#!/usr/bin/env python3
"""Compare the current MINCO gate with FIRI->GCOPTER on the nine person cases.

This is test-only: it calls neither ROS nor production guidance.  Both curves
are checked against the same inflated OccupancyGrid snapshot.  The geometric
margin is a conservative point-cloud proxy (voxel-centre distance minus the
inflation and half a voxel diagonal), reported separately from that gate.
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free, select_first_safe


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXE = os.path.join(WORKSPACE, "build", "sobits_intball2_gnc_cpp", "experiment_firi_jem_corridor")
SAMPLE_DT_S = 0.01


def min_margin_proxy(samples, static_tree, center, half):
    """Lower-bound proxy to the uninflated point/box geometry, in metres."""
    points = np.asarray(samples, dtype=float)
    static_distance = float(np.min(static_tree.query(points, workers=-1)[0]))
    delta = np.maximum(np.abs(points - center) - half, 0.0)
    box_distance = float(np.min(np.linalg.norm(delta, axis=1)))
    voxel_half_diagonal = base.RESOLUTION * np.sqrt(3.0) / 2.0
    return min(static_distance, box_distance) - base.INFLATION - voxel_half_diagonal


def minco_samples(trajectory):
    samples, t = [], 0.0
    while True:
        position, _velocity, _accel, _quat = trajectory.sample(t)
        samples.append(position)
        if t >= trajectory.global_total_duration:
            return np.asarray(samples)
        t = min(trajectory.global_total_duration, t + SAMPLE_DT_S)


def parse_gcopter(output):
    summary = next(line for line in output.splitlines() if line.startswith("GCOPTER "))
    fields = dict(item.split("=", 1) for item in summary.split()[1:] if "=" in item)
    points = []
    for line in output.splitlines():
        if line.startswith("SAMPLE "):
            sample = dict(item.split("=", 1) for item in line.split()[1:])
            points.append([float(sample["x"]), float(sample["y"]), float(sample["z"])])
    return fields, np.asarray(points)


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    static_tree = cKDTree(np.asarray(static, dtype=float).reshape(-1, 3))
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,minco_ok,minco_candidate,minco_collision,minco_margin_m,minco_length_m,minco_duration_s,minco_compute_ms,gcopter_ok,gcopter_collision,gcopter_margin_m,gcopter_length_m,gcopter_duration_s,gcopter_compute_ms")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            center = np.asarray([x, y, 4.90])
            grid = base.make_grid(static, [(center, base.PERSON_HALF)])
            planner = base.AStarPlanner(base.RESOLUTION, grid=grid, search_bounds=base.BOUNDS, connectivity=6)
            search_started = time.perf_counter()
            route = base.shortcut_path(planner.plan(start, goal), grid, base.BOUNDS)
            search_ms = (time.perf_counter() - search_started) * 1000
            def build(points):
                return base.MincoTrajectory(points, base.Q0, face_travel=False,
                                            via_half_width=0.0, target_speed=base.TARGET_SPEED,
                                            max_accel=base.MAX_ACCEL)
            selected = select_first_safe(route, grid, base.BOUNDS, build,
                                         spacings=(None, *base.DENSE_SPACINGS_M))
            if selected["ok"]:
                attempt = selected["selected"]
                trajectory = attempt["trajectory"]
                points = minco_samples(trajectory)
                minco_collision = not curve_is_free(trajectory, grid, base.BOUNDS)["ok"]
                minco_margin = min_margin_proxy(points, static_tree, center, np.asarray(base.PERSON_HALF))
                minco_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                minco_duration = float(trajectory.global_total_duration)
                minco_compute = search_ms + trajectory.solve_wall_seconds * 1000
                minco_candidate = "raw" if attempt["spacing_m"] is None else f"{attempt['spacing_m']:.2f}"
            else:
                minco_collision, minco_margin, minco_length, minco_duration, minco_compute, minco_candidate = True, float("nan"), float("nan"), float("nan"), float("nan"), "none"
            encoded = ";".join(",".join(f"{value:.6f}" for value in point) for point in route)
            run = subprocess.run([EXE, base.MAP, str(x), str(y), encoded], text=True,
                                 capture_output=True, timeout=30, check=False)
            fields, points = parse_gcopter(run.stdout)
            gcopter_collision = any(grid.inflated_occupied(point.tolist()) for point in points)
            gcopter_margin = min_margin_proxy(points, static_tree, center, np.asarray(base.PERSON_HALF))
            print(
                f"x={x:.2f} y={y:.2f},{selected['ok']},{minco_candidate},{minco_collision},"
                f"{minco_margin:.3f},{minco_length:.3f},{minco_duration:.3f},{minco_compute:.3f},"
                f"{run.returncode == 0 and fields['corridor_inside'] == '1'},{gcopter_collision},"
                f"{gcopter_margin:.3f},{float(fields['length_m']):.3f},{float(fields['duration_s']):.3f},"
                f"{float(fields['elapsed_ms']):.3f}", flush=True)


if __name__ == "__main__":
    main()
