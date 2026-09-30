#!/usr/bin/env python3
"""A*6 -> FIRI half-spaces -> Int-Ball MINCO over the established nine cases."""
import os
import subprocess
import sys

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import curve_is_free
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXE = os.path.join(WORKSPACE, "build", "sobits_intball2_gnc_cpp", "experiment_firi_jem_corridor")
Q0, FWD = np.array([0., 0., 0., 1.]), np.array([1., 0., 0.])


def planes_flat(output, segments):
    result = []
    found = [0] * segments
    for line in output.splitlines():
        if line.startswith("PLANE "):
            item = dict(value.split("=", 1) for value in line.split()[1:])
            segment = int(item["segment"])
            result.extend([segment, float(item["nx"]), float(item["ny"]),
                           float(item["nz"]), float(item["b"])])
            found[segment] += 1
    if not all(found):
        raise RuntimeError("FIRI did not emit a corridor for every MINCO segment")
    return result


def corridor_min_margin(trajectory, planes, segments):
    grouped = [[] for _ in range(segments)]
    for index in range(0, len(planes), 5): grouped[int(planes[index])].append(planes[index + 1:index + 5])
    margin, t = float("inf"), 0.0
    while True:
        p, _v, _a, _q = trajectory.sample(t)
        segment = min(np.searchsorted(trajectory._cum_times[1:], t, side="right"), segments - 1)
        margin = min(margin, *[-(np.dot(plane[:3], p) + plane[3]) / np.linalg.norm(plane[:3])
                             for plane in grouped[segment]])
        if t >= trajectory.global_total_duration: return margin
        t = min(trajectory.global_total_duration, t + 0.005)


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,firi_planes,minco_solved,grid_free,corridor_margin_m,duration_s,solve_ms,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            try:
                grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
                route = base.shortcut_path(base.AStarPlanner(base.RESOLUTION, grid=grid,
                    search_bounds=base.BOUNDS, connectivity=6).plan(start, goal), grid, base.BOUNDS)
                encoded = ";".join(",".join(f"{value:.6f}" for value in point) for point in route)
                run = subprocess.run([EXE, base.MAP, str(x), str(y), encoded], text=True,
                    capture_output=True, timeout=30, check=True)
                planes = planes_flat(run.stdout, len(route) - 1)
                trajectory = MincoTrajectory(route, Q0, face_travel=True, forward_axis=FWD,
                    body_frame_wrench=True, via_half_width=0., corridor_planes=planes)
                check = curve_is_free(trajectory, grid, base.BOUNDS, max_spacing_m=.01, max_time_step_s=.005)
                margin = corridor_min_margin(trajectory, planes, len(route) - 1)
                print(f"x={x:.2f} y={y:.2f},{len(planes)//5},True,{check['ok']},{margin:.4f},"
                      f"{trajectory.global_total_duration:.3f},{trajectory.solve_wall_seconds * 1000:.3f},", flush=True)
            except Exception as error:
                print(f"x={x:.2f} y={y:.2f},-,False,False,-,-,-,{type(error).__name__}:{error}", flush=True)


if __name__ == "__main__":
    main()
