#!/usr/bin/env python3
"""Offline RRT (three deterministic seeds) -> FIRI -> GCOPTER over nine JEM cases."""
import os
import subprocess
import sys
import time

import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
EXE = os.path.join(WORKSPACE, "build", "sobits_intball2_gnc_cpp", "experiment_firi_jem_corridor")


def parse(output):
    summary = next(line for line in output.splitlines() if line.startswith("GCOPTER "))
    fields = dict(item.split("=", 1) for item in summary.split()[1:] if "=" in item)
    samples = []
    for line in output.splitlines():
        if line.startswith("SAMPLE "):
            item = dict(value.split("=", 1) for value in line.split()[1:])
            samples.append([float(item["x"]), float(item["y"]), float(item["z"])])
    return fields, samples


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,seed,rrt_search,search_ms,shortcut_points,firi_gcopter,grid_collision,length_m,duration_s,gcopter_ms,note")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            grid = base.make_grid(static, [([x, y, 4.90], base.PERSON_HALF)])
            for seed in range(3):
                label = f"x={x:.2f} y={y:.2f}"
                try:
                    planner = base.RRTPlanner(grid=grid, search_bounds=base.BOUNDS, step_size=.20,
                                              goal_tolerance=.20, goal_bias=.30,
                                              max_iterations=5000, seed=seed)
                    started = time.perf_counter()
                    route = base.shortcut_path(planner.plan(start, goal), grid, base.BOUNDS)
                    search_ms = (time.perf_counter() - started) * 1000
                    encoded = ";".join(",".join(f"{value:.6f}" for value in point) for point in route)
                    run = subprocess.run([EXE, base.MAP, str(x), str(y), encoded], text=True,
                                         capture_output=True, timeout=30, check=False)
                    fields, samples = parse(run.stdout)
                    collision = any(grid.inflated_occupied(point) for point in samples)
                    ok = run.returncode == 0 and fields["corridor_inside"] == "1"
                    print(f"{label},{seed},True,{search_ms:.3f},{len(route)},{ok},{collision},"
                          f"{float(fields['length_m']):.3f},{float(fields['duration_s']):.3f},"
                          f"{float(fields['elapsed_ms']):.3f},", flush=True)
                except Exception as error:
                    print(f"{label},{seed},False,-,-,False,-,-,-,-,{type(error).__name__}", flush=True)


if __name__ == "__main__":
    main()
