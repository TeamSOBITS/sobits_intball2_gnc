"""Benchmark jaxa_rrt plan_local_path wall time on the layout-0 grid (static map + 3 boxes).

Isolates where solve time goes, outside guidance: run it with and without the
sim stack, and with ``--depth-thread`` (a 10 Hz depth integration in a second
thread of this process, like guidance's DepthImageSubscriber) or
``--busy-thread`` (a Python thread holding the GIL in a tight loop).
See docs/jaxa_baseline_gazebo_port_plan.md 10 節.
"""
import argparse
import json
from pathlib import Path
import os
import statistics
import sys
import threading
import time

import numpy as np
import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_global_planner_minco_jem_cases as base
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import (
    JaxaPlanError, JaxaPlannerConfig, plan_local_path)

BOXES = [(11.108, -6.476, 5.027), (10.843, -7.721, 5.017), (11.202, -4.891, 5.222)]
HALF = (0.3, 0.3, 0.5)
START, GOAL = np.array([10.936, -3.03, 5.0]), np.array([10.936, -8.33, 5.0])

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--depth-thread", action="store_true")
ap.add_argument("--busy-thread", action="store_true")
ap.add_argument("--attempts", type=int, default=1)
ap.add_argument("--output", type=Path, help="write raw wall times and condition metadata as JSON")
args = ap.parse_args()
if args.runs < 1 or args.attempts < 1:
    ap.error("--runs and --attempts must be positive")
_res, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, base.INFLATION)
grid.add_points(static)
for c in BOXES:
    grid.add_box(list(c), list(HALF))
stop = False
if args.depth_thread:
    d = base._GUIDANCE["depth"]
    grid.enable_depth_layer([9.3, -12.3, 2.9], [12.6, -2.0, 6.4], d["p_hit"], d["p_miss"], d["p_min"],
                            d["p_max"], d["p_occ"], d["min_range"], d["max_range"], d["skip_pixel"])
    depth = np.full((200, 200), 2.0, dtype=np.float32)
    rot = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]) @ np.array(
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])

    def integrate():
        while not stop:
            t0 = time.perf_counter()
            grid.integrate_depth(depth, 119.2, 119.2, 100.0, 100.0, rot, START.tolist())
            time.sleep(max(0.0, 0.1 - (time.perf_counter() - t0)))
    threading.Thread(target=integrate, daemon=True).start()
if args.busy_thread:
    def busy():
        x = 0
        while not stop:
            x += 1
    threading.Thread(target=busy, daemon=True).start()
times, failed = [], 0
for k in range(args.runs):
    t0 = time.perf_counter()
    try:
        plan_local_path(START, GOAL, grid, base.BOUNDS, 100 * (k + 1), JaxaPlannerConfig(max_attempts=args.attempts))
    except JaxaPlanError:
        failed += 1
    times.append(time.perf_counter() - t0)
stop = True
print("failed attempts %d/%d (max attempts per run = %d)" % (failed, len(times), args.attempts))
print("depth_thread=%s busy_thread=%s runs=%d attempt_s median %.3f min %.3f max %.3f" % (
    args.depth_thread, args.busy_thread, len(times), statistics.median(times), min(times), max(times)))

if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(backend="cpp", depth_thread=args.depth_thread,
        busy_thread=args.busy_thread, runs=args.runs, max_attempts=args.attempts,
        failed=failed, seconds=times, median_s=statistics.median(times), max_s=max(times)), indent=2)+"\n")
