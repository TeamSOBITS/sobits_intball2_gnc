"""Offline comparison; build the extension in test/jaxa_cpp_probe first."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import sobits_intball2_gnc_cpp as core
import jaxa_cpp_probe as native
import offline_common as base
from jaxa_python_reference import (
    RRTStar, bspline_waypoints, path_is_free,
)

BOXES = [(11.108, -6.476, 5.027), (10.843, -7.721, 5.017), (11.202, -4.891, 5.222)]
START = np.array([10.936, -3.03, 5.0])
GOAL = np.array([10.936, -8.33, 5.0])


def make_grid():
    _, points = core.load_octomap_points(base.MAP)
    grid = core.OccupancyGrid(base.RESOLUTION, base.INFLATION)
    grid.add_points(points)
    for box in BOXES:
        grid.add_box(box, [0.3, 0.3, 0.5])
    return grid


def python_attempt(grid, seed):
    snapshot = grid.snapshot()
    raw = RRTStar(snapshot, base.BOUNDS, seed=seed).plan(START, GOAL)
    path = bspline_waypoints(raw)
    return np.asarray(raw), path, path_is_free(path, snapshot, base.BOUNDS)


def cpp_attempt(grid, seed, release_gil=True):
    raw, path, free = native.attempt(grid, START.tolist(), GOAL.tolist(), *base.BOUNDS, seed, release_gil=release_gil)
    return np.asarray(raw), np.asarray(path), free


def check_equivalence(grid, seeds):
    for seed in seeds:
        raw_py, path_py, free_py = python_attempt(grid, seed)
        raw_cpp, path_cpp, free_cpp = cpp_attempt(grid, seed)
        np.testing.assert_allclose(raw_cpp, raw_py, rtol=0, atol=1e-10)
        np.testing.assert_allclose(path_cpp, path_py, rtol=0, atol=1e-10)
        assert free_py == free_cpp, (seed, free_py, free_cpp)
        assert path_is_free(raw_cpp, grid, base.BOUNDS)
        if free_cpp:
            assert path_is_free(path_cpp, grid, base.BOUNDS)
    return len(seeds)


def benchmark(grid, seeds, busy, attempts):
    stop = threading.Event()
    ticks = [0]

    def worker():
        while not stop.is_set():
            ticks[0] += 1

    thread = threading.Thread(target=worker) if busy else None
    if thread:
        thread.start()
    rows = []
    try:
        for seed in seeds:
            for backend in ("python", "cpp_hold_gil", "cpp_release_gil"):
                tick_start = ticks[0]
                t0 = time.perf_counter()
                success, error = False, ""
                for attempt in range(attempts):
                    try:
                        if backend == "python":
                            _, path, success = python_attempt(grid, seed + attempt)
                        else:
                            _, path, success = cpp_attempt(grid, seed + attempt, backend == "cpp_release_gil")
                    except RuntimeError as exc:
                        error = str(exc)
                        break
                    if success:
                        break
                elapsed = time.perf_counter() - t0
                rows.append(dict(backend=backend, busy=busy, seed=seed, attempt_limit=attempts,
                                 attempts=attempt + 1, success=bool(success), seconds=elapsed,
                                 worker_ticks=ticks[0] - tick_start, error=error))
    finally:
        stop.set()
        if thread:
            thread.join()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    seeds = [100 * (k + 1) for k in range(args.runs)]
    grid = make_grid()
    count = check_equivalence(grid, seeds)
    print("layout0 raw/spline/collision equivalence: %d seeds passed" % count, flush=True)
    rows = []
    summaries = []
    for attempts in (1, 5):
        for busy in (False, True):
            measured = benchmark(grid, seeds, busy, attempts)
            rows.extend(measured)
            for backend in ("python", "cpp_hold_gil", "cpp_release_gil"):
                group = [r for r in measured if r["backend"] == backend]
                seconds = [r["seconds"] for r in group]
                item = dict(backend=backend, busy=busy, attempt_limit=attempts,
                            runs=len(group), successes=sum(r["success"] for r in group),
                            median_s=statistics.median(seconds), max_s=max(seconds),
                            median_worker_ticks=statistics.median(r["worker_ticks"] for r in group))
                summaries.append(item)
                print(json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(dict(equivalent_seeds=count, summary=summaries), indent=2)+"\n")


if __name__ == "__main__":
    main()
