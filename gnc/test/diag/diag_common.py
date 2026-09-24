"""Shared helpers for the obstacle-avoidance solve diagnostics (see README.md)."""
import importlib.util
import json
import os

import numpy as np

import minco_native_py

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JEM_MAP = os.path.join(TEST_DIR, "..", "maps", "jem_octomap.bt")
GRID_RES = 0.1


def load_stage5():
    spec = importlib.util.spec_from_file_location(
        "stage5", os.path.join(TEST_DIR, "experiment_obstacle_stage5_emergency_stop.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable_args(args, kwargs):
    """plan_minco positional/keyword arguments as JSON (the grid is rebuilt on replay)."""
    def conv(v):
        return list(map(float, v)) if hasattr(v, "__len__") else v
    return [conv(a) for a in args], {k: (None if k == "grid" else conv(v)) for k, v in kwargs.items()}


def save_capture(path, inflation, boxes, calls):
    with open(path, "w") as f:
        json.dump({"inflation": inflation, "boxes": boxes, "calls": calls}, f)


def load_capture(path):
    with open(path) as f:
        return json.load(f)


def build_grid(capture):
    grid = minco_native_py.OccupancyGrid(GRID_RES, capture["inflation"])
    grid.add_points(minco_native_py.load_octomap_points(JEM_MAP)[1])
    for center, half in capture["boxes"]:
        grid.add_box(center, half)
    return grid


def position_at(coeffs, segment_times, t):
    """Position of a plan_minco result (coeffs_flat: per piece 6 ascending coeffs x 3 position
    dims, then 3 rotation dims)."""
    for k, duration in enumerate(segment_times):
        if t <= duration or k == len(segment_times) - 1:
            c = np.array(coeffs[k * 36:k * 36 + 18]).reshape(3, 6)
            return c @ (min(t, duration) ** np.arange(6))
        t -= duration


def inflated_hits(grid, coeffs, segment_times, step=0.02):
    """Sample times at which the solved trajectory is inside the inflated grid."""
    duration = sum(segment_times)
    return [t for t in np.arange(0.0, duration, step)
            if grid.inflated_occupied(list(position_at(coeffs, segment_times, t)))]


def replay(call, grid):
    kwargs = dict(call["kw"])
    kwargs["grid"] = grid
    return minco_native_py.plan_minco(*call["args"], **kwargs)
