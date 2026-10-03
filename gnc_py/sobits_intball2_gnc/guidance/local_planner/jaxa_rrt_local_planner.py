#!/usr/bin/env python3
"""Python interface to the native JAXA Int-Ball2 local planner.

Re-implements T. Nishishita et al., "Dynamic Motion Planning of FPV Camera
Free-Flyers for Autonomous Crew Tracking and Collision Avoidance", IAC-22-D1.6.2
(2022), 3.1-3.2 from the paper's text (no JAXA source exists for this part):
OMPL RRTstar on the inflated occupancy grid -> partialShortcutPath ->
smoothBSpline (OMPL defaults) -> collision check of the final path, replanning
on collision; and the tracking point of Eqs. (1)-(2). The author used OMPL; the
call sequence and the values the paper does not give are decided in
docs/jaxa_baseline_ompl_reproduction.md. ``planner="rrt"`` is the earlier
hand-written RRT* + B-spline, kept for the reference-equivalence tests.
``trajectory_tracking/jaxa_tracking_point_tracker.py`` drives it.
"""
import time
from dataclasses import dataclass, fields

import numpy as np
import sobits_intball2_gnc_cpp as core

JaxaPlanError = core.JaxaPlanError


@dataclass(frozen=True)
class JaxaPlannerConfig:
    """Values the paper does not give; see docs/jaxa_baseline_ompl_reproduction.md.

    ``rrt_*`` and ``waypoint_spacing_m`` apply to ``planner="rrt"`` only."""
    planner: str = "ompl"
    ompl_solve_time_s: float = 0.1
    rrt_step_m: float = 0.3
    rrt_radius_m: float = 0.6
    rrt_iterations: int = 1000
    rrt_goal_bias: float = 0.1
    rrt_goal_tolerance_m: float = 0.3
    waypoint_spacing_m: float = 0.5
    max_attempts: int = 50



def _native_config(config):
    native = core.JaxaPlannerConfig()
    for field in fields(config):
        setattr(native, field.name, getattr(config, field.name))
    return native


def _point(point):
    values = np.asarray(point, dtype=float)
    if values.shape != (3,):
        raise ValueError("point must have three coordinates")
    return values.tolist()


def _path(path):
    values = np.asarray(path, dtype=float)
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("path must have shape (N, 3)")
    return values.tolist()


class RRTStar:
    """Native RRT* interface for diagnostics; guidance uses plan_local_path."""

    def __init__(self, grid, bounds, step=0.3, radius=0.6, max_iterations=1000,
                 goal_bias=0.1, goal_tolerance=0.3, seed=None):
        self.grid, self.bounds = grid, bounds
        self.config = JaxaPlannerConfig(planner="rrt", rrt_step_m=step, rrt_radius_m=radius,
                                        rrt_iterations=max_iterations, rrt_goal_bias=goal_bias,
                                        rrt_goal_tolerance_m=goal_tolerance)
        self.seed = int(np.random.randint(0, 2**32)) if seed is None else int(seed)

    def plan(self, start, goal):
        return np.asarray(core.jaxa_rrt_path(
            self.grid, _point(start), _point(goal), _point(self.bounds[0]), _point(self.bounds[1]),
            self.seed, _native_config(self.config)), dtype=float)


def bspline_waypoints(points, spacing=0.5):
    """Chord-parameter B-spline interpolation with SciPy's not-a-knot boundaries."""
    return np.asarray(core.jaxa_bspline_waypoints(_path(points), spacing), dtype=float)


def path_is_free(path, grid, bounds):
    """One native collision batch under the grid's read lock; releases the GIL."""
    return core.jaxa_path_is_free(_path(path), grid, _point(bounds[0]), _point(bounds[1]))


def plan_local_path(start, goal, grid, bounds, seed, config):
    """RRT*, interpolation and retries share one snapshot and release the GIL.

    Returns (waypoints (N,3), info) with the existing attempts/plan_s keys.
    plan_s includes snapshot creation and the Python/native call overhead.
    """
    t0 = time.perf_counter()
    path, attempts = core.jaxa_plan_local_path(
        grid, _point(start), _point(goal), _point(bounds[0]), _point(bounds[1]),
        int(seed), _native_config(config))
    waypoints = np.asarray(path, dtype=float)
    return waypoints, {"attempts": attempts, "plan_s": time.perf_counter() - t0}


def tracking_point(path, p, lookahead):
    """Paper Eqs. (1)-(2), with the existing unit-direction and goal-clamp rules."""
    target, index = core.jaxa_tracking_point(_path(path), _point(p), lookahead)
    return np.asarray(target, dtype=float), index
