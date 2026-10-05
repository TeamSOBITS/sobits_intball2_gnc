#!/usr/bin/env python3
"""Replan MINCO reference jumps at async adoption: offline comparison of two fixes.

Usage: experiment_minco_switch_time_offline.py VARIANT CASES [known]
  CASES: comma list of paper layout seeds and/or "person" (person_x_plus_015: a float_blue mesh
         at (11.10, -6.60), inspection_entry_1 -> nav_entry). Without "known" every obstacle is
         seen only through the depth camera (the sim condition).
  VARIANT: none | switch (fix 1) | switch_global (fix 1 + 2)
  fix 1: plan from the reference at the expected adoption time (now + expected wait) and switch
         exactly there; a result that arrives later than that is dropped.
  fix 2: the A* seed tier of the local fallback is replaced by the global reference between the
         vehicle and the local target (seed only; the local may leave the global).
Runs plan 4 (experiment_astar_reference_global_offline.run_minco_global); production code is
patched at run time only. See docs/minco_astar_reference_global.md 1.6.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sobits_intball2_gnc_cpp  # noqa: E402
import experiment_jaxa_baseline_offline as e  # noqa: E402
import experiment_astar_reference_global_offline as plan4  # noqa: E402
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle  # noqa: E402
from sobits_intball2_gnc.guidance.local_planner import minco_local_planner as mlp  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoInfeasibleError  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking import replan_minco_tracker as rmt  # noqa: E402

Tracker = rmt.ReplanMincoTracker
STATS = {}
_orig_adopt = Tracker._adopt_local
_orig_start = Tracker._start_background_replan
_orig_sample = Tracker.sample


def _log_jump(self, before_q):
    after_q = self._local_trajectory.sample(self._local_elapsed)[3]
    STATS["jumps"].append(np.degrees(geodesic_angle(before_q, after_q)))


def adopt_logged(self, result, lag, source):
    before = getattr(self, "_last_output", None)
    _orig_adopt(self, result, lag, source)
    if before is not None and result is not None and not isinstance(result, Exception):
        _log_jump(self, before[3])
    if result is not None and not isinstance(result, Exception):
        STATS["lags"].append(lag)


# --- fix 1 -------------------------------------------------------------------------------------
def start_at_switch(self):
    if getattr(self, "_deferred", None) is not None:
        return  # a result is already waiting for its switch time
    local = self._local_trajectory
    t_sw = min(self._local_elapsed + self._expected_replan_wait_s(), local.global_total_duration)
    p, v, a, _q = local.sample(t_sw)
    rv = local.sample_rotvec_derivatives(t_sw)
    self._sw_at, self._sw_local = t_sw, local
    self._start_background_solve((p, v, a, *rv, local, t_sw))


def adopt_at_switch(self, result, lag, source):
    if (source != "async" or result is None or isinstance(result, Exception)
            or getattr(self, "_sw_local", None) is None):
        return adopt_logged(self, result, lag, source)
    if self._local_trajectory is not self._sw_local or self._local_elapsed > self._sw_at + 1e-9:
        STATS["dropped"] += 1  # late, or the local changed meanwhile: its start is gone
        self.last_local_fallback = True
        return None
    self._deferred = result
    return None


def sample_at_switch(self, t):
    out = _orig_sample(self, t)
    deferred = getattr(self, "_deferred", None)
    if deferred is None:
        return out
    if self._stop_profile is not None or self._local_trajectory is not self._sw_local:
        self._deferred = None
        STATS["dropped"] += 1
        return out
    if self._local_elapsed + 1e-9 < self._sw_at:
        return out
    self._deferred = None
    adopt_logged(self, deferred, self._local_elapsed - self._sw_at, "async")
    p, v, a, q = self._local_trajectory.sample(self._local_elapsed)
    self.last_body_angular = self._local_trajectory.sample_body_angular(self._local_elapsed)
    self._last_output = (p, v, a, q)
    return self._last_output


# --- fix 2 -------------------------------------------------------------------------------------
def global_seed(self, p0, target_pos):
    """Seed along the global reference from the point nearest ``p0`` to the local target."""
    p0 = np.asarray(p0, dtype=float)
    g = self.global_trajectory
    t_end = max(self._global_search_t, 1e-3)
    ts = np.linspace(0.0, t_end, max(20, int(t_end / 0.1)))
    pts = np.array([g.sample(t)[0] for t in ts])
    k = int(np.argmin(np.linalg.norm(pts - p0, axis=1)))
    path = np.vstack([p0, pts[k + 1:], np.asarray(target_pos, dtype=float)])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6])
    path = path[keep]
    if len(path) < 2:
        raise MincoInfeasibleError("global seed: degenerate path")
    n_pieces = max(2, math.ceil(mlp._polyline_length(path) / (0.5 * self._local_piece_length_m)))
    cuts = mlp._equal_arc_cuts(path, n_pieces)
    if cuts is None:
        raise MincoInfeasibleError("global seed: degenerate path")
    length, inner_points, segments = cuts
    inner_directions = np.array([mlp._unit(path[s + 1] - path[s]) for s in segments])
    return inner_points, inner_directions, np.full(n_pieces, length / self.global_avg_speed / n_pieces)


def main():
    variant, cases = sys.argv[1], sys.argv[2].split(",")
    res, e.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(e.base.MAP)
    e.WORLD = e.DepthWorld(e.STATIC, res)
    e.OPTS.update(controller="jaxa", known_boxes=len(sys.argv) > 3 and sys.argv[3] == "known")
    Tracker._adopt_local = adopt_logged
    if variant in ("switch", "switch_global"):
        Tracker._start_background_replan = start_at_switch
        Tracker._adopt_local = adopt_at_switch
        Tracker.sample = sample_at_switch
    if variant == "switch_global":
        mlp.MincoLocalPlanner._astar_shape_seed = global_seed
    print("variant,case,status,time_s,min_clear_m,att_err_max_deg,att_err_mean_deg,sat_frac,"
          "switches,jumps_gt5,max_jump_deg,dropped,max_lag_s")
    for case in cases:
        STATS.update(jumps=[], lags=[], dropped=0)
        if case == "person":
            scen = e.Scenario("person", e.base.location("inspection_entry_1"), e.base.location("nav_entry"),
                              people=[[11.10, -6.60, 4.90]])
        else:
            scen = e.paper_scenario(int(case))
        r = plan4.run_minco_global(scen)
        j = np.asarray(STATS["jumps"])
        print("%s,%s,%s,%.1f,%.3f,%.1f,%.1f,%.2f,%d,%d,%.1f,%d,%.2f" % (
            variant, case, r["status"], r["time_s"], r["min_obstacle_clear_m"], r["att_err_max_deg"],
            r["att_err_mean_deg"], r["sat_frac"], len(j), int((j > 5).sum()), j.max() if len(j) else 0,
            STATS["dropped"], max(STATS["lags"]) if STATS["lags"] else 0), flush=True)


if __name__ == "__main__":
    main()
