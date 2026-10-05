#!/usr/bin/env python3
"""Attitude-error diagnosis: diag_attitude.py OUT.csv LAYOUT plan4|current [known]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sobits_intball2_gnc_cpp  # noqa: E402
import experiment_jaxa_baseline_offline as e  # noqa: E402
import experiment_astar_reference_global_offline as global_ref_run  # noqa: E402
import experiment_minco_seen_before_departure_offline as current_simlike_run  # noqa: E402
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_rotate  # noqa: E402
from sobits_intball2_gnc.guidance.trajectory_tracking import replan_minco_tracker  # noqa: E402
from sobits_intball2_gnc.guidance.local_planner import minco_local_planner as mlp  # noqa: E402
import time  # noqa: E402

OUT, LAYOUT, METHOD = sys.argv[1], int(sys.argv[2]), sys.argv[3]
res, e.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(e.base.MAP)
e.WORLD = e.DepthWorld(e.STATIC, res)
e.OPTS.update(controller="jaxa", known_boxes=len(sys.argv) > 4 and sys.argv[4] == "known")
scen = e.paper_scenario(LAYOUT)
if os.environ.get("DIAG_PIECE_LENGTH"):  # diagnostic: local_piece_length_m override
    e.base.LOCAL_PIECE_LENGTH = float(os.environ["DIAG_PIECE_LENGTH"])
    print("local_piece_length_m =", e.base.LOCAL_PIECE_LENGTH)
sim = e.simulate_jaxa
LAST = {}


def yaw_pitch(q):
    f = quat_rotate(np.asarray(q, dtype=float), [1.0, 0.0, 0.0])
    return np.degrees(np.arctan2(f[1], f[0])), np.degrees(np.arcsin(np.clip(f[2], -1, 1)))


def sim_wrapped(step, scen_, grid, fin, trace=None, trace_extra=None, pose_out=None):
    def step2(t, p):
        out = step(t, p)
        LAST.update(q_des=out[3], w_des=out[4], t_guid=t)
        return out

    def extra(p, state):
        row = trace_extra(p, state) if trace_extra else {}
        if "q_des" in LAST and pose_out is not None:
            yd, pd = yaw_pitch(LAST["q_des"])
            ya, pa = yaw_pitch(pose_out["q"])
            row.update(yaw_des=yd, pitch_des=pd, yaw_act=ya, pitch_act=pa,
                       w_des=float(np.linalg.norm(LAST["w_des"])))
        return row
    return sim(step2, scen_, grid, fin, trace, extra, pose_out)


e.simulate_jaxa = sim_wrapped
ADOPTS = []
_adopt = replan_minco_tracker.ReplanMincoTracker._adopt_local


def adopt_logged(self, result, lag, source):
    """Log the reference attitude just before and just after each local switch."""
    before = getattr(self, "_last_output", None)
    before = None if before is None else before[3]  # what was last sent to the controller
    stopping = self._stop_profile is not None
    _adopt(self, result, lag, source)
    after = self._local_trajectory.sample(self._local_elapsed)[3]
    ADOPTS.append((LAST.get("t_guid", 0.0), source, lag, result is None, stopping,
                   None if before is None else np.degrees(geodesic_angle(before, after))))


replan_minco_tracker.ReplanMincoTracker._adopt_local = adopt_logged
TIERS = []
_seeded = mlp.MincoLocalPlanner._build_local_seeded


FAILS = []
_target = mlp.MincoLocalPlanner._get_local_target


def target_logged(self, p_from):
    out = _target(self, p_from)
    self._diag_last_target = out
    return out


mlp.MincoLocalPlanner._get_local_target = target_logged


def seeded_logged(self, *args):
    """Time each fallback tier of one local build (warm/straight/astar/random)."""
    t0 = time.perf_counter()
    sys.stderr.write("[tier] %s t=%.1f\n" % (args[-1], LAST.get("t_guid", 0.0)))
    sys.stderr.flush()
    try:
        out = _seeded(self, *args)
        TIERS.append((args[-1], True, time.perf_counter() - t0))
        return out
    except Exception as exc:
        TIERS.append((args[-1], False, time.perf_counter() - t0))
        if args[-1] == "warm":
            grid = self.obstacle_grid
            p0 = np.asarray(args[0], dtype=float)
            tgt, tv, touch = getattr(self, "_diag_last_target", (np.full(3, np.nan), np.zeros(3), False))
            FAILS.append(dict(t=LAST.get("t_guid", 0.0), err=str(exc)[:90],
                              p0_occ=bool(grid.inflated_occupied(list(p0))),
                              tgt_occ=bool(grid.inflated_occupied(list(tgt))),
                              dist=float(np.linalg.norm(tgt - p0)), tv=float(np.linalg.norm(tv)),
                              touch=touch, v0=float(np.linalg.norm(args[1]))))
        raise


mlp.MincoLocalPlanner._build_local_seeded = seeded_logged
SEEDS = []
_seed = mlp.MincoLocalPlanner._astar_shape_seed


def seed_timed(self, p0, target_pos):
    t0 = time.perf_counter()
    try:
        out = _seed(self, p0, target_pos)
        SEEDS.append((True, time.perf_counter() - t0))
        return out
    except Exception:
        SEEDS.append((False, time.perf_counter() - t0))
        raise


mlp.MincoLocalPlanner._astar_shape_seed = seed_timed
trace = []
if METHOD == "plan4":
    r = global_ref_run.run_minco_global(scen, trace)
else:
    e.DepthWorld.new_grid = current_simlike_run.new_grid_seen
    r = e.run_minco(scen, trace)
print({k: r.get(k) for k in ("status", "time_s", "att_err_max_deg", "sat_frac", "note")})
jumps = [a for a in ADOPTS if a[5] is not None and a[5] > 5.0]
print("local switches %d, failed %d; reference attitude jump > 5 deg: %d" % (
    len(ADOPTS), sum(a[3] for a in ADOPTS), len(jumps)))
for a in jumps:
    print("  t=%.2f source=%s lag=%.2f s jump=%.1f deg (stop profile active before: %s)" % (a[0], a[1], a[2], a[5], a[4]))
for mode in ("warm", "straight", "astar", "random"):
    xs = [x for x in TIERS if x[0] == mode]
    if xs:
        ok = [x[2] for x in xs if x[1]]
        ng = [x[2] for x in xs if not x[1]]
        print("tier %-8s tried %3d ok %3d (median %.2f s, max %.2f s) failed %3d (median %.2f s, max %.2f s)" % (
            mode, len(xs), len(ok), np.median(ok) if ok else 0, max(ok) if ok else 0,
            len(ng), np.median(ng) if ng else 0, max(ng) if ng else 0))
if SEEDS:
    xs = [x[1] for x in SEEDS]
    print("A* seed alone: %d calls, ok %d, median %.3f s, max %.3f s" % (
        len(xs), sum(x[0] for x in SEEDS), np.median(xs), max(xs)))
lags = [a[2] for a in ADOPTS if a[2] is not None]
if lags:
    print("lag: median %.2f s, max %.2f s, > 1.5 s: %d" % (np.median(lags), max(lags), sum(l > 1.5 for l in lags)))
print("warm failures:")
for f in FAILS:
    print("  t=%(t).1f p0_in_infl=%(p0_occ)s target_in_infl=%(tgt_occ)s dist=%(dist).2f v0=%(v0).3f target_v=%(tv).3f touch_goal=%(touch)s err=%(err)s" % f)
keys = list(dict.fromkeys(k for row in trace for k in row))
with open(OUT, "w", encoding="utf-8") as out:
    out.write(",".join(keys) + "\n")
    for row in trace:
        out.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
