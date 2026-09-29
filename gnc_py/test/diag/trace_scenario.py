"""async_paced_scenario.py + per-solve log (start, occupancy, clearance, result) and free gaps.

    python3 trace_scenario.py <ahead_m> <appear_after_m> [--sync] [--t-max=90] [--latency=S] [--latency-after-box=S] [--proto]

--latency forces each background solve to land no earlier than S sim seconds (sim-like solve times),
--latency-after-box only for solves started once the box is in;
--proto runs early_stop_tracker.EarlyStopTracker instead of ReplanMincoTracker.
"""
import math, sys
import numpy as np
from rclpy.clock import Clock, ClockType
import diag_common
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


class PacedThread:
    def __init__(self, thread, steps):
        self._thread, self._left = thread, steps
    def is_alive(self):
        self._left -= 1
        return self._left > 0
    def join(self):
        self._thread.join()


s5 = diag_common.load_stage5(); stage4, jem = s5.stage4, s5.jem
ahead, appear_after = float(sys.argv[1]), float(sys.argv[2])
sync = "--sync" in sys.argv
t_max = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--t-max=")), 90.0)
steady = Clock(clock_type=ClockType.STEADY_TIME)
omap = ObstacleMap(0.1, 0.2, diag_common.JEM_MAP)
start, goal = jem.location("inspection_entry_1"), jem.location("nav_entry")
q0 = jem.facing_quat(goal - start); route_dir = (goal - start) / np.linalg.norm(goal - start)
state = {"p": start.copy(), "s": 0.0}
latency = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--latency=")), 0.0)
latency_after_box = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--latency-after-box=")), latency)
proto = "--proto" in sys.argv
if proto:
    from early_stop_tracker import EarlyStopTracker as Tracker
else:
    Tracker = ReplanMincoTracker
tr = Tracker(start, goal, lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
    stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
    attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON, face_travel=True,
    forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE, local_piece_length_m=stage4.PIECE_LENGTH,
    obstacle_grid=omap.grid, obstacle_clearance_soft=stage4.CLEARANCE_SOFT,
    stop_profile_fn=s5.stop_profile_factory(), async_replan=not sync)
omap.add_listener(tr.set_obstacle_grid)
half = jem.PERSON_ACROSS_Y
box = None
clock = {"t": 0.0}

orig_build = ReplanMincoTracker._build_local
def logged_build(self, p0, v0, *rest):
    p0 = np.asarray(p0, dtype=float)
    occ = self._planner.obstacle_grid.inflated_occupied(list(p0))
    clr = stage4.box_distance(p0, box, half) - stage4.ROBOT_RADIUS if box is not None else float("nan")
    rest_state = self._stop_profile is not None
    try:
        out = orig_build(self, p0, v0, *rest)
        ok = True
        return out
    except Exception as exc:
        ok = False
        raise
    finally:
        if box is not None:
            print("SOLVE t=%.2f p=[%.3f,%.3f,%.3f] |v|=%.3f startOcc=%d boxClr=%.3f rest=%d ok=%d" % (
                clock["t"], *p0, np.linalg.norm(v0), occ, clr, rest_state, ok), flush=True)
ReplanMincoTracker._build_local = logged_build

def free_gaps(grid, z, ys):
    xs = np.arange(9.6, 12.4, 0.01)
    for y in ys:
        runs, s = [], None
        for x in xs:
            o = grid.inflated_occupied([x, y, z])
            if not o and s is None: s = x
            if o and s is not None: runs.append((s, x)); s = None
        if s is not None: runs.append((s, xs[-1]))
        print("  GAP y=%.2f z=%.2f free x: %s" % (y, z, ", ".join("%.2f..%.2f(%.2f)" % (a, b, b - a) for a, b in runs)), flush=True)

t, clr, stop_ts, seen_thread, first_col = 0.0, np.inf, [], None, False
dt = stage4.DT
while t <= tr.total_duration and t < t_max:
    t += dt; clock["t"] = t
    stops_before = tr.emergency_stops
    p, v, _a, _q = tr.sample(t)
    state["p"], state["s"] = p, t
    th = tr._pending_thread
    if not sync and th is not None and th is not seen_thread and not isinstance(th, PacedThread):
        t0 = steady.now(); th.join(); wall = (steady.now() - t0).nanoseconds * 1e-9
        tr._pending_thread = seen_thread = PacedThread(th, max(1, math.ceil(max(wall, latency_after_box if box is not None else latency) / dt)))
    if box is None and (p - start) @ route_dir >= appear_after:
        depth = abs(half @ np.abs(route_dir))
        box = start + route_dir * ((p - start) @ route_dir + ahead + depth)
        box[2] = jem.FLOOR_Z + half[2]
        omap.apply(set_boxes={("manual", 0): (box, half)})
        print("BOX t=%.2f center=%s half=%s vehicle p=%s |v|=%.3f" % (t, np.round(box, 3), half, np.round(p, 3), np.linalg.norm(v)), flush=True)
        print("GAPS inflation 0.2 (grid):", flush=True)
        for z in (4.8, 5.0, 5.2):
            free_gaps(omap.grid, z, [box[1] - half[1], box[1], box[1] + half[1]])
    if box is not None:
        clr = min(clr, stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS)
        if not first_col and tr.last_collision_ahead_s is not None:
            first_col = True
            print("COLLISION-SEEN t=%.2f p=%s |v|=%.3f ahead_s=%.2f" % (t, np.round(p, 3), np.linalg.norm(v), tr.last_collision_ahead_s), flush=True)
    if tr.emergency_stops > stops_before:
        stop_ts.append(round(t, 2))
        print("ESTOP t=%.2f p=%s boxClr=%.3f" % (t, np.round(p, 3), stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS), flush=True)
print("RESULT latency_after_box=%.2f proto=%s early_stops=%s brake_resumes=%s outcomes=%s latency=%.2f sync=%s ahead=%.1f stops=%d at %s clearance=%.3f t=%.1f goal_err=%.3f final_p=%s" % (
    latency_after_box, proto, getattr(tr, "early_stops", None), getattr(tr, "brake_resumes", None), dict(getattr(tr, "brake_outcomes", {})), latency, sync, ahead, tr.emergency_stops, stop_ts, clr, t, np.linalg.norm(state["p"] - goal), np.round(state["p"], 3)), flush=True)
