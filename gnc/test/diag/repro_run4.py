"""Offline reproduction of sim run 4: reverse route, box at [10.958,-6.877,4.9] appearing at a given
surface clearance while a background solve (old grid) is in flight, solve latency forced like the sim.

    python3 repro_run4.py [--latency=0.7] [--latency-after-box=S] [--appear-clr=0.62] [--no-inflight] [--t-max=60] [--proto [--no-b1] [--no-b2] [--no-b4]]
"""
import math, sys
import numpy as np
from rclpy.clock import Clock, ClockType
import diag_common
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker


def arg(name, default):
    return next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--%s=" % name)), default)


class PacedThread:
    def __init__(self, thread, steps):
        self._thread, self._left = thread, steps
    def is_alive(self):
        self._left -= 1
        return self._left > 0
    def join(self):
        self._thread.join()


s5 = diag_common.load_stage5(); stage4, jem = s5.stage4, s5.jem
latency, appear_clr, t_max = arg("latency", 0.7), arg("appear-clr", 0.62), arg("t-max", 60.0)
latency_after_box = arg("latency-after-box", latency)
want_inflight = "--no-inflight" not in sys.argv
steady = Clock(clock_type=ClockType.STEADY_TIME)
omap = ObstacleMap(0.1, 0.2, diag_common.JEM_MAP)
start, goal = jem.location("nav_entry"), jem.location("inspection_entry_1")
q0 = jem.facing_quat(goal - start)
state = {"p": start.copy(), "s": 0.0}
proto = "--proto" in sys.argv
if proto:
    from early_stop_tracker import EarlyStopTracker
    Tracker = lambda *a, **k: EarlyStopTracker(*a, use_b1="--no-b1" not in sys.argv, use_b2="--no-b2" not in sys.argv, use_b4="--no-b4" not in sys.argv, **k)
else:
    Tracker = ReplanMincoTracker
tr = Tracker(start, goal, lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
    stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
    attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON, face_travel=True,
    forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE, local_piece_length_m=stage4.PIECE_LENGTH,
    obstacle_grid=omap.grid, obstacle_clearance_soft=stage4.CLEARANCE_SOFT,
    stop_profile_fn=s5.stop_profile_factory(), async_replan=True)
omap.add_listener(tr.set_obstacle_grid)
box, half = np.array([10.958, -6.877, 4.9]), jem.PERSON_ACROSS_Y
def clr(p):
    return stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS

clock = {"t": 0.0}
orig_build = ReplanMincoTracker._build_local
def logged_build(self, p0, v0, *rest):
    p0 = np.asarray(p0, dtype=float)
    try:
        out = orig_build(self, p0, v0, *rest); ok = 1; return out
    except Exception:
        ok = 0; raise
    finally:
        if box_in["on"]:
            print("  SOLVE start_t=%.2f p_y=%.3f |v|=%.3f startOcc=%d clr=%.3f rest=%d ok=%d" % (
                self._solve_start_t, p0[1], np.linalg.norm(v0), self._planner.obstacle_grid.inflated_occupied(list(p0)),
                clr(p0), self._stop_profile is not None, ok), flush=True)
ReplanMincoTracker._build_local = logged_build
orig_start_bg = ReplanMincoTracker._start_background_replan
def start_bg(self):
    self._solve_start_t = clock["t"]
    return orig_start_bg(self)
ReplanMincoTracker._start_background_replan = start_bg
ReplanMincoTracker._solve_start_t = float("nan")

box_in = {"on": False}
t, dt, seen, min_clr, events = 0.0, stage4.DT, None, np.inf, []
armed_t = None
resumes_seen = 0
while t <= tr.total_duration and t < t_max:
    t += dt; clock["t"] = t
    stops_before = tr.emergency_stops
    p, v, _a, _q = tr.sample(t)
    state["p"], state["s"] = p, t
    th = tr._pending_thread
    fresh = th is not None and th is not seen and not isinstance(th, PacedThread)
    if fresh:
        t0 = steady.now(); th.join(); wall = (steady.now() - t0).nanoseconds * 1e-9
        tr._pending_thread = seen = PacedThread(th, max(1, math.ceil(max(wall, latency_after_box if box_in["on"] else latency) / dt)))
    if not box_in["on"] and clr(p) <= appear_clr + 1e-9 and armed_t is None:
        armed_t = t
    if not box_in["on"] and armed_t is not None and (not want_inflight or fresh):
        omap.apply(set_boxes={("manual", 0): (box, half)})
        box_in["on"] = True
        print("BOX t=%.2f p_y=%.3f clr=%.3f |v|=%.3f inflight_old_solve=%s expected_wait=%s" % (
            t, p[1], clr(p), np.linalg.norm(v), fresh,
            round(tr.expected_wait_s(), 3) if hasattr(tr, "expected_wait_s") else None), flush=True)
    if box_in["on"]:
        min_clr = min(min_clr, clr(p))
    if getattr(tr, "brake_resumes", 0) > resumes_seen:
        resumes_seen = tr.brake_resumes
        print("RESUME t=%.2f p_y=%.3f clr=%.3f |v|=%.3f" % (t, p[1], clr(p), np.linalg.norm(v)), flush=True)
    if tr.emergency_stops > stops_before:
        print("ESTOP t=%.2f p_y=%.3f clr=%.3f |v|=%.3f" % (t, p[1], clr(p), np.linalg.norm(v)), flush=True)
    if box_in["on"] and int(t / 5) != int((t - dt) / 5):
        print("  t=%.1f p_y=%.3f clr=%.3f |v|=%.3f stopped=%s rest_failures=%d" % (
            t, p[1], clr(p), np.linalg.norm(v), tr._stop_profile is not None, tr._rest_replan_failures), flush=True)
print("RESULT latency_after_box=%.2f proto=%s b1=%s b2=%s b4=%s early_stops=%s brake_resumes=%s brake_solves=%s outcomes=%s latency=%.2f appear_clr=%.2f stops=%d min_clr=%.3f startOcc_end=%d t=%.1f goal_err=%.3f still_stopped=%s" % (
    latency_after_box, proto, "--no-b1" not in sys.argv, "--no-b2" not in sys.argv, "--no-b4" not in sys.argv, getattr(tr, "early_stops", None), getattr(tr, "brake_resumes", None), getattr(tr, "brake_solves", None), dict(getattr(tr, "brake_outcomes", {})),
    latency, appear_clr, tr.emergency_stops, min_clr, omap.grid.inflated_occupied(list(state["p"])), t,
    np.linalg.norm(state["p"] - goal), tr._stop_profile is not None), flush=True)
