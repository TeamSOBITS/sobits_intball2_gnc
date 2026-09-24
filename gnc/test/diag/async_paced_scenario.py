"""Stage 5 JEM person scenario with async replanning (production), sim time paced like
RTF=1: each background solve is joined, its wall time measured (rclpy steady clock, not
the time module), and the tracker keeps seeing it as running for that many sim seconds
before it lands. Without this pacing, sim time races ahead of the solve offline.

    python3 async_paced_scenario.py <ahead_m> <appear_after_m> [--sync-on-collision]

--sync-on-collision makes only the collision-triggered replan synchronous (diagnostic).
"""
import math, sys
import numpy as np
from rclpy.clock import Clock, ClockType
import diag_common
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.trajectory_tracking.replanning_minco_v3_tracker import ReplanningMincoV3Tracker

class PacedThread:
    """Looks alive for ceil(wall/dt) more sample() calls after the real solve finished."""
    def __init__(self, thread, steps):
        self._thread, self._left = thread, steps
    def is_alive(self):
        self._left -= 1
        return self._left > 0
    def join(self):
        self._thread.join()

s5 = diag_common.load_stage5(); stage4, jem = s5.stage4, s5.jem
ahead, appear_after = float(sys.argv[1]), float(sys.argv[2])
steady = Clock(clock_type=ClockType.STEADY_TIME)
omap = ObstacleMap(0.1, 0.2, diag_common.JEM_MAP)
start, goal = jem.location("inspection_entry_1"), jem.location("nav_entry")
q0 = jem.facing_quat(goal - start); route_dir = (goal - start) / np.linalg.norm(goal - start)
state = {"p": start.copy(), "s": 0.0}
tr = ReplanningMincoV3Tracker(start, goal, lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
    stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
    attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON, face_travel=True,
    forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE, local_piece_length_m=stage4.PIECE_LENGTH,
    obstacle_grid=omap.grid, obstacle_clearance_soft=stage4.CLEARANCE_SOFT,
    stop_profile_fn=s5.stop_profile_factory(), async_replan=True)
omap.add_listener(tr.set_obstacle_grid)
if "--sync-on-collision" in sys.argv:
    orig_check = ReplanningMincoV3Tracker._check_collision
    def check(self):
        self._async_replan = False
        try:
            orig_check(self)
        finally:
            self._async_replan = True
    ReplanningMincoV3Tracker._check_collision = check
half = jem.PERSON_ACROSS_Y
box, t, clr, stop_ts, walls, lags = None, 0.0, np.inf, [], [], []
seen_thread = None
while t <= tr.total_duration and t < 90.0:
    t += stage4.DT
    stops_before = tr.emergency_stops
    p, v, _a, _q = tr.sample(t)
    state["p"], state["s"] = p, t
    th = tr._pending_thread
    if th is not None and th is not seen_thread and not isinstance(th, PacedThread):
        t0 = steady.now(); th.join(); wall = (steady.now() - t0).nanoseconds * 1e-9
        walls.append(wall)
        tr._pending_thread = seen_thread = PacedThread(th, max(1, math.ceil(wall / stage4.DT)))
    if tr.last_replan_lag_seconds is not None and tr.last_replan_occurred:
        lags.append(tr.last_replan_lag_seconds)
    if box is None and (p - start) @ route_dir >= appear_after:
        depth = abs(half @ np.abs(route_dir))
        box = start + route_dir * ((p - start) @ route_dir + ahead + depth)
        box[2] = jem.FLOOR_Z + half[2]
        omap.apply(set_boxes={("manual", 0): (box, half)})
    if box is not None:
        clr = min(clr, stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS)
    if tr.emergency_stops > stops_before:
        stop_ts.append(round(t, 2))
print("RESULT async RTF=1 sync_on_collision=%s ahead=%.1f stops=%d at %s clearance=%.3f t=%.1f goal_err=%.3f solves=%d wall_mean=%.3f wall_max=%.3f lag_max=%.2f" % (
    "--sync-on-collision" in sys.argv, ahead, tr.emergency_stops, stop_ts, clr, t, np.linalg.norm(state["p"] - goal), len(walls),
    np.mean(walls), max(walls), max(lags, default=0.0)), flush=True)
