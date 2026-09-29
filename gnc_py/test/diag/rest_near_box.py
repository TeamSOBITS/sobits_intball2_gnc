"""Start at rest on the route at a given clearance in front of the 1.5 m-case box, box already in the
map, and see whether replan_minco ever gets moving (like sim run 4 stopping 0.059 m from the box).

    python3 rest_near_box.py <box_x> <box_y> <box_z> <clearance_m> [--t-max=20] [--reverse]
"""
import sys
import numpy as np
import diag_common
from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

s5 = diag_common.load_stage5(); stage4, jem = s5.stage4, s5.jem
box = np.array([float(a) for a in sys.argv[1:4]]); target_clr = float(sys.argv[4])
t_max = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--t-max=")), 20.0)
half = jem.PERSON_ACROSS_Y
route_start, goal = jem.location("inspection_entry_1"), jem.location("nav_entry")
if "--reverse" in sys.argv:
    route_start, goal = goal, route_start
route_dir = (goal - route_start) / np.linalg.norm(goal - route_start)

def clr_at(s):
    return stage4.box_distance(route_start + route_dir * s, box, half) - stage4.ROBOT_RADIUS
lo, hi = 0.0, float((box - route_start) @ route_dir)
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if clr_at(mid) > target_clr else (lo, mid)
start = route_start + route_dir * lo
start[2] = route_start[2]

omap = ObstacleMap(0.1, 0.2, diag_common.JEM_MAP)
q0 = jem.facing_quat(goal - start)
state = {"p": start.copy(), "s": 0.0}

orig_build = ReplanMincoTracker._build_local
def logged_build(self, p0, v0, *rest):
    try:
        out = orig_build(self, p0, v0, *rest); ok = 1; return out
    except Exception as exc:
        ok = 0; print("  build fail: %s" % str(exc)[:120], flush=True); raise
    finally:
        print("SOLVE t=%.2f p=%s ok=%d" % (state["s"], np.round(np.asarray(p0), 3), ok), flush=True)
ReplanMincoTracker._build_local = logged_build

try:
    tr = ReplanMincoTracker(start, goal, lambda: (state["p"], list(q0), state["s"]), lambda s: True, q0,
        stage4.TS, stage4.MA, via_half_width=0.0, wrench_safety_margin=stage4.MARGIN,
        attitude_resample_spacing_m=stage4.SPACING, planning_horizon_m=stage4.HORIZON, face_travel=True,
        forward_axis=stage4.FWD, local_max_vel=stage4.CRUISE, local_piece_length_m=stage4.PIECE_LENGTH,
        obstacle_grid=omap.grid, obstacle_clearance_soft=stage4.CLEARANCE_SOFT,
        stop_profile_fn=s5.stop_profile_factory(), async_replan=False)
except Exception as exc:
    print("RESULT clr=%.3f constructor failed: %s" % (target_clr, str(exc)[:200]), flush=True)
    raise SystemExit(0)
omap.add_listener(tr.set_obstacle_grid)
omap.apply(set_boxes={("manual", 0): (box, half)})
print("START p=%s boxClr=%.3f startOcc=%d" % (np.round(start, 3), clr_at(lo), omap.grid.inflated_occupied(list(start))), flush=True)
t, min_clr = 0.0, np.inf
while t < t_max:
    t += stage4.DT
    p, v, _a, _q = tr.sample(t)
    state["p"], state["s"] = p, t
    min_clr = min(min_clr, stage4.box_distance(p, box, half) - stage4.ROBOT_RADIUS)
print("RESULT clr=%.3f rest_failures=%d moved=%.3f m stops=%d min_clr=%.3f fallback=%s" % (
    target_clr, tr._rest_replan_failures, np.linalg.norm(state["p"] - start), tr.emergency_stops, min_clr, tr.last_fallback_reason), flush=True)
