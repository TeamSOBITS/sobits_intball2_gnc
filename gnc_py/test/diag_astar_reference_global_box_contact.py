"""Diagnose box contacts on the paper scenario (serial).

Usage: diag_minco_box_contact.py OUT.csv [LAYOUT] [minco|jaxa:<lookahead_m>]

Adds per-box columns to the 10 Hz trace: clearance to each box, depth-layer
occupied cells near each box (how much of it the grid has), and whether the
vehicle position is inside the planner's inflated grid.
See docs/jaxa_baseline_jaxa_controller_comparison.md section 6.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sobits_intball2_gnc_cpp
import experiment_jaxa_baseline_offline as e
import experiment_astar_reference_global_offline as global_ref_run

OUT = sys.argv[1] if len(sys.argv) > 1 else "minco_contact_trace.csv"
LAYOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
METHOD = sys.argv[3] if len(sys.argv) > 3 else "minco"
res, e.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(e.base.MAP)
e.WORLD = e.DepthWorld(e.STATIC, res)
e.OPTS.update(controller="jaxa")
scen = e.paper_scenario(LAYOUT)
grids = []
new_grid = e.WORLD.new_grid
e.WORLD.new_grid = lambda scen=None: grids.append(new_grid(scen)) or grids[-1]
sim = e.simulate_jaxa


def extra(p, state):
    grid = grids[-1]
    cells = np.asarray(grid.depth_occupied_cells(), dtype=float).reshape(-1, 3)
    row = dict(in_inflation=bool(grid.inflated_occupied(list(p))))
    for k, (c, h) in enumerate(state[1]):
        q = np.clip(p, c - h, c + h)
        row["box%d_near_cells" % k] = int((np.linalg.norm(cells - q, axis=1) <= 0.15).sum()) if len(cells) else 0
        row["box%d_near_occ" % k] = bool(grid.inflated_occupied(list(q)))
    for k, (c, h) in enumerate(state[1]):
        row["box%d_clear" % k] = float(e.box_distance(p, c, h)[0]) - e.ROBOT_RADIUS
        row["box%d_cells" % k] = int((e.box_distance(cells, c, h) <= 2 * e.base.RESOLUTION).sum()) if len(cells) else 0
    return row


def sim_with_extra(step, scen_, grid, fin, trace=None, trace_extra=None, pose_out=None):
    def both(p, state):
        row = trace_extra(p, state) if trace_extra else {}
        row.update(extra(p, state))
        return row
    return sim(step, scen_, grid, fin, trace, both, pose_out)


e.simulate_jaxa = sim_with_extra
trace = []
if METHOD == "minco":
    r = global_ref_run.run_minco_global(scen, trace)
else:
    r = e.run_jaxa(scen, float(METHOD.split(":")[1]), 0, dict(max_iterations=1000, radius=0.6), trace)
print({k: r[k] for k in ("status", "time_s", "min_obstacle_clear_m", "plan_ms_max")})
# Which faces of each box the final grid holds (cells within 2 voxels of the box).
cells = np.asarray(grids[-1].depth_occupied_cells(), dtype=float).reshape(-1, 3)
for k, (c, h) in enumerate(scen.boxes):
    near = cells[e.box_distance(cells, c, h) <= 2 * e.base.RESOLUTION]
    faces = {}
    for name, axis, sign in (("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1), ("+z", 2, 1), ("-z", 2, -1)):
        faces[name] = int((sign * (near[:, axis] - c[axis]) >= h[axis] - 2 * e.base.RESOLUTION).sum())
    print("box%d center %s cells %d per face %s" % (k, np.round(c, 3).tolist(), len(near), faces))
keys = list(dict.fromkeys(k for row in trace for k in row))
with open(OUT, "w", encoding="utf-8") as out:
    out.write(",".join(keys) + "\n")
    for row in trace:
        out.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
