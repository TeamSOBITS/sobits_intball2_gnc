#!/usr/bin/env python3
"""Plan 4 (replan MINCO with the A* reference global) against a person mesh turned so
its extent along the travel direction (+y, inspection_entry_1 -> nav_entry) is largest.

Usage: experiment_person_max_depth_offline.py <standing|max_depth> X[,X...] [MESH]
  MESH: human_obstacles mesh name (default float_blue; float2_blue is the stretched pose).
  standing: the mesh as in every other experiment (float_blue y-extent 0.94 m, float2 0.82 m).
  max_depth: extra rotation with the largest y-extent on a 10 deg zyx grid (float_blue 1.32 m at
  (20, -50, 100) deg, float2 1.80 m at (40, 40, 60) deg).
  X: person x [m] at y = -6.60, z = 4.90. Obstacles are seen only through the depth camera.
  LATE_D (optional 5th argument): JEM-axis corridor instead (start 0.7 m in from the hatch as the
  paper scenario, goal y = -9.0), person LATE_D [m] ahead of the start so that it is out of the
  camera range (3 m) at departure and seen only on the way.
See docs/minco_astar_reference_global.md 1.7.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sobits_intball2_gnc_cpp  # noqa: E402
import experiment_jaxa_baseline_offline as e  # noqa: E402
import experiment_astar_reference_global_offline as plan4  # noqa: E402


def max_depth_rotation(points):
    """Extra rotation maximizing the extent along +y (the travel direction), 10 deg zyx grid."""
    best = (-1.0, None)
    for yaw in range(0, 180, 10):
        for pitch in range(-90, 91, 10):
            for roll in range(0, 180, 10):
                r = Rotation.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix()
                y = points @ r[1]
                if y.max() - y.min() > best[0]:
                    best = (y.max() - y.min(), r)
    return best[1]


def main():
    pose, xs = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 3:
        e.MESH = os.path.join(os.path.dirname(e.MESH), sys.argv[3] + ".dae")
    res, e.STATIC = sobits_intball2_gnc_cpp.load_octomap_points(e.base.MAP)
    e.WORLD = e.DepthWorld(e.STATIC, res)
    e.OPTS.update(controller="jaxa")
    if pose == "max_depth":
        voxels = np.asarray(sobits_intball2_gnc_cpp.load_mesh_surface_voxels(
            e.MESH, e.VC["mesh_resolution"])).reshape(-1, 3)
        e.WORLD.mesh_r = max_depth_rotation(voxels @ e.WORLD.mesh_r.T) @ e.WORLD.mesh_r
        e.WORLD.tree = cKDTree(voxels @ e.WORLD.mesh_r.T)
    pts = np.asarray(e.WORLD.tree.data)
    print("# mesh=%s pose=%s extent %s" % (os.path.basename(e.MESH), pose, np.round(pts.max(0) - pts.min(0), 2).tolist()))
    print("pose,x,status,time_s,min_clear_m,att_err_max_deg,plans,plan_ms_max,note")
    start, goal = e.base.location("inspection_entry_1"), e.base.location("nav_entry")
    person_y = -6.60
    if len(sys.argv) > 4:
        start = e.paper_scenario(0).start
        goal = np.array([e.JEM_AXIS_X, -9.0, e.JEM_AXIS_Z])
        person_y = start[1] - float(sys.argv[4])
    for x in xs:
        scen = e.Scenario("%s_%.2f" % (pose, x), start, goal, people=[[x, person_y, 4.90]])
        r = plan4.run_minco_global(scen)
        print("%s,%.2f,%s,%.1f,%.3f,%.1f,%d,%.0f,%s" % (
            pose, x, r["status"], r["time_s"], r["min_obstacle_clear_m"], r["att_err_max_deg"],
            r["plans"], r["plan_ms_max"], r["note"]), flush=True)


if __name__ == "__main__":
    main()
