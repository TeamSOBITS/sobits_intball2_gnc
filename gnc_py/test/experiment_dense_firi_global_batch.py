#!/usr/bin/env python3
"""FIRI per <=0.5m A* leg, then one corridor-constrained MINCO, nine cases."""
import os, subprocess, sys
import numpy as np
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from experiment_firi_corridor_minco_batch import EXE,planes_flat,Q0,FWD,corridor_min_margin
from global_minco_candidate_selector import densify_polyline,curve_is_free
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
def main():
 _,static=sobits_intball2_gnc_cpp.load_octomap_points(base.MAP);start,goal=base.location('inspection_entry_1'),base.location('nav_entry')
 print('case,legs,minco,grid_free,corridor_margin_m,duration_s,note')
 for x in (10.65,10.95,11.25):
  for y in (-7.20,-6.60,-6.00):
   try:
    grid=base.make_grid(static,[([x,y,4.90],base.PERSON_HALF)])
    raw=base.shortcut_path(base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=6).plan(start,goal),grid,base.BOUNDS);route=densify_polyline(raw,.5);planes=[]
    for i,(a,b) in enumerate(zip(route[:-1],route[1:])):
     enc=';'.join(','.join(f'{v:.6f}' for v in p) for p in (a,b));out=subprocess.run([EXE,base.MAP,str(x),str(y),enc],text=True,capture_output=True,check=True,timeout=30).stdout
     p=planes_flat(out,1)
     for j in range(0,len(p),5): planes.extend([i,*p[j+1:j+5]])
    tr=MincoTrajectory(route,Q0,face_travel=True,forward_axis=FWD,body_frame_wrench=True,via_half_width=0.,corridor_planes=planes)
    ok=curve_is_free(tr,grid,base.BOUNDS,max_spacing_m=.01,max_time_step_s=.005)['ok']
    print(f'x={x:.2f} y={y:.2f},{len(route)-1},True,{ok},{corridor_min_margin(tr,planes,len(route)-1):.4f},{tr.global_total_duration:.3f},',flush=True)
   except Exception as e:print(f'x={x:.2f} y={y:.2f},-,False,False,-,-,{type(e).__name__}',flush=True)
if __name__=='__main__':main()
