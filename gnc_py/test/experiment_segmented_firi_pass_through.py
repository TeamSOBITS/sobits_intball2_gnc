#!/usr/bin/env python3
"""Offline FIRI/MINCO waypoint pass-through sweep over nine person cases."""
import os, subprocess, sys
import numpy as np
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from experiment_firi_corridor_minco_batch import EXE, planes_flat, Q0, FWD, corridor_min_margin
from global_minco_candidate_selector import curve_is_free
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

SPEED=.15
def axis_legs(raw):
    """Keep A* cell-centre turns; each returned edge is a verified 6-neighbour run."""
    points=[np.asarray(p,float) for p in raw]
    result=[points[0]]; previous=None
    for a,b in zip(points[:-1],points[1:]):
        direction=np.sign(b-a).astype(int)
        if previous is not None and not np.array_equal(direction,previous): result.append(a)
        previous=direction
    result.append(points[-1]); return np.asarray(result)
def unit(v):
    n=np.linalg.norm(v); return v/n if n else np.zeros(3)
def main():
 _,static=sobits_intball2_gnc_cpp.load_octomap_points(base.MAP); start,goal=base.location('inspection_entry_1'),base.location('nav_entry')
 print('case,legs,pass_speeds_mps,global_free,corridor_margin_m,total_duration_s,note')
 for x in (10.65,10.95,11.25):
  for y in (-7.20,-6.60,-6.00):
   try:
    grid=base.make_grid(static,[([x,y,4.90],base.PERSON_HALF)])
    raw=base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=6).plan(start,goal)
    route=axis_legs(raw)
    v0=np.zeros(3); speeds=[]; duration=0.; margins=[]; free=True
    for i,(a,b) in enumerate(zip(route[:-1],route[1:])):
     tail=np.zeros(3) if i==len(route)-2 else SPEED*max(0.,np.dot(unit(b-a),unit(route[i+2]-b)))*unit(route[i+2]-b)
     speeds.append(np.linalg.norm(tail)); encoded=';'.join(','.join(f'{v:.6f}' for v in p) for p in (a,b))
     out=subprocess.run([EXE,base.MAP,str(x),str(y),encoded],text=True,capture_output=True,check=True,timeout=30).stdout
     planes=planes_flat(out,1)
     tr=MincoTrajectory([a,b],Q0,v0=v0,face_travel=True,forward_axis=FWD,body_frame_wrench=True,via_half_width=0.,v_tail=tail,corridor_planes=planes)
     free &= curve_is_free(tr,grid,base.BOUNDS,max_spacing_m=.01,max_time_step_s=.005)['ok']; margins.append(corridor_min_margin(tr,planes,1)); duration+=tr.global_total_duration; v0=tail
    print(f'x={x:.2f} y={y:.2f},{len(route)-1},{";".join(f"{s:.3f}" for s in speeds)},{free},{min(margins):.4f},{duration:.3f},',flush=True)
   except Exception as e: print(f'x={x:.2f} y={y:.2f},-,-,False,-,-,{type(e).__name__}',flush=True)
if __name__=='__main__':main()
