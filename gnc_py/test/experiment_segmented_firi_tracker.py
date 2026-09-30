#!/usr/bin/env python3
"""Offline waypoint-by-waypoint A*6/FIRI/MINCO/local validation, nine cases."""
import os, subprocess, sys
import numpy as np
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from experiment_firi_corridor_minco_batch import EXE, planes_flat, Q0, FWD
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

def main():
    _, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location('inspection_entry_1'), base.location('nav_entry')
    print('case,legs,global_local_ok,leg_goal_error_m,note')
    for x in (10.65,10.95,11.25):
      for y in (-7.20,-6.60,-6.00):
        try:
          grid=base.make_grid(static,[([x,y,4.90],base.PERSON_HALF)])
          route=base.shortcut_path(base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=6).plan(start,goal),grid,base.BOUNDS)
          current=route[0].copy(); errors=[]
          for target in route[1:]:
            encoded=';'.join(','.join(f'{v:.6f}' for v in p) for p in (current,target))
            out=subprocess.run([EXE,base.MAP,str(x),str(y),encoded],text=True,capture_output=True,check=True,timeout=30).stdout
            state={'p':current.copy(),'t':0.0}
            tracker=ReplanMincoTracker(current,target,lambda:(state['p'],list(Q0),state['t']),lambda _s:True,Q0,
              None,None,via_half_width=0.,face_travel=True,forward_axis=FWD,local_max_vel=.15,
              local_piece_length_m=1.5,obstacle_grid=grid,obstacle_clearance_soft=.2,
              local_replan_period=1.,planning_horizon_m=4.,corridor_planes=planes_flat(out,1))
            t=0.0
            while t < tracker.trajectory.global_total_duration + 1.0:
              t += .05; p,*_ = tracker.sample(t); state['p'],state['t']=p,t
            errors.append(float(np.linalg.norm(state['p']-target)))
            current=target.copy()
          print(f'x={x:.2f} y={y:.2f},{len(route)-1},True,{max(errors):.4f},',flush=True)
        except Exception as e: print(f'x={x:.2f} y={y:.2f},-,False,-,{type(e).__name__}',flush=True)
if __name__=='__main__':main()
