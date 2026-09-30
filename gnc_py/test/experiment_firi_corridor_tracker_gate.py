#!/usr/bin/env python3
"""Initial local-solve gate for A*6/FIRI/corridor-MINCO, nine person cases."""
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
    print('case,horizon_m,global_and_initial_local,note')
    for x in (10.65, 10.95, 11.25):
      for y in (-7.20, -6.60, -6.00):
        try:
          grid=base.make_grid(static,[([x,y,4.90],base.PERSON_HALF)])
          route=base.shortcut_path(base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=6).plan(start,goal),grid,base.BOUNDS)
          encoded=';'.join(','.join(f'{v:.6f}' for v in p) for p in route)
          out=subprocess.run([EXE,base.MAP,str(x),str(y),encoded],text=True,capture_output=True,check=True,timeout=30).stdout
          for horizon in (4.,):
            state={'p':start.copy(),'t':0.0}
            try:
              ReplanMincoTracker(start,goal,lambda:(state['p'],list(Q0),state['t']),lambda _s:True,Q0,
                None,None,route_waypoints=route[1:-1],via_half_width=0.,face_travel=True,forward_axis=FWD,
                local_max_vel=.15,local_piece_length_m=.5,obstacle_grid=grid,obstacle_clearance_soft=.2,
                local_replan_period=1.,planning_horizon_m=horizon,corridor_planes=planes_flat(out,len(route)-1))
              print(f'x={x:.2f} y={y:.2f},{horizon},True,',flush=True)
            except Exception as e: print(f'x={x:.2f} y={y:.2f},{horizon},False,{type(e).__name__}',flush=True)
        except Exception as e: print(f'x={x:.2f} y={y:.2f},-,False,{type(e).__name__}',flush=True)
if __name__=='__main__': main()
