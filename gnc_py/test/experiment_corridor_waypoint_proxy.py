#!/usr/bin/env python3
"""Test-only proxy for a corridor: add clearance waypoints before MINCO."""
import os, sys
import numpy as np
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from global_minco_candidate_selector import densify_polyline, curve_is_free
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory

Q0=np.array([0.,0.,0.,1.]); FWD=np.array([1.,0.,0.])
def main():
  _, static=sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
  start,goal=base.location('inspection_entry_1'),base.location('nav_entry')
  # The A*6 case that failed the tracker gate.
  grid=base.make_grid(static,[([11.25,-7.20,4.90],base.PERSON_HALF)])
  print('connectivity,variant,waypoints,ok,hit_time')
  for connectivity in (6,26):
    route=base.shortcut_path(base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=connectivity).plan(start,goal),grid,base.BOUNDS)
    for name,offset in [('raw',None),('y+.45',np.array([0.,.45,0.])),('y-.45',np.array([0.,-.45,0.])),('y+.9',np.array([0.,.9,0.])),('y-.9',np.array([0.,-.9,0.]))]:
      points=densify_polyline(route,.5)
      if offset is not None:
        mid=len(points)//2; points=np.insert(points,mid,points[mid]+offset,axis=0)
      try:
        tr=MincoTrajectory(points,Q0,face_travel=True,forward_axis=FWD,body_frame_wrench=True,via_half_width=0.,target_speed=base.TARGET_SPEED,max_accel=base.MAX_ACCEL)
        check=curve_is_free(tr,grid,base.BOUNDS)
        print('%d,%s,%d,%s,%s'%(connectivity,name,len(points),check['ok'],check['hit_time_s']),flush=True)
      except Exception as e: print('%d,%s,%d,False,%s'%(connectivity,name,len(points),type(e).__name__),flush=True)
if __name__=='__main__': main()
