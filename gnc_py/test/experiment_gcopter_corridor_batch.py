#!/usr/bin/env python3
"""Offline A*6 -> FIRI -> GCOPTER batch over the established person sweep."""
import os, subprocess, sys
import sobits_intball2_gnc_cpp
sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
EXE = os.path.join(WORKSPACE, 'build', 'sobits_intball2_gnc_cpp', 'experiment_firi_jem_corridor')

def main():
    _, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location('inspection_entry_1'), base.location('nav_entry')
    print('case,astar,route_points,gcopter,corridor_inside,elapsed_ms')
    for x in (10.65, 10.95, 11.25):
      for y in (-7.20, -6.60, -6.00):
        grid = base.make_grid(static, [([x,y,4.90],base.PERSON_HALF)])
        try:
          route = base.shortcut_path(base.AStarPlanner(base.RESOLUTION,grid=grid,search_bounds=base.BOUNDS,connectivity=6).plan(start,goal),grid,base.BOUNDS)
          encoded = ';'.join(','.join(f'{v:.6f}' for v in p) for p in route)
          run = subprocess.run([EXE, base.MAP, str(x), str(y), encoded], text=True, capture_output=True, timeout=30)
          summary = next((line for line in run.stdout.splitlines() if line.startswith('GCOPTER ')), '')
          fields = dict(item.split('=',1) for item in summary.split()[1:] if '=' in item)
          print(f'x={x:.2f} y={y:.2f},True,{len(route)},{run.returncode == 0},{fields.get("corridor_inside", "False")},{fields.get("elapsed_ms", "-")}', flush=True)
        except Exception as error:
          print(f'x={x:.2f} y={y:.2f},False,-,False,False,{type(error).__name__}', flush=True)
if __name__ == '__main__': main()
