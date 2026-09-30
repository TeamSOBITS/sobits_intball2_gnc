#!/usr/bin/env python3
"""Start/goal variation sweep for the offline A*6/RRT-to-MINCO gate."""
import os
import sys

import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base


CASES = [
    ("entry1_center", "inspection_entry_1", "nav_entry", [10.95, -6.60, 4.90]),
    ("entry2_center", "inspection_entry_2", "nav_entry", [10.95, -6.60, 4.90]),
    ("entry3_center", "inspection_entry_3", "nav_entry", [10.95, -6.60, 4.90]),
    ("reverse_center", "nav_entry", "inspection_entry_1", [10.95, -6.60, 4.90]),
    ("entry3_offset", "inspection_entry_3", "nav_entry", [11.25, -6.00, 4.90]),
]


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    print("case,A*_search,A*_accepted,A*_selected,RRT_search,RRT_accepted,RRT_selected")
    for name, start_name, goal_name, center in CASES:
        start, goal = base.location(start_name), base.location(goal_name)
        def grid():
            return base.make_grid(static, [(center, base.PERSON_HALF)])
        astar = base.run_planner(
            lambda _seed: base.AStarPlanner(base.RESOLUTION, grid=grid(),
                                              search_bounds=base.BOUNDS, connectivity=6), start, goal, 1)
        rrt = base.run_planner(
            lambda seed: base.RRTPlanner(grid=grid(), search_bounds=base.BOUNDS,
                                          step_size=0.20, goal_tolerance=0.20,
                                          goal_bias=0.30, max_iterations=5000, seed=seed), start, goal, 3)
        print("%s,%s,%s,%s,%s,%s,%s" %
              (name, astar["search"], astar["accepted_curve"], astar["selected"],
               rrt["search"], rrt["accepted_curve"], rrt["selected"]), flush=True)


if __name__ == "__main__":
    main()
