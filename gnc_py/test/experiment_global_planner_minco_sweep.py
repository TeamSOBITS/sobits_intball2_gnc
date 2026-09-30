#!/usr/bin/env python3
"""Systematic person-box sweep for the test-only global-path MINCO gate."""
import os
import sys

import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    print("case,A*_search,A*_accepted,A*_selected,RRT_search,RRT_accepted,RRT_selected")
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            boxes = [([x, y, 4.90], base.PERSON_HALF)]
            def grid():
                return base.make_grid(static, boxes)
            astar = base.run_planner(
                lambda _seed: base.AStarPlanner(base.RESOLUTION, grid=grid(),
                                                  search_bounds=base.BOUNDS, connectivity=6),
                start, goal, 1)
            rrt = base.run_planner(
                lambda seed: base.RRTPlanner(grid=grid(), search_bounds=base.BOUNDS,
                                              step_size=0.20, goal_tolerance=0.20,
                                              goal_bias=0.30, max_iterations=5000, seed=seed),
                start, goal, 3)
            print("x=%.2f y=%.2f,%s,%s,%s,%s,%s,%s" %
                  (x, y, astar["search"], astar["accepted_curve"], astar["selected"],
                   rrt["search"], rrt["accepted_curve"], rrt["selected"]), flush=True)


if __name__ == "__main__":
    main()
