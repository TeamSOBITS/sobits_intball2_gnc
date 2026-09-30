#!/usr/bin/env python3
"""JEM offline failure-mode checks for the global planner contract."""
import os
import sys

import sobits_intball2_gnc_cpp

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base


def outcome(factory, start, goal):
    try:
        factory().plan(start, goal)
        return "UNEXPECTED_PATH"
    except RuntimeError as error:
        return type(error).__name__


def main():
    _resolution, static = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    cases = [
        ("start_occupied", [(start.tolist(), base.PERSON_HALF)], start, goal),
        ("goal_occupied", [(goal.tolist(), base.PERSON_HALF)], start, goal),
        ("start_outside_bounds", [], [9.4, -9.0, 5.0], goal),
        ("full_cross_section", [([10.95, -6.60, 4.90], [1.40, 0.15, 1.30])], start, goal),
    ]
    print("case,A*_6,RRT")
    for name, boxes, case_start, case_goal in cases:
        def grid():
            return base.make_grid(static, boxes)
        astar = outcome(lambda: base.AStarPlanner(base.RESOLUTION, grid=grid(),
                                                   search_bounds=base.BOUNDS, connectivity=6),
                        case_start, case_goal)
        rrt = outcome(lambda: base.RRTPlanner(grid=grid(), search_bounds=base.BOUNDS,
                                               step_size=0.20, goal_tolerance=0.20,
                                               goal_bias=0.30, max_iterations=5000, seed=0),
                      case_start, case_goal)
        print("%s,%s,%s" % (name, astar, rrt), flush=True)


if __name__ == "__main__":
    main()
