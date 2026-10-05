#!/usr/bin/env python3
"""Current replan MINCO (straight global) with the sim order: face the goal, integrate depth,
then build the tracker (as guidance_executor: pre_align, then TrackerBuilder.build)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_jaxa_baseline_offline as ex  # noqa: E402
from experiment_local_only_delayed_detection import initial_facing_quat  # noqa: E402

DEPTH_FRAMES = 6
_new_grid = ex.DepthWorld.new_grid


def new_grid_seen(self, scen=None):
    grid = _new_grid(self, scen)
    q = initial_facing_quat(scen.start, scen.goal)
    for _ in range(DEPTH_FRAMES):
        self.integrate(grid, scen.start, q, scen.at(0.0))
    return grid


if __name__ == "__main__":
    ex.DepthWorld.new_grid = new_grid_seen
    ex.main()
