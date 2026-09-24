"""Stage 4 with async_replan=True (the production default), emulating RTF=1: the
background solve stays "alive" for its own compute time (solve_wall_seconds) in sim
time, so the tracker keeps playing the old local meanwhile, as on the robot.
"""
import importlib.util
import os

import numpy as np

from sobits_intball2_gnc.guidance.trajectory_tracking import replanning_minco_v3_tracker as v3

_spec = importlib.util.spec_from_file_location(
    "stage4", os.path.join(os.path.dirname(__file__), "experiment_obstacle_stage4_tracker.py"))
stage4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage4)


class SimTimedThread:
    """Runs the solve at start(), then reports alive for its compute time in sim time."""
    remaining = []
    last_solve = [0.3]

    def __init__(self, target, args, daemon):
        self._target, self._args = target, args
        self._left = 0.0

    def start(self):
        self._target(*self._args)
        result = SimTimedThread.owner._pending_result
        if isinstance(result, tuple):
            SimTimedThread.last_solve[0] = result[0].solve_wall_seconds
        self._left = SimTimedThread.last_solve[0]
        SimTimedThread.remaining.append(self)

    def is_alive(self):
        return self._left > 0.0


class _Threading:
    Thread = SimTimedThread


def run_async(boxes):
    orig_threading = v3.threading
    orig_init = v3.ReplanningMincoV3Tracker.__init__

    def init(self, *a, **kw):
        SimTimedThread.owner = self
        orig_init(self, *a, async_replan=True, **kw)

    v3.threading = _Threading
    v3.ReplanningMincoV3Tracker.__init__ = init
    orig_sample = v3.ReplanningMincoV3Tracker.sample

    def sample(self, t):
        for th in SimTimedThread.remaining:
            th._left -= stage4.DT
        return orig_sample(self, t)

    v3.ReplanningMincoV3Tracker.sample = sample
    try:
        return stage4.run(boxes)
    finally:
        v3.threading = orig_threading
        v3.ReplanningMincoV3Tracker.__init__ = orig_init
        v3.ReplanningMincoV3Tracker.sample = orig_sample
        SimTimedThread.remaining.clear()


def main():
    print("async_replan=True, RTF=1 emulated; cruise %.2f, soft %.1f" % (stage4.CRUISE, stage4.CLEARANCE_SOFT))
    for name, boxes in stage4.SCENARIOS:
        r = run_async(boxes)
        print("%-24s t=%6.1fs replans=%3d fails=%3d clearance=%7.3f lateral=%.3f face=%5.1f "
              "solve_max=%.3fs goal_moved=%s end=%s" % (
                  name, r["t"], r["n"], r["fails"], r["clr"], r["lat"], r["face"],
                  r["solve_max"], r["moved"], np.round(r["end"], 3)), flush=True)


if __name__ == "__main__":
    main()
