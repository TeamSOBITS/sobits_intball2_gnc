"""Capture every obstacle-grid shape solve made while moving in the stage 5 JEM person
scenario, from the moment the box appears up to the first emergency stop.

    python3 capture_flight.py <ahead_m> <appear_after_m> <inflation_m> <out.json>
"""
import sys

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

import diag_common


def main():
    ahead, appear_after, inflation, out = (float(sys.argv[1]), float(sys.argv[2]),
                                           float(sys.argv[3]), sys.argv[4])
    stage5 = diag_common.load_stage5()
    stage5.stage4.GRID_INFLATION = inflation

    boxes, calls, current = [], [], {"tracker": None, "t": 0.0}

    class RecordingGrid(sobits_intball2_gnc_cpp.OccupancyGrid):
        def add_box(self, center, half):
            boxes.append((list(center), list(half)))
            return super().add_box(center, half)

    stage5.sobits_intball2_gnc_cpp = type("M", (), {k: getattr(sobits_intball2_gnc_cpp, k)
                                            for k in dir(sobits_intball2_gnc_cpp) if not k.startswith("__")})
    stage5.sobits_intball2_gnc_cpp.OccupancyGrid = RecordingGrid

    plan = sobits_intball2_gnc_cpp.plan_minco

    def recording_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        tracker = current["tracker"]
        if kwargs.get("grid") is not None and boxes and tracker is not None and tracker._stop_profile is None:
            a, kw = diag_common.jsonable_args(args, kwargs)
            calls.append({"label": "t%.2f_code%d" % (current["t"], result[1]), "args": a, "kw": kw,
                          "error_code": result[1]})
        return result

    sobits_intball2_gnc_cpp.plan_minco = recording_plan

    sample = ReplanMincoTracker.sample

    def watching_sample(self, t):
        current["tracker"], current["t"] = self, t
        result = sample(self, t)
        if self.emergency_stops > 0:
            diag_common.save_capture(out, inflation, boxes, calls)
            print("CAPTURED stop at t=%.2f" % t, [c["label"] for c in calls], flush=True)
            raise SystemExit(0)
        return result

    ReplanMincoTracker.sample = watching_sample
    stage5.run(True, ahead, stage5.jem.PERSON_ACROSS_Y, stage5.stop_profile_factory(), t_max=60.0,
               progress="capture", appear_after=appear_after)
    diag_common.save_capture(out, inflation, boxes, calls)
    print("NO STOP (saved %d calls)" % len(calls), flush=True)


if __name__ == "__main__":
    main()
