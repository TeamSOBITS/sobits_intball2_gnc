"""Capture the first rest replan after an emergency stop in the stage 5 JEM person scenario:
the tracker's own shape solve plus the same solve with the seed midpoint forced to given x.

    python3 capture_rest.py <ahead_m> <inflation_m> <out.json> [--mid-x=10.45,10.60,...]
"""
import sys

import numpy as np

import minco_native_py
from sobits_intball2_gnc.guidance.local_planner.minco_local_planner import MincoLocalPlanner
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)

import diag_common

FORCED_MID_X = [10.45, 10.60, 10.95, 11.30]


def seed_with_mid(planner, p0, target_pos, mid, n_pieces):
    """Rest shape seed through a given midpoint (MincoLocalPlanner._rest_shape_seed's polyline)."""
    legs = np.linalg.norm(mid - p0), np.linalg.norm(target_pos - mid)
    total = sum(legs)
    points = []
    for i in range(1, n_pieces):
        s = total * i / n_pieces
        points.append(p0 + (mid - p0) * s / legs[0] if s <= legs[0]
                      else mid + (target_pos - mid) * (s - legs[0]) / legs[1])
    points = np.array(points)
    directions = np.diff(np.vstack([p0, points, target_pos]), axis=0)[1:]
    speed = planner.local_max_vel or planner.global_avg_speed
    return points, directions, [total / speed / n_pieces] * n_pieces


def main():
    ahead, inflation, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    mid_xs = next((list(map(float, a.split("=", 1)[1].split(","))) for a in sys.argv[4:]
                   if a.startswith("--mid-x=")), FORCED_MID_X)
    stage5 = diag_common.load_stage5()
    stage5.stage4.GRID_INFLATION = inflation

    boxes, calls, label, current = [], [], {"now": None}, {"tracker": None}

    class RecordingGrid(minco_native_py.OccupancyGrid):
        def add_box(self, center, half):
            boxes.append((list(center), list(half)))
            return super().add_box(center, half)

    stage5.minco_native_py = type("M", (), {k: getattr(minco_native_py, k)
                                            for k in dir(minco_native_py) if not k.startswith("__")})
    stage5.minco_native_py.OccupancyGrid = RecordingGrid

    plan = minco_native_py.plan_minco

    def recording_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        if label["now"] is not None and kwargs.get("grid") is not None:
            a, kw = diag_common.jsonable_args(args, kwargs)
            calls.append({"label": label["now"], "args": a, "kw": kw, "error_code": result[1]})
        return result

    minco_native_py.plan_minco = recording_plan

    sample = ReplanMincoTracker.sample

    def watching_sample(self, t):
        current["tracker"] = self
        return sample(self, t)

    ReplanMincoTracker.sample = watching_sample

    build = MincoLocalPlanner.build_face_travel_local

    def capturing_build(self, p0, v0, a0, rv0, rr0, ra0, target_pos, v_tail, shape_seed=None,
                        touch_goal=False):
        tracker = current["tracker"]
        if tracker is None or tracker._stop_profile is None or shape_seed is None:
            return build(self, p0, v0, a0, rv0, rr0, ra0, target_pos, v_tail, shape_seed, touch_goal)
        n_pieces = len(shape_seed[0]) + 1
        chord_mid = (np.asarray(p0) + np.asarray(target_pos)) / 2.0
        for x in mid_xs:
            label["now"] = "forced_%.2f" % x
            try:
                build(self, p0, v0, a0, rv0, rr0, ra0, target_pos, v_tail,
                      seed_with_mid(self, p0, target_pos, np.array([x, chord_mid[1], chord_mid[2]]),
                                    n_pieces), touch_goal)
            except Exception:
                pass
        label["now"] = "own_seed"
        try:
            build(self, p0, v0, a0, rv0, rr0, ra0, target_pos, v_tail, shape_seed, touch_goal)
        except Exception:
            pass
        diag_common.save_capture(out, inflation, boxes, calls)
        print("CAPTURED", [(c["label"], c["error_code"]) for c in calls], flush=True)
        raise SystemExit(0)

    MincoLocalPlanner.build_face_travel_local = capturing_build
    stage5.run(True, ahead, stage5.jem.PERSON_ACROSS_Y, stage5.stop_profile_factory(), t_max=40.0,
               progress="capture")
    print("NO REST REPLAN", flush=True)


if __name__ == "__main__":
    main()
