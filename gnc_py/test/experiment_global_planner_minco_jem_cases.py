#!/usr/bin/env python3
"""Offline A*6/RRT global-path to MINCO safety comparison on the JEM map.

This file is a verification experiment only.  It does not instantiate a ROS
node, tracker, action, or production guidance code.  A shortcut-safe polyline
is passed to ``MincoTrajectory`` with fixed vias, then the *resulting curve*
is sampled against the same inflated occupancy grid.
"""
import os
import statistics
import time

import numpy as np
import yaml

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import shortcut_path
from sobits_intball2_gnc.guidance.global_planner.rrt_planner import RRTPlanner
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoTrajectory,
)
from global_minco_candidate_selector import select_first_safe


HERE = os.path.dirname(__file__)
MAP = os.path.join(HERE, "..", "maps", "jem_octomap.bt")
LOCATIONS = yaml.safe_load(open(os.path.join(HERE, "..", "maps", "iss_location.yaml"), encoding="utf-8"))["location_pose"]
# This experiment deliberately uses the shipped parameters, not independent
# literals.  It evaluates the per-goal avoidance profile without changing it.
_PARAMS = yaml.safe_load(open(os.path.join(HERE, "..", "config", "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
_GUIDANCE = _PARAMS["guidance"]
_TRAJECTORY_CONTROLLER = _PARAMS["trajectory_controller"]
OFFLINE_PROFILE = "avoidance"
RESOLUTION = float(_GUIDANCE["obstacle_grid_resolution"])
INFLATION = float(_GUIDANCE["obstacle_grid_inflation"])
BOUNDS = ([9.6, -11.9, 3.6], [12.3, -2.4, 6.0])
PERSON_HALF = [0.25, 0.15, 0.85]
Q0 = [0.0, 0.0, 0.0, 1.0]
TARGET_SPEED = float(_GUIDANCE["target_speed"])
MAX_ACCEL = min(_TRAJECTORY_CONTROLLER["max_force"]) / float(_TRAJECTORY_CONTROLLER["mass"])
LOCAL_REPLAN_PERIOD = float(_GUIDANCE["minco_local_replan_period"])
LOCAL_HORIZON = float(_GUIDANCE["minco_planning_horizon_m"])
LOCAL_MAX_VEL = float(_GUIDANCE["minco_local_max_vel"])
LOCAL_PIECE_LENGTH = float(_GUIDANCE["minco_local_piece_length_m"])
LOCAL_CLEARANCE_SOFT = float(_GUIDANCE["minco_obstacle_clearance_soft"])
CURVE_SAMPLE_PERIOD_S = 0.01
DENSE_SPACINGS_M = (0.50, 0.25, 0.10)

SCENARIOS = [
    ("walls_only", "inspection_entry_1", "nav_entry", []),
    ("person_center", "inspection_entry_1", "nav_entry", [([10.95, -6.60, 4.90], PERSON_HALF)]),
    ("person_x_plus_015", "inspection_entry_1", "nav_entry", [([11.10, -6.60, 4.90], PERSON_HALF)]),
    ("two_people_staggered", "inspection_entry_1", "nav_entry", [
        ([10.70, -6.60, 4.90], PERSON_HALF), ([11.25, -7.25, 4.90], PERSON_HALF)]),
    ("box_full_cross_section", "inspection_entry_1", "nav_entry", [
        ([10.95, -6.60, 4.90], [1.40, 0.15, 1.30])]),
    ("reverse_person_center", "nav_entry", "inspection_entry_1", [([10.95, -6.60, 4.90], PERSON_HALF)]),
]


def location(name):
    translation = LOCATIONS[name]["translation"]
    return np.array([translation["x"], translation["y"], translation["z"]], dtype=float)


def make_grid(static_points, boxes):
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RESOLUTION, INFLATION)
    grid.add_points(static_points)
    for center, half_size in boxes:
        grid.add_box(center, half_size)
    return grid.snapshot()


def global_path(planner, start, goal):
    t0 = time.perf_counter()
    raw = planner.plan(start, goal)
    return shortcut_path(raw, planner._grid, BOUNDS), (time.perf_counter() - t0) * 1000


def summarize(samples):
    good = [sample for sample in samples if sample["planned"]]
    if not good:
        return dict(search="0/%d" % len(samples), accepted_curve="-", raw_curve="-", dense_050="-",
                    dense_025="-", dense_010="-",
                    plan_ms="-", solve_ms="-", short_points="-", selected="-", note=samples[0]["note"])
    selected_good = [sample for sample in good if sample["selection"]["ok"]]
    return dict(
        search="%d/%d" % (len(good), len(samples)),
        accepted_curve="%d/%d" % (sum(s["selection"]["ok"] for s in good), len(good)),
        raw_curve="%d/%d" % (sum(s["raw"]["ok"] for s in good), len(good)),
        dense_050="%d/%d" % (sum(s["dense"][0.50]["ok"] for s in good), len(good)),
        dense_025="%d/%d" % (sum(s["dense"][0.25]["ok"] for s in good), len(good)),
        dense_010="%d/%d" % (sum(s["dense"][0.10]["ok"] for s in good), len(good)),
        plan_ms="%.1f" % statistics.median(s["plan_ms"] for s in good),
        solve_ms=("%.1f" % statistics.median(
            s["selection"]["selected"]["trajectory"].solve_wall_seconds * 1000
            for s in selected_good) if selected_good else "-"),
        short_points="%.0f" % statistics.median(len(s["path"]) for s in good),
        selected=", ".join(sorted({"raw" if s["selection"]["selected"]["spacing_m"] is None
                                    else "%.2fm" % s["selection"]["selected"]["spacing_m"]
                                    for s in selected_good})),
        note=next(("hit at %.2fs" % s["raw"]["hit_time_s"]
                   for s in good if not s["raw"]["ok"] and "hit_time_s" in s["raw"]), ""),
    )


def run_planner(factory, starts, goals, runs):
    samples = []
    for run in range(runs):
        planner = factory(run)
        try:
            path, plan_ms = global_path(planner, starts, goals)
            def build(points):
                return MincoTrajectory(
                    points, Q0, face_travel=False, via_half_width=0.0,
                    target_speed=TARGET_SPEED, max_accel=MAX_ACCEL)
            selection = select_first_safe(path, planner._grid, BOUNDS, build,
                                          spacings=(None, *DENSE_SPACINGS_M),
                                          stop_on_first_safe=False)
            attempts = {attempt["spacing_m"]: attempt for attempt in selection["attempts"]}
            samples.append(dict(planned=True, path=path, plan_ms=plan_ms, selection=selection,
                                raw=attempts[None],
                                dense={spacing: attempts.get(spacing, {"ok": False})
                                       for spacing in DENSE_SPACINGS_M}))
        except RuntimeError as error:
            samples.append(dict(planned=False, note=str(error)))
    return summarize(samples)


def benchmark(static_points):
    rows = []
    for name, start_name, goal_name, boxes in SCENARIOS:
        start, goal = location(start_name), location(goal_name)
        def grid():
            return make_grid(static_points, boxes)
        rows.append((name, start_name + " -> " + goal_name, "A* 6", run_planner(
            lambda _seed: AStarPlanner(RESOLUTION, grid=grid(), search_bounds=BOUNDS,
                                        connectivity=6), start, goal, 1)))
        rows.append((name, start_name + " -> " + goal_name, "RRT (seed 0-4)", run_planner(
            lambda seed: RRTPlanner(grid=grid(), search_bounds=BOUNDS, step_size=0.20,
                                    goal_tolerance=0.20, goal_bias=0.30,
                                    max_iterations=5000, seed=seed), start, goal, 5)))
    return rows


def render(rows):
    lines = [
        "# A* 6方向・RRT → MINCO のオフライン安全比較（JEM）",
        "",
        "本書は global path を shortcut した後に MINCO 曲線へ変換した検証である。ROS node・action・`ReplanMincoTracker`・本番 guidance は起動も変更もしていない。",
        "",
        "## 判定条件",
        "",
        "- JEM static OctoMap + 人／箱を `0.10 m` inflation した occupancy grid。search bounds は `[9.6, -11.9, 3.6]`–`[12.3, -2.4, 6.0]` (`iss_body`)。",
        "- global path は A* 6方向または RRT で探索後、同じ occupancy に対し最大 `0.05 m` 間隔の線分衝突確認をする greedy shortcut を適用。",
        "- MINCO は via を固定（`via_half_width=0`）、姿勢固定、`target_speed=0.15 m/s`、`max_accel=0.03097 m/s²`。曲線は最大 `0.01 s`、かつ瞬時速度から最大 `0.025 m` 間隔となるよう占有を確認する。",
        "- `raw curve` は shortcut 点をそのまま MINCO に入力。`0.5/0.25/0.1m-dense curve` は各 shortcut edge を各値以下へ直線分割してから同じ MINCO に入力した結果。",
        "",
        "## 結果",
        "",
        "| scenario | route | planner | search | accepted | selected candidate | raw curve free | 0.5m | 0.25m | 0.1m | search ms | selected MINCO ms | shortcut points | raw failure evidence |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario, route, planner, result in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" %
                     (scenario, route, planner, result["search"], result["accepted_curve"], result["selected"],
                      result["raw_curve"], result["dense_050"], result["dense_025"], result["dense_010"],
                      result["plan_ms"], result["solve_ms"],
                      result["short_points"], result["note"]))
    lines += [
        "",
        "## 結論と次のゲート",
        "",
        "- shortcut 線分が安全でも、少数の固定 via を通す MINCO 曲線は角を丸めて障害物へ入り得る。従って shortcut だけを production の `via_waypoints` に渡してはならない。",
        "- 密化の効果はケース依存である。比較上の `accepted candidate` は raw/0.5/0.25/0.1 m の少なくとも一つが実曲線チェックに通った率である。実装するなら固定幅を安全保証として扱わず、候補ごとに解いた global MINCO 曲線を occupancy チェックし、通った候補だけ採用、全滅なら安全に失敗させる gate にする。",
        "- A* 6 と RRT は、上記の dense MINCO curve が通ること、最悪時間、RRT の seed 成功率を増やしたケースで比較を継続する。ここを通過するまで本番 guidance へ接続しない。",
        "",
    ]
    return "\n".join(lines)


def main():
    _resolution, static_points = sobits_intball2_gnc_cpp.load_octomap_points(MAP)
    print("offline_profile=%s target_speed=%.2f grid=%.2f/%.2f local=%.1fm/%.1fs/%.2fmps/%.1fm/%.1fm" %
          (OFFLINE_PROFILE, TARGET_SPEED, RESOLUTION, INFLATION, LOCAL_HORIZON,
           LOCAL_REPLAN_PERIOD, LOCAL_MAX_VEL, LOCAL_PIECE_LENGTH, LOCAL_CLEARANCE_SOFT))
    document = render(benchmark(static_points))
    print(document)
    out = os.path.join(HERE, "..", "..", "docs", "global_planner_minco_offline_comparison.md")
    with open(out, "w", encoding="utf-8") as output:
        output.write(document)
    print("\nWrote " + out)


if __name__ == "__main__":
    main()
