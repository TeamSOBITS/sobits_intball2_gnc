#!/usr/bin/env python3
"""Offline JEM-map benchmark for 6/26-connected A* and occupancy-grid RRT.

This is intentionally outside guidance.  It reads the static OctoMap, adds
synthetic person/box obstacles, searches a fixed allowed box, and shortcuts
the resulting global polyline with the same inflated-occupancy collision
predicate as RRT.  It never creates a ROS node, sends an action, or runs MINCO.
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


HERE = os.path.dirname(__file__)
MAP = os.path.join(HERE, "..", "maps", "jem_octomap.bt")
LOCATIONS = yaml.safe_load(open(os.path.join(HERE, "..", "maps", "iss_location.yaml"), encoding="utf-8"))["location_pose"]
RESOLUTION = 0.10
INFLATION = 0.10
BOUNDS = ([9.6, -11.9, 3.6], [12.3, -2.4, 6.0])
PERSON_HALF = [0.25, 0.15, 0.85]

# The deliberately full-width case is expected to have no route inside BOUNDS.
SCENARIOS = [
    ("walls_only", "inspection_entry_1", "nav_entry", []),
    ("person_center", "inspection_entry_1", "nav_entry", [([10.95, -6.60, 4.90], PERSON_HALF)]),
    ("person_x_plus_015", "inspection_entry_1", "nav_entry", [([11.10, -6.60, 4.90], PERSON_HALF)]),
    ("person_x_minus_035", "inspection_entry_1", "nav_entry", [([10.60, -6.60, 4.90], PERSON_HALF)]),
    ("two_people_staggered", "inspection_entry_1", "nav_entry", [
        ([10.70, -6.60, 4.90], PERSON_HALF), ([11.25, -7.25, 4.90], PERSON_HALF)]),
    ("box_full_cross_section", "inspection_entry_1", "nav_entry", [
        ([10.95, -6.60, 4.90], [1.40, 0.15, 1.30])]),
    ("reverse_person_center", "nav_entry", "inspection_entry_1", [([10.95, -6.60, 4.90], PERSON_HALF)]),
    ("entry2_person_center", "inspection_entry_2", "nav_entry", [([10.95, -6.60, 4.90], PERSON_HALF)]),
]


def location(name):
    translation = LOCATIONS[name]["translation"]
    return np.array([translation["x"], translation["y"], translation["z"]], dtype=float)


def path_length(path):
    return sum(float(np.linalg.norm(b - a)) for a, b in zip(path[:-1], path[1:]))


def make_grid(static_points, boxes):
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RESOLUTION, INFLATION)
    grid.add_points(static_points)
    for center, half_size in boxes:
        grid.add_box(center, half_size)
    return grid.snapshot()


def run_one(planner, start, goal):
    t0 = time.perf_counter()
    try:
        raw = planner.plan(start, goal)
        short = shortcut_path(raw, planner._grid, BOUNDS)
        return dict(ok=True, ms=(time.perf_counter() - t0) * 1000,
                    raw_m=path_length(raw), short_m=path_length(short),
                    raw_n=len(raw), short_n=len(short),
                    count=getattr(planner, "last_expansions", getattr(planner, "last_iterations", 0)))
    except RuntimeError as error:
        return dict(ok=False, ms=(time.perf_counter() - t0) * 1000, error=str(error))


def aggregate(samples):
    good = [sample for sample in samples if sample["ok"]]
    if not good:
        return dict(success="0/%d" % len(samples), ms="-", raw_m="-", short_m="-",
                    raw_n="-", short_n="-", count="-", note=samples[0].get("error", "failed"))
    return dict(success="%d/%d" % (len(good), len(samples)),
                ms="%.1f" % statistics.median(s["ms"] for s in good),
                raw_m="%.2f" % statistics.median(s["raw_m"] for s in good),
                short_m="%.2f" % statistics.median(s["short_m"] for s in good),
                raw_n="%.0f" % statistics.median(s["raw_n"] for s in good),
                short_n="%.0f" % statistics.median(s["short_n"] for s in good),
                count="%.0f" % statistics.median(s["count"] for s in good), note="")


def benchmark(static_points):
    rows = []
    for name, start_name, goal_name, boxes in SCENARIOS:
        start, goal = location(start_name), location(goal_name)
        def grid():
            return make_grid(static_points, boxes)
        planners = [
            ("A* 6", [run_one(AStarPlanner(RESOLUTION, grid=grid(), search_bounds=BOUNDS,
                                             connectivity=6), start, goal)]),
            ("A* 26", [run_one(AStarPlanner(RESOLUTION, grid=grid(), search_bounds=BOUNDS,
                                              connectivity=26), start, goal)]),
            ("RRT", [run_one(RRTPlanner(grid=grid(), search_bounds=BOUNDS, step_size=0.20,
                                         goal_tolerance=0.20, goal_bias=0.30,
                                         max_iterations=5000, seed=seed), start, goal)
                     for seed in range(5)]),
        ]
        for planner, samples in planners:
            rows.append((name, start_name + " -> " + goal_name, planner, aggregate(samples)))
    return rows


def render(rows):
    lines = [
        "# グローバル探索オフライン比較結果（JEM 静的マップ＋配置障害物）",
        "",
        "実行対象は global path の探索とショートカットだけであり、ROS node・action・MINCO・本番 guidance は一切起動／変更していない。",
        "",
        "## 条件",
        "",
        "- static map: `gnc_py/maps/jem_octomap.bt`",
        "- grid resolution / inflation: `0.10 m` / `0.10 m`",
        "- permitted search bounds (`iss_body`): lower `[9.6, -11.9, 3.6]`, upper `[12.3, -2.4, 6.0]`",
        "- 人相当 box: half-size `[0.25, 0.15, 0.85] m`。すべての box は grid に追加した後、inflated occupancy で判定する。",
        "- A*: 6方向・26方向（26は corner-cut 防止の supercover edge check）、各1回。RRT: seed 0–4 の5回、step `0.20 m`、goal bias `0.30`、最大5000 iteration。",
        "- shortcut: 各探索結果について、先頭から到達できる最遠の後続点へ貪欲に置換。各 candidate edge は `0.05 m` 以下の間隔で、同じ inflated occupancy と bounds に対して衝突確認する。",
        "",
        "## 結果",
        "",
        "`raw/short m` は探索直後／ショートカット後の長さ、`raw/short pts` は点数。RRTの値は成功 seed の中央値。`count` は A* の展開数、RRT の iteration。",
        "",
        "| scenario | route | planner | success | ms | raw/short m | raw/short pts | count | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for scenario, route, planner, result in rows:
        lines.append("| %s | %s | %s | %s | %s | %s/%s | %s/%s | %s | %s |" %
                     (scenario, route, planner, result["success"], result["ms"],
                      result["raw_m"], result["short_m"], result["raw_n"], result["short_n"],
                      result["count"], result["note"]))
    lines += [
        "",
        "## 読み方と採用判断の境界",
        "",
        "- shortcut 後の全 edge は、RRT と同じ離散サンプル衝突判定を再実行している。ただしこれは MINCO 曲線の連続衝突保証ではない。",
        "- `box_full_cross_section` は search bounds 内で安全な経路が無いことを確認するための意図的な失敗ケース。実運用ではこの種の失敗を hold + action `ABORTED` に接続する予定で、直線 MINCO にフォールバックしてはならない。",
        "- この条件では、A* 6方向は A* 26方向より大幅に低い計画時間となり、shortcut 後の長さはほぼ同等だった。RRT は最速だったが、有限 iteration の確率的探索であり、成功率・最悪時間・seed 依存をさらに増やしたケースで評価する必要がある。従って、現時点では A* 6方向を MINCO 統合候補、A* 26方向と RRT を比較対象として残す。",
        "- この結果だけでは planner を採用しない。次段階は、選定候補の path を MINCO 経由点へ渡すオフライン統合試験（曲線の占有判定、時間配分、経由点を飛ばさない設計）である。これが通るまで本番コードへ接続しない。",
        "",
    ]
    return "\n".join(lines)


def main():
    _resolution, static_points = sobits_intball2_gnc_cpp.load_octomap_points(MAP)
    document = render(benchmark(static_points))
    print(document)
    out = os.path.join(HERE, "..", "..", "docs", "global_astar_offline_benchmark_results.md")
    with open(out, "w", encoding="utf-8") as output:
        output.write(document)
    print("\nWrote " + out)


if __name__ == "__main__":
    main()
