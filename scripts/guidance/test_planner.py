#!/usr/bin/env python3
"""Path Planner 結合テストスクリプト.

使い方:
  rosrun sobits_intball2_gnc test_planner.py --start 0 0 4.25 --goal 2 0 4.25
  rosrun sobits_intball2_gnc test_planner.py --planner safety  # 安全重視プランナーをテスト

事前準備:
  roslaunch sobits_intball2_gnc iss_static_map_server.launch
"""
import argparse
import sys
import os

# guidance パッケージを読み込めるようにパスを調整
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import rospy
# 最新のディレクトリ構造に合わせてインポート
from guidance import (
    CollisionChecker, 
    AStarPlanner, 
    SafetyAwareAStarPlanner, 
    apply_path_filters, 
    publish_path
)


def main():
    parser = argparse.ArgumentParser(description="Path Planner test")
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 4.25])
    parser.add_argument("--goal", nargs=3, type=float, default=[2.0, 0.0, 4.25])
    parser.add_argument("--planner", type=str, choices=["normal", "safety"], default="normal")
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--resolution", type=float, default=0.15)
    args, _ = parser.parse_known_args()

    rospy.init_node("test_path_planner", anonymous=True)

    start = np.array(args.start)
    goal = np.array(args.goal)

    # 探索範囲の計算
    bbox_pad = 2.0
    bbox_min = np.minimum(start, goal) - bbox_pad
    bbox_max = np.maximum(start, goal) + bbox_pad

    # 1. CollisionChecker の初期化
    rospy.loginfo("=== Test: CollisionChecker ===")
    try:
        # 最新の引数名 (robot_radius, safety_margin) に対応
        cc = CollisionChecker(
            robot_radius=args.radius, 
            safety_margin=args.margin, 
            bbox_min=bbox_min, 
            bbox_max=bbox_max
        )
        rospy.loginfo("CollisionChecker OK")
    except Exception as e:
        rospy.logerr("CollisionChecker FAILED: %s", e)
        return

    s_ok = cc.check_point(start)
    g_ok = cc.check_point(goal)
    rospy.loginfo("Start %s: %s", start, "FREE" if s_ok else "COLLISION")
    rospy.loginfo("Goal  %s: %s", goal, "FREE" if g_ok else "COLLISION")

    # 2. Planner の選択と実行
    rospy.loginfo("=== Test: %s Planner ===", args.planner.upper())
    if args.planner == "safety":
        cc.compute_edt()
        planner = SafetyAwareAStarPlanner(grid_resolution=args.resolution, weight=1.5)
    else:
        planner = AStarPlanner(grid_resolution=args.resolution)

    raw_path = planner.plan(start, goal, cc)
    
    if not raw_path:
        rospy.logwarn("No path found!")
        return
    
    rospy.loginfo("Raw path: %d waypoints", len(raw_path))
    publish_path(raw_path, topic="/planned_path_raw", frame_id="map")

    # 3. Filter (Smoother) パイプラインの適用
    rospy.loginfo("=== Test: apply_path_filters ===")
    # 以前の個別関数呼び出しから統合パイプラインへ変更
    smooth_path = apply_path_filters(raw_path, cc, push_step=0.05)
    
    if not smooth_path:
        rospy.logerr("Smoothing failed!")
        return

    rospy.loginfo("Smoothed path: %d waypoints", len(smooth_path))
    publish_path(smooth_path, topic="/planned_path", frame_id="map")

    for i, wp in enumerate(smooth_path):
        rospy.loginfo("  WP[%d]: (%.3f, %.3f, %.3f)", i, wp[0], wp[1], wp[2])

    rospy.loginfo("=== Done. Topics: /planned_path_raw, /planned_path ===")
    rospy.loginfo("Check the paths in RViz. Press Ctrl+C to exit")
    rospy.spin()


if __name__ == "__main__":
    main()