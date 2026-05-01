#!/usr/bin/env python3
"""A* アルゴリズムによる 3D 経路計画実装."""
import heapq
import math
import numpy as np
import rospy
from gnc_defaults import GNC_DEFAULTS
from .base_planner import BasePlanner

class AStarPlanner(BasePlanner):
    """3D グリッド上の A* 経路計画."""

    # 26 方向の隣接オフセット (dx, dy, dz)
    # 事前にコスト（距離）を計算しておくことでループ内の sqrt を削減
    _NEIGHBORS = [
        (dx, dy, dz, math.sqrt(dx**2 + dy**2 + dz**2))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(self, grid_resolution=None, bbox_margin=1.0):
        self._res = grid_resolution if grid_resolution is not None else rospy.get_param('/gnc/grid_resolution', GNC_DEFAULTS['grid_resolution'])
        self._bbox_margin = bbox_margin
        self._max_iter = int(rospy.get_param('/gnc/astar_max_iter', GNC_DEFAULTS['astar_max_iter']))
        rospy.loginfo("AStarPlanner params: grid_resolution=%.3f, max_iter=%d", self._res, self._max_iter)

    def _compute_cost(self, world_pos, move_dist, collision_checker):
        """移動コストを計算する。オーバーライドして安全マージンコストを追加可能."""
        # デフォルトは単純な移動距離（最短経路重視）
        return move_dist

    def plan(self, start, goal, collision_checker, **kwargs):
        start_pos = np.asarray(start, dtype=float)
        goal_pos = np.asarray(goal, dtype=float)
        res = self._res

        # 探索範囲の確定 (Navigator側と整合性をとる)
        bb_min = np.minimum(start_pos, goal_pos) - self._bbox_margin
        bb_max = np.maximum(start_pos, goal_pos) + self._bbox_margin

        def to_grid(pos):
            # 四捨五入によるインデックス化
            return tuple(np.round((pos - bb_min) / res).astype(int))

        def to_world(gidx):
            return bb_min + np.array(gidx, dtype=float) * res

        start_idx = to_grid(start_pos)
        goal_idx = to_grid(goal_pos)

        # start/goal 自体が衝突していたら即失敗
        if not collision_checker.check_point(start_pos):
            rospy.logwarn("AStarPlanner: start position is in collision")
            return []
        if not collision_checker.check_point(goal_pos):
            rospy.logwarn("AStarPlanner: goal position is in collision")
            return []

        # ヒューリスティック: ユークリッド距離
        def heuristic(idx):
            return math.sqrt(sum((idx[d] - goal_idx[d])**2 for d in range(3))) * res

        # A* 探索
        open_set = []  # (f, counter, node)
        counter = 0
        g_score = {start_idx: 0.0}
        came_from = {}

        heapq.heappush(open_set, (heuristic(start_idx), counter, start_idx))
        counter += 1

        # 探索ノード制限 (無限ループ防止)
        max_iter = kwargs.get("max_iter", self._max_iter)
        iters = 0

        while open_set and iters < max_iter:
            iters += 1
            f, _, current = heapq.heappop(open_set)

            # ゴール到達判定 (インデックス一致、または物理距離が解像度以下)
            if current == goal_idx:
                return self._reconstruct_path(came_from, current, start_pos, goal_pos, to_world)

            current_g = g_score[current]

            for dx, dy, dz, move_dist_unit in self._NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                # 物理座標での衝突判定
                world_pos = to_world(neighbor)
                
                # CollisionChecker の境界チェックを含めた判定
                if not collision_checker.check_point(world_pos):
                    continue

                # コスト計算
                move_dist = move_dist_unit * res
                move_cost = self._compute_cost(world_pos, move_dist, collision_checker)
                tentative_g = current_g + move_cost

                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_new = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_new, counter, neighbor))
                counter += 1

        if iters >= max_iter:
            rospy.logwarn("AStarPlanner: iteration limit reached")
        else:
            rospy.logwarn("AStarPlanner: no path found")
        return []

    def _reconstruct_path(self, came_from, current, start_pos, goal_pos, to_world_func):
        """経路を復元し、始点・終点を元の正確な座標に置き換える."""
        path = [goal_pos]
        node = current
        while node in came_from:
            node = came_from[node]
            path.append(to_world_func(node))
        
        path[-1] = start_pos  # スタートを正確な座標へ
        path.reverse()
        rospy.loginfo("AStarPlanner: path found (%d waypoints)", len(path))
        return path
