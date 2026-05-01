#!/usr/bin/env python3
"""EDT ベースのポテンシャル場 A* プランナー."""
import math
import rospy
from .astar_planner import AStarPlanner


class SafetyAwareAStarPlanner(AStarPlanner):
    """壁からの距離をコストに算入し、広い道を優先的に選ぶ A* プランナー.

    EDT (Euclidean Distance Transform) による O(1) 距離参照を使用。
    使用前に collision_checker.compute_edt() が呼ばれている必要がある。
    """

    def __init__(self, grid_resolution=None, bbox_margin=1.0, weight=None):
        super().__init__(grid_resolution, bbox_margin)
        self.weight = weight if weight is not None else rospy.get_param('/gnc/safety_weight', 1.5)
        self._safety_threshold = rospy.get_param('/gnc/safety_threshold', 0.30)
        self._dynamic_threshold = rospy.get_param('/gnc/dynamic_threshold', 0.30)
        self._use_dual_edt = rospy.get_param('/gnc/use_dual_edt', False)
        self._static_weight = rospy.get_param('/gnc/static_weight', 1.5)
        self._dynamic_weight = rospy.get_param('/gnc/dynamic_weight', 5.0)
        rospy.loginfo("SafetyAwareAStarPlanner: weight=%.2f, threshold=%.3f, use_dual_edt=%s",
                      self.weight, self._safety_threshold, self._use_dual_edt)
        if self._use_dual_edt:
            rospy.loginfo(
                "SafetyAwareAStarPlanner: dual EDT enabled — static_threshold=%.3f, dynamic_threshold=%.3f, static_weight=%.2f, dynamic_weight=%.2f",
                self._safety_threshold, self._dynamic_threshold, self._static_weight, self._dynamic_weight
            )

    def _compute_cost(self, world_pos, move_dist, collision_checker):
        """壁に近いほど指数関数的にペナルティを加算する（EDT O(1) 参照）."""
        if self._use_dual_edt:
            dist_s = collision_checker.get_distance_edt_static(world_pos)
            dist_d = collision_checker.get_distance_edt_dynamic(world_pos)
            penalty = 0.0
            if dist_s < self._safety_threshold:
                penalty += math.exp(self._safety_threshold - dist_s) * self._static_weight
            if dist_d < self._dynamic_threshold:
                penalty += math.exp(self._dynamic_threshold - dist_d) * self._dynamic_weight
            return move_dist + penalty
        else:
            dist = collision_checker.get_distance_edt(world_pos)
            penalty = 0.0
            if dist < self._safety_threshold:
                penalty = math.exp(self._safety_threshold - dist) * self.weight
            return move_dist + penalty
