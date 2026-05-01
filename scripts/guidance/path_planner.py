#!/usr/bin/env python3
"""経路計画責務をカプセル化する PathPlanner クラス."""
import math
import numpy as np
import rospy
from gnc_defaults import GNC_DEFAULTS
from .collision_checker import CollisionChecker
from .astar_planner import AStarPlanner
from .safety_astar_planner import SafetyAwareAStarPlanner
from .smoother import shortcut, push_from_walls
from .visualize import publish_path

_FILTER_MAP = {
    'push_from_walls': push_from_walls,
    'shortcut': shortcut,
}


def insert_virtual_waypoints(path, scan_interval=1.0):
    """スムージング済みパスに仮想WPを挿入する."""
    if not path:
        return []
    result = [(path[0], False)]
    for i in range(len(path) - 1):
        seg_start = np.asarray(path[i], dtype=float)
        seg_end = np.asarray(path[i + 1], dtype=float)
        dist = np.linalg.norm(seg_end - seg_start)
        if dist > scan_interval:
            n_sub = int(math.ceil(dist / scan_interval))
            for k in range(1, n_sub):
                t = float(k) / n_sub
                virtual_pt = seg_start + t * (seg_end - seg_start)
                result.append((virtual_pt, True))
        result.append((path[i + 1], False))
    return result


class PathPlanner:
    """経路計画（CollisionChecker 初期化・A* 実行・クリアランス緩和・脱出点探索）をカプセル化する."""

    _BBOX_BUFFER = 0.05

    def __init__(self, robot_radius=None, min_clearance=None, bbox_pad=None,
                 grid_resolution=None, push_step=None):
        self._robot_radius = robot_radius if robot_radius is not None else rospy.get_param('/gnc/robot_radius', GNC_DEFAULTS['robot_radius'])
        self._min_clearance = min_clearance if min_clearance is not None else rospy.get_param('/gnc/min_clearance', GNC_DEFAULTS['min_clearance'])
        self._bbox_pad = bbox_pad if bbox_pad is not None else rospy.get_param('/gnc/bbox_pad', GNC_DEFAULTS['bbox_pad'])
        self._bbox_extra_min_x = rospy.get_param('/gnc/bbox_extra_min_x', GNC_DEFAULTS['bbox_extra_min_x'])
        self._bbox_extra_max_x = rospy.get_param('/gnc/bbox_extra_max_x', GNC_DEFAULTS['bbox_extra_max_x'])
        self._bbox_extra_min_y = rospy.get_param('/gnc/bbox_extra_min_y', GNC_DEFAULTS['bbox_extra_min_y'])
        self._bbox_extra_max_y = rospy.get_param('/gnc/bbox_extra_max_y', GNC_DEFAULTS['bbox_extra_max_y'])
        self._bbox_extra_min_z = rospy.get_param('/gnc/bbox_extra_min_z', GNC_DEFAULTS['bbox_extra_min_z'])
        self._bbox_extra_max_z = rospy.get_param('/gnc/bbox_extra_max_z', GNC_DEFAULTS['bbox_extra_max_z'])
        self._grid_resolution = grid_resolution if grid_resolution is not None else rospy.get_param('/gnc/grid_resolution', GNC_DEFAULTS['grid_resolution'])
        self._push_step = push_step if push_step is not None else rospy.get_param('/gnc/push_step', GNC_DEFAULTS['push_step'])
        self._use_potential = rospy.get_param('/gnc/use_potential_astar', False)
        self._use_dual_edt = rospy.get_param('/gnc/use_dual_edt', False)
        self._recheck_enabled = rospy.get_param('/gnc/dynamic_clearance_recheck_enabled', GNC_DEFAULTS['dynamic_clearance_recheck_enabled'])
        self._recheck_min_distance = rospy.get_param('/gnc/dynamic_clearance_min_distance', GNC_DEFAULTS['dynamic_clearance_min_distance'])
        self._recheck_max_retries = rospy.get_param('/gnc/dynamic_clearance_recheck_max_retries', GNC_DEFAULTS['dynamic_clearance_recheck_max_retries'])
        self._cached_static_cc = None
        self._cached_octomap_hash = None
        self._dynamic_dirty = True

        # フィルタチェーン設定
        filter_names = rospy.get_param('/gnc/path_filters', ['push_from_walls', 'shortcut'])
        self._filters = []
        for name in filter_names:
            fn = _FILTER_MAP.get(name)
            if fn is None:
                rospy.logwarn("PathPlanner: unknown filter '%s', skipping", name)
                continue
            self._filters.append(fn)
        self._shortcut_margin = rospy.get_param('/gnc/shortcut_margin', 0.0)

        self._cached_cc = None
        rospy.loginfo("PathPlanner params: robot_radius=%.2f, min_clearance=%.2f, bbox_pad=%.2f, "
                  "bbox_extra=[-x:%.2f,+x:%.2f,-y:%.2f,+y:%.2f,-z:%.2f,+z:%.2f], "
                  "grid_res=%.3f, push_step=%.3f, filters=%s, shortcut_margin=%.3f",
                      self._robot_radius, self._min_clearance, self._bbox_pad,
                  self._bbox_extra_min_x, self._bbox_extra_max_x,
                  self._bbox_extra_min_y, self._bbox_extra_max_y,
                  self._bbox_extra_min_z, self._bbox_extra_max_z,
                      self._grid_resolution, self._push_step,
                      [f.__name__ for f in self._filters], self._shortcut_margin)
        rospy.loginfo(
            "PathPlanner recheck params: enabled=%s, min_distance=%.3f, max_retries=%d",
            self._recheck_enabled,
            self._recheck_min_distance,
            self._recheck_max_retries,
        )

    @property
    def current_cc(self):
        """キャッシュされた CollisionChecker を返す（未初期化時は None）."""
        return self._cached_cc

    def plan(self, start_iss, goal_iss, clearance, dynamic_clearance=0.05,
             scan=False, reuse_cc=False, obstacle_manager=None, scan_fov_deg=20.0,
             debug_pub=None, fallback_points=None, fallback_fov_deg=20.0):
        """衝突回避を考慮した経路計画.

        Args:
            start_iss: ISS 座標系での出発点
            goal_iss: ISS 座標系での目標点
            clearance: 静的障害物からのクリアランス [m]
            dynamic_clearance: 動的障害物の追加マージン [m]
            scan: True の場合、ここで点群を取得する
            reuse_cc: True の場合、BBOX 範囲内なら既存 CC を再利用する
            obstacle_manager: スキャン用の ObstacleManager（scan=True 時に使用）
            scan_fov_deg: スキャン FOV 半角 [deg]
            debug_pub: デバッグ可視化用パブリッシャ

        Returns:
            スムージング済みパス (list of np.array) or None
        """
        safety_margin = max(0, clearance - self._robot_radius)

        bbox_min = np.minimum(start_iss, goal_iss) - self._bbox_pad
        bbox_max = np.maximum(start_iss, goal_iss) + self._bbox_pad
        bbox_min -= np.array([self._bbox_extra_min_x, self._bbox_extra_min_y, self._bbox_extra_min_z], dtype=float)
        bbox_max += np.array([self._bbox_extra_max_x, self._bbox_extra_max_y, self._bbox_extra_max_z], dtype=float)

        cc = None
        cc_is_new = False
        # キャッシュの再利用判定
        if (reuse_cc and self._cached_cc is not None
                and np.all(bbox_min >= self._cached_cc._bbox_min - self._BBOX_BUFFER)
                and np.all(bbox_max <= self._cached_cc._bbox_max + self._BBOX_BUFFER)):
            cc = self._cached_cc
        else:
            cc = CollisionChecker(
                robot_radius=self._robot_radius,
                safety_margin=safety_margin,
                dynamic_safety_margin=dynamic_clearance,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                unknown_as_free=True
            )
            self._cached_cc = cc
            cc_is_new = True
            if fallback_points is not None:
                self._apply_fallback_points(cc, fallback_points)

        # 明示的に scan=True の場合のみここで撮る
        if scan and obstacle_manager is not None:
            obstacle_manager.capture_once(cc)
            if obstacle_manager.last_captured_points is None:
                rospy.logwarn("PathPlanner: scan requested but no fresh point cloud captured; planning with static map fallback.")
            if debug_pub:
                cc.publish_debug_cloud(debug_pub)

        effective_start = start_iss
        effective_goal = goal_iss
        escape_path = []
        approach_path = []

        # スタート地点の安全確認と脱出点計算
        if not cc.check_point(start_iss):
            rospy.logwarn("PathPlanner: Start is in collision, trying escape point.")
            escape_pt = self._find_escape_point(start_iss, cc)
            if escape_pt is None:
                rospy.logerr("PathPlanner: Start is in collision and no escape point found.")
                return None
            effective_start = escape_pt
            escape_path = [start_iss, escape_pt]

        if not cc.check_point(goal_iss):
            rospy.logwarn("PathPlanner: Goal is in collision, trying approach point.")
            approach_pt = self._find_escape_point(goal_iss, cc)
            if approach_pt is None:
                rospy.logerr("PathPlanner: Goal is in collision and no approach point found.")
                return None
            effective_goal = approach_pt
            approach_path = [approach_pt, goal_iss]

        if self._use_potential:
            if self._use_dual_edt:
                if cc_is_new or cc._edt_static is None:
                    cc.compute_edt_static()
                cc.compute_edt_dynamic()
            else:
                cc.compute_edt()
            planner = SafetyAwareAStarPlanner(grid_resolution=self._grid_resolution)
        else:
            planner = AStarPlanner(grid_resolution=self._grid_resolution)
        raw_path = planner.plan(effective_start, effective_goal, cc)
        if not raw_path:
            rospy.logwarn(
                "PathPlanner: planning failed (empty path). start=%s goal=%s bbox_min=%s bbox_max=%s clearance=%.3f dynamic_clearance=%.3f",
                np.round(effective_start, 3).tolist(),
                np.round(effective_goal, 3).tolist(),
                np.round(bbox_min, 3).tolist(),
                np.round(bbox_max, 3).tolist(),
                clearance,
                dynamic_clearance,
            )
            return None

        if escape_path:
            raw_path = escape_path[:-1] + raw_path
        if approach_path:
            raw_path = raw_path + approach_path[1:]

        smooth_path = self._apply_filters(raw_path, cc)

        publish_path(raw_path, topic="/planned_path_raw")
        publish_path(smooth_path, topic="/planned_path")
        return smooth_path

    def plan_with_retry(self, start_iss, goal_iss, clearance, dynamic_clearance=0.05,
                        scan=False, reuse_cc=False, obstacle_manager=None,
                        scan_fov_deg=20.0, debug_pub=None,
                        fallback_points=None, fallback_fov_deg=20.0):
        """クリアランスを段階的に緩和しながら計画を試みる.

        Returns:
            スムージング済みパス (list of np.array) or None
        """
        try_clearance = clearance
        attempts = 0
        while try_clearance >= self._min_clearance:
            attempts += 1
            smooth_path = self.plan(start_iss, goal_iss, try_clearance,
                                    dynamic_clearance=dynamic_clearance,
                                    scan=scan, reuse_cc=reuse_cc,
                                    obstacle_manager=obstacle_manager,
                                    scan_fov_deg=scan_fov_deg,
                                    debug_pub=debug_pub,
                                    fallback_points=fallback_points,
                                    fallback_fov_deg=fallback_fov_deg)
            if smooth_path:
                if not self._is_recheck_applicable():
                    return smooth_path

                recheck_path = smooth_path
                recheck_attempts = 0
                while True:
                    d_min = self._min_dynamic_distance(recheck_path, self._cached_cc)
                    if d_min >= self._recheck_min_distance:
                        if recheck_attempts > 0:
                            rospy.loginfo(
                                "PathPlanner: dynamic recheck passed after %d retries (d_min=%.3f, threshold=%.3f)",
                                recheck_attempts,
                                d_min,
                                self._recheck_min_distance,
                            )
                        return recheck_path

                    if recheck_attempts >= self._recheck_max_retries:
                        rospy.logwarn(
                            "PathPlanner: dynamic recheck max retries reached (%d). d_min=%.3f < threshold=%.3f; using last path",
                            self._recheck_max_retries,
                            d_min,
                            self._recheck_min_distance,
                        )
                        return recheck_path

                    recheck_attempts += 1
                    rospy.logwarn(
                        "PathPlanner: dynamic recheck failed (d_min=%.3f < threshold=%.3f), retrying plan (%d/%d)",
                        d_min,
                        self._recheck_min_distance,
                        recheck_attempts,
                        self._recheck_max_retries,
                    )
                    retry_path = self.plan(
                        start_iss,
                        goal_iss,
                        try_clearance,
                        dynamic_clearance=dynamic_clearance,
                        scan=scan,
                        reuse_cc=reuse_cc,
                        obstacle_manager=obstacle_manager,
                        scan_fov_deg=scan_fov_deg,
                        debug_pub=debug_pub,
                        fallback_points=fallback_points,
                        fallback_fov_deg=fallback_fov_deg,
                    )
                    if not retry_path:
                        rospy.logwarn(
                            "PathPlanner: replan failed during dynamic recheck at clearance=%.3f",
                            try_clearance,
                        )
                        break
                    recheck_path = retry_path
            rospy.logwarn(
                "PathPlanner: plan_with_retry failed at clearance=%.3f (dynamic_clearance=%.3f)",
                try_clearance,
                dynamic_clearance,
            )
            try_clearance -= 0.05
        rospy.logerr(
            "PathPlanner: plan_with_retry exhausted (%d attempts). start=%s goal=%s initial_clearance=%.3f min_clearance=%.3f dynamic_clearance=%.3f",
            attempts,
            np.round(start_iss, 3).tolist(),
            np.round(goal_iss, 3).tolist(),
            clearance,
            self._min_clearance,
            dynamic_clearance,
        )
        return None

    @staticmethod
    def _apply_fallback_points(cc, points):
        """点群 (N,3) ndarray を CC の動的レイヤーに書き込む（フィルタ済み前提）."""
        n_written = 0
        for pt in points:
            idx = cc.pos_to_idx(pt)
            if idx is not None:
                cc.set_dynamic_occupied(*idx)
                n_written += 1
        rospy.loginfo("PathPlanner: applied %d/%d fallback points to new CC", n_written, len(points))

    def _is_recheck_applicable(self):
        """動的クリアランス再検証が有効に機能する構成かを返す."""
        if not self._recheck_enabled:
            return False
        if self._use_potential and self._use_dual_edt:
            return True
        rospy.logwarn_once(
            "PathPlanner: dynamic clearance recheck is enabled but requires /gnc/use_potential_astar=true and /gnc/use_dual_edt=true; fallback to legacy behavior."
        )
        return False

    def _min_dynamic_distance(self, path, cc):
        """パス上の最小動的 EDT 距離を返す."""
        if not path or cc is None or cc._edt_dynamic is None:
            return float('inf')

        min_dist = float('inf')
        sample_step = max(self._grid_resolution * 0.5, 1e-6)
        points = [np.asarray(p, dtype=float) for p in path]

        for pt in points:
            min_dist = min(min_dist, cc.get_distance_edt_dynamic(pt))

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            if seg_len <= 1e-9:
                continue
            n_seg = int(math.ceil(seg_len / sample_step))
            for k in range(1, n_seg):
                t = float(k) / float(n_seg)
                pt = p0 + t * seg
                min_dist = min(min_dist, cc.get_distance_edt_dynamic(pt))

        return float(min_dist)

    def _apply_filters(self, path, cc):
        """経路後処理パイプライン."""
        if not path:
            return []
        from . import apply_path_filters
        return apply_path_filters(
            path, cc,
            filters=self._filters,
            push_step=self._push_step,
            shortcut_margin=self._shortcut_margin,
        )

    @staticmethod
    def _find_escape_point(pos, cc, max_radius=0.5):
        """障害物に埋まった際の脱出点を探索."""
        res = cc.resolution
        directions = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    directions.append(np.array([dx, dy, dz]) / np.linalg.norm([dx, dy, dz]))
        for step in range(1, int(max_radius / res) + 1):
            for d in directions:
                candidate = pos + d * (step * res)
                if cc.check_point(candidate) and cc.check_line(pos, candidate):
                    return candidate
        rospy.logwarn(
            "PathPlanner: no escape point found around pos=%s within radius=%.3f",
            np.round(pos, 3).tolist(),
            max_radius,
        )
        return None
