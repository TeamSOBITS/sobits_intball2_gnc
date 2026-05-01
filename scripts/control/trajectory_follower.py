#!/usr/bin/env python3
"""パス追従・事前回転・スキャン再計画をカプセル化する TrajectoryFollower クラス."""
import numpy as np
import rospy
from navigation.tf_frame_resolver import get_body_in_iss, iss_wp_to_nav
from guidance import check_remaining_path
from guidance.path_planner import insert_virtual_waypoints
from sensor_msgs.msg import PointCloud2
from .smooth_executor import SmoothActionExecutor
from visualization_msgs.msg import Marker
from gnc_defaults import GNC_DEFAULTS

class TrajectoryFollower:
    """パス追従（逐次/スムース/ストリーミング）・事前回転・スキャン再計画を担う."""

    def __init__(self, tf_buffer, executor, path_planner, obstacle_manager=None):
        self._tf_buffer = tf_buffer
        self._executor = executor
        self._path_planner = path_planner
        self._obstacle_manager = obstacle_manager
        self._default_clearance = rospy.get_param('/gnc/static_clearance', GNC_DEFAULTS['static_clearance'])
        self._default_dynamic_clearance = rospy.get_param('/gnc/dynamic_clearance', GNC_DEFAULTS['dynamic_clearance'])
        self._scan_interval = rospy.get_param('/gnc/scan_interval', GNC_DEFAULTS['scan_interval'])
        self._scan_fov_deg = rospy.get_param('/gnc/scan_fov_deg', GNC_DEFAULTS['scan_fov_deg'])
        self._min_clearance = rospy.get_param('/gnc/min_clearance', GNC_DEFAULTS['min_clearance'])
        self._max_replan_retries = rospy.get_param('/gnc/max_replan_retries', GNC_DEFAULTS['max_replan_retries'])
        self._stabilize_wait = rospy.get_param('/gnc/stabilize_wait', GNC_DEFAULTS['stabilize_wait'])
        self._dynamic_clear_mode = rospy.get_param('/gnc/dynamic_clear_mode', GNC_DEFAULTS['dynamic_clear_mode'])
        self._wp_move_timeout = rospy.get_param('/gnc/wp_move_timeout', GNC_DEFAULTS['wp_move_timeout'])
        self._has_moved = False
        self._post_replan_fresh = False
        self._debug_pub = rospy.Publisher("/debug/accumulated_dynamic_voxels", PointCloud2, queue_size=1, latch=True)
        self._marker_pub = rospy.Publisher("/debug/collision_marker", Marker, queue_size=1, latch=True)
        rospy.loginfo("TrajectoryFollower params: clearance=%.2f, dynamic_clearance=%.2f, scan_interval=%.2f, "
                      "min_clearance=%.2f, max_replan=%d, stabilize_wait=%.1f",
                      self._default_clearance, self._default_dynamic_clearance, self._scan_interval,
                      self._min_clearance, self._max_replan_retries, self._stabilize_wait)
        rospy.loginfo("TrajectoryFollower: wp_move_timeout=%.1f", self._wp_move_timeout)

    def publish_collision_marker(self, pos, collision_type):
        marker = Marker()
        marker.header.frame_id = "iss_body"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "collision"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1 # 10cmの球
        
        if collision_type == 1: # 静的衝突: 黒色 (Black)
            # R=0, G=0, B=0, A=1 (不透明)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 0.0, 0.0, 1.0)
        else: # 動的衝突: 黄色 (Yellow)
            # R=1, G=1, B=0, A=1 (不透明)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 1.0, 0.0, 1.0)
        marker.lifetime = rospy.Duration(5.0) # 5秒で消える
        self._marker_pub.publish(marker)
    
    def pre_rotate(self, smooth_path, yaw_offset):
        """最初の有効な WP 方向へ機首を向ける."""
        nav_now, yaw_now, _ = self._executor.get_nav_pose()
        if nav_now is None:
            return

        for wp in smooth_path:
            wp_nav, q_wp = iss_wp_to_nav(wp, self._tf_buffer, nav_now, yaw_now, yaw_offset)
            if np.linalg.norm(wp_nav - nav_now) > 0.05:
                rospy.loginfo("TrajectoryFollower: Pre-move rotation toward WP1.")
                self._executor.move_to(nav_now, q_wp, timeout=30.0)
                rospy.sleep(self._stabilize_wait)
                break

    def execute(self, annotated_path, yaw_offset, goal_iss=None, clearance=None, dynamic_clearance=None):
        """annotated_path を追従する。Executor 種別に応じてループを自動選択する."""

        if isinstance(self._executor, SmoothActionExecutor):
            return self._execute_smooth(annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance)

        return self._execute_standard(annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance)

    # ---- Standard (ActionExecutor) ----

    def _execute_standard(self, annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance):
        """ActionExecutor 用の Look-then-Move ループ."""
        i = 0
        replan_count = 0
        while i < len(annotated_path):
            if rospy.is_shutdown():
                return False
            wp_iss, is_virtual = annotated_path[i]
            nav_now, yaw_now, _ = self._executor.get_nav_pose()
            if nav_now is None:
                i += 1
                continue

            wp_nav, q_wp = iss_wp_to_nav(wp_iss, self._tf_buffer, nav_now, yaw_now, yaw_offset)
            dist = np.linalg.norm(wp_nav - nav_now)
            if dist < 0.05:
                i += 1
                continue

            # 1. 向きだけ更新（その場回転） ※全WP共通
            self._executor.update_target(nav_now, q_wp)
            while not self._executor.is_reached(nav_now, q_wp):
                if rospy.is_shutdown():
                    return False
                rospy.sleep(0.1)

            # 2. 前進前のスキャンと再計画（仮想・実WP共通）
            if self._obstacle_manager is not None:
                result = self._scan_and_replan(annotated_path, max(0, i - 1), is_virtual, goal_iss, clearance, dynamic_clearance)
                if isinstance(result, list):
                    replan_count += 1
                    if replan_count > self._max_replan_retries:
                        rospy.logerr("TrajectoryFollower: Max replan retries (%d) exceeded.", self._max_replan_retries)
                        return False
                    annotated_path = result
                    result2 = self._post_replan_verify(annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance, replan_count)
                    if result2 == "failed":
                        return False
                    if isinstance(result2, tuple):
                        annotated_path, replan_count = result2
                    i = 0
                    continue
                elif result == "failed":
                    return False

            # 3. 前進実行
            self._executor.update_target(wp_nav, q_wp)
            deadline = rospy.Time.now() + rospy.Duration(self._wp_move_timeout)
            while not self._executor.is_reached(wp_nav, q_wp):
                if rospy.is_shutdown():
                    return False
                if rospy.Time.now() > deadline:
                    rospy.logwarn("WP move timeout (%.1fs). Skipping waypoint %d.", self._wp_move_timeout, i)
                    break
                rospy.sleep(0.1)
            self._has_moved = True
            replan_count = 0  # 前進成功でリセット

            i += 1
        return True

    # ---- Smooth (SmoothActionExecutor) ----

    def _execute_smooth(self, annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance):
        """SmoothActionExecutor 用ループ."""
        i = 0
        replan_count = 0
        while i < len(annotated_path):
            if rospy.is_shutdown():
                return False
            wp_iss, is_virtual = annotated_path[i]
            nav_now, yaw_now, _ = self._executor.get_nav_pose()
            if nav_now is None:
                i += 1
                continue

            wp_nav, q_wp = iss_wp_to_nav(wp_iss, self._tf_buffer, nav_now, yaw_now, yaw_offset)
            if np.linalg.norm(wp_nav - nav_now) < 0.05:
                i += 1
                continue

            # SmoothExecutorではis_waypoint=Falseで確実に一度停止させる
            if not self._executor.move_to(wp_nav, q_wp, is_waypoint=False):
                return False
            self._has_moved = True  # 移動成功
            replan_count = 0  # 前進成功でリセット

            # 到着後のスキャン
            if i < len(annotated_path) - 1 and self._obstacle_manager is not None:
                result = self._scan_and_replan(annotated_path, i, is_virtual, goal_iss, clearance, dynamic_clearance)
                if isinstance(result, list):
                    replan_count += 1
                    if replan_count > self._max_replan_retries:
                        rospy.logerr("TrajectoryFollower: Max replan retries (%d) exceeded.", self._max_replan_retries)
                        return False
                    annotated_path = result
                    result2 = self._post_replan_verify(annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance, replan_count)
                    if result2 == "failed":
                        return False
                    if isinstance(result2, tuple):
                        annotated_path, replan_count = result2
                    i = 0
                    continue
                elif result == "failed":
                    return False
            i += 1
        return True

    # ---- 再計画後の安全確認（共通） ----

    def _post_replan_verify(self, annotated_path, yaw_offset, goal_iss, clearance, dynamic_clearance, replan_count):
        """再計画後: 回転→安定待ち→スキャン→衝突チェック→必要なら再度再計画."""
        smooth_path = [pos for pos, _ in annotated_path]
        self.pre_rotate(smooth_path, yaw_offset)
        rospy.loginfo("TrajectoryFollower: Post-replan scan after rotation.")
        rospy.sleep(self._stabilize_wait)
        cc = self._path_planner.current_cc
        self._obstacle_manager.capture_incremental(cc, fov_deg=self._scan_fov_deg)
        self._post_replan_fresh = True
        # 可視化更新
        if cc:
            cc.publish_debug_cloud(self._debug_pub)

        collision_idx = check_remaining_path(smooth_path, 0, cc)
        if collision_idx is None:
            return "clear"

        rospy.logwarn("TrajectoryFollower: Collision still detected after replan rotation (idx=%s), re-planning.", collision_idx)
        current_pos = get_body_in_iss(self._tf_buffer)
        rp_clearance = clearance or self._default_clearance
        new_path = None
        while rp_clearance >= self._min_clearance:
            new_path = self._path_planner.plan(current_pos, goal_iss, rp_clearance, dynamic_clearance, scan=False, reuse_cc=True,
                                               fallback_points=self._obstacle_manager.last_captured_points)
            if new_path:
                break
            rp_clearance -= 0.05
        if new_path:
            new_annotated = insert_virtual_waypoints(new_path, self._scan_interval)
            # --- デバッグ: 計画直後のパス先頭とCC状態 ---
            cc_now = self._path_planner.current_cc
            n_dyn_now = int(np.sum(cc_now._grid_dynamic != 0)) if cc_now is not None else -1
            rospy.logwarn("DEBUG post_replan plan done: dynamic_voxels=%d, new_path_len=%d, annotated_len=%d",
                          n_dyn_now, len(new_path), len(new_annotated))
            positions_new = [pos for pos, _ in new_annotated]
            for k in range(min(4, len(positions_new))):
                rospy.logwarn("DEBUG  new_annotated[%d]=%s  check_point=%s", k,
                              np.round(positions_new[k], 3).tolist(),
                              cc_now.check_point(positions_new[k]) if cc_now else "N/A")
            rospy.logwarn("DEBUG  cc object id=%d", id(cc_now))
            # -------------------------------------------
            replan_count += 1
            if replan_count > self._max_replan_retries:
                rospy.logerr("TrajectoryFollower: Max replan retries (%d) exceeded.", self._max_replan_retries)
                return "failed"
            return (new_annotated, replan_count)
        else:
            rospy.logerr("TrajectoryFollower: Post-rotate replan failed.")
            return "failed"

    # ---- スキャンと再計画 ----

    def _scan_and_replan(self, annotated_path, current_idx, is_virtual, goal_iss, clearance, dynamic_clearance):
        """WP到着・回転後の再計画判定 (安定待ちを追加)."""
        # 回転直後の揺れが収まるまで待機
        rospy.sleep(self._stabilize_wait)
        current_pos = get_body_in_iss(self._tf_buffer)

        cc = self._path_planner.current_cc
        # 領域チェック
        if cc is None or not (np.all(current_pos >= cc._bbox_min - 0.05)
                              and np.all(current_pos <= cc._bbox_max + 0.05)):
            new_path = self._path_planner.plan(current_pos, goal_iss, clearance, dynamic_clearance, scan=True, reuse_cc=False)
            return insert_virtual_waypoints(new_path, self._scan_interval) if new_path else "failed"

        # --- リセット判定 ---
        # 物理的な移動が発生しており、かつ直前にpost-replanスキャンを行っていない場合のみクリア
        if self._has_moved and not self._post_replan_fresh:
            rospy.loginfo("TrajectoryFollower: Move completed. Clearing dynamic layer for fresh scan.")
            self._obstacle_manager.reset_captured_points()
            cc.clear_dynamic()
            self._has_moved = False  # クリア実行によりフラグを消費
            _use_fov_clear = False
        else:
            rospy.loginfo("TrajectoryFollower: Skipping clear_dynamic (Post-replan or Rotation-only).")
            # _post_replan_verify がスキャン＆計画を済ませている場合は再スキャン不要
            if self._post_replan_fresh:
                rospy.loginfo("TrajectoryFollower: Skipping scan (post-replan-verify already scanned).")
                self._post_replan_fresh = False
                cc.publish_debug_cloud(self._debug_pub)
                positions = [pos for pos, _ in annotated_path]
                # --- デバッグ: CC状態とパス先頭WPを出力 ---
                n_dyn = int(np.sum(cc._grid_dynamic != 0)) if cc is not None else -1
                rospy.logwarn("DEBUG skip_scan: dynamic_voxels=%d, path_len=%d, current_idx=%d, cc_id=%d",
                              n_dyn, len(positions), current_idx, id(cc))
                for k in range(min(4, len(positions))):
                    rospy.logwarn("DEBUG  positions[%d]=%s  check_point=%s", k,
                                  np.round(positions[k], 3).tolist(),
                                  cc.check_point(positions[k]) if cc else "N/A")
                # -----------------------------------------
                collision_idx = check_remaining_path(positions, current_idx, cc)
                if collision_idx is None:
                    return "clear"
                # 衝突セグメントの詳細ログ
                p1 = positions[collision_idx]
                p2 = positions[collision_idx + 1] if collision_idx + 1 < len(positions) else None
                rospy.logwarn("DEBUG blocked at seg %d: [%s]->[%s]",
                              collision_idx,
                              np.round(p1, 3).tolist(),
                              np.round(p2, 3).tolist() if p2 is not None else "N/A")
                if p2 is not None:
                    # セグメント中間点も確認
                    mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
                    rospy.logwarn("DEBUG  mid=[%s] check_point=%s",
                                  np.round(mid, 3).tolist(), cc.check_point(mid))
                    _, c_pos = cc.check_collision_detailed(np.asarray(p1))
                    rospy.logwarn("DEBUG  p1 collision_detailed: pos=%s", c_pos)
                rospy.logwarn("TrajectoryFollower: Path still blocked after verify; proceeding to rescan.")
                # フォールスルー: 以降の通常スキャン+再計画ブロックを実行するため return しない
                _use_fov_clear = (self._dynamic_clear_mode == 'fov')
            else:
                # post-replanフラグはここで消費（次のWP到着時にはクリアを許可するため）
                self._post_replan_fresh = False
                _use_fov_clear = (self._dynamic_clear_mode == 'fov')

        rospy.loginfo("TrajectoryFollower: Scanning for obstacles before forward move...")
        if _use_fov_clear:
            rospy.loginfo("TrajectoryFollower: Using FOV-clear scan (dynamic_clear_mode=fov).")
            self._obstacle_manager.capture_incremental_fov_clear(cc, fov_deg=self._scan_fov_deg)
        else:
            self._obstacle_manager.capture_incremental(cc, fov_deg=self._scan_fov_deg)
        # スキャン後の蓄積状態を可視化
        cc.publish_debug_cloud(self._debug_pub)

        positions = [pos for pos, _ in annotated_path]
        collision_idx = check_remaining_path(positions, current_idx, cc)
        if collision_idx is None:
            return "clear"

        # 衝突検知時のログ出力
        rospy.logwarn("TrajectoryFollower: Collision detected at path index %s! Re-planning...", collision_idx)
        
        if collision_idx is not None:
            # 衝突したセグメントを check_line と同じ刻みで走査して衝突点を特定
            p1 = np.asarray(positions[collision_idx], dtype=float)
            p2 = np.asarray(positions[collision_idx + 1], dtype=float)
            diff = p2 - p1
            dist = np.linalg.norm(diff)
            step = cc.resolution * 0.5
            n_steps = max(int(np.ceil(dist / step)), 1) if dist > 1e-9 else 1
            c_type, c_pos = 0, None
            for k in range(n_steps + 1):
                pt = p1 + (k / n_steps) * diff
                c_type, c_pos = cc.check_collision_detailed(pt)
                if c_type != 0:
                    break

            if c_type != 0:
                rospy.logwarn("Collision Found! Type: %s at %s",
                              "STATIC" if c_type == 1 else "DYNAMIC", c_pos)
                self.publish_collision_marker(c_pos, c_type)
        

        active_clearance = clearance or self._default_clearance
        while active_clearance >= self._min_clearance:
            # reuse_cc=True で、今撮ったばかりの点群を使って計算
            new_path = self._path_planner.plan(current_pos, goal_iss, active_clearance, dynamic_clearance, scan=False, reuse_cc=True,
                                               fallback_points=self._obstacle_manager.last_captured_points)
            if new_path:
                rospy.loginfo("TrajectoryFollower: New path found with clearance %.2f", active_clearance)
                return insert_virtual_waypoints(new_path, self._scan_interval)
            active_clearance -= 0.05

        # フォールバック: clearance 緩和でも通れない → もう一度クリア＆再スキャン
        rospy.logwarn("TrajectoryFollower: No path with accumulated points. Fallback: clear dynamic and rescan.")
        self._obstacle_manager.reset_captured_points()
        cc.clear_dynamic()
        self._obstacle_manager.capture_incremental(cc, fov_deg=self._scan_fov_deg)
        # リセット後の可視化更新
        cc.publish_debug_cloud(self._debug_pub)

        active_clearance = clearance or self._default_clearance
        while active_clearance >= self._min_clearance:
            new_path = self._path_planner.plan(current_pos, goal_iss, active_clearance, dynamic_clearance, scan=False, reuse_cc=True,
                                               fallback_points=self._obstacle_manager.last_captured_points)
            if new_path:
                rospy.loginfo("TrajectoryFollower: Fallback path found with clearance %.2f", active_clearance)
                return insert_virtual_waypoints(new_path, self._scan_interval)
            active_clearance -= 0.05

        rospy.logerr("TrajectoryFollower: No safe path found even with fallback.")
        return "failed"
