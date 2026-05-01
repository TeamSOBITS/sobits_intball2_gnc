#!/usr/bin/env python3
"""経路計画から移動実行までのフローを管理する指揮者モジュール.

責務の実装は以下の専門クラスに委譲する:
  - PathPlanner: 経路計画（guidance パッケージ）
  - PoseResolver: 座標解決・最終姿勢調整・到達判定（navigation パッケージ）
  - TrajectoryFollower: パス追従・事前回転・スキャン再計画（control パッケージ）
"""
import math
import numpy as np
import rospy
from typing import cast
from tf.transformations import quaternion_from_euler
from navigation.tf_frame_resolver import iss_wp_to_nav, _yaw_from_quaternion
from navigation.pose_resolver import PoseResolver
from guidance.path_planner import PathPlanner, insert_virtual_waypoints
from control.trajectory_follower import TrajectoryFollower
from gnc_defaults import GNC_DEFAULTS


class Navigator:
    def __init__(self, tf_buffer, executor, obstacle_manager=None,
                 path_planner=None, pose_resolver=None, trajectory_follower=None):
        self.tf_buffer = tf_buffer
        self.executor = executor
        self.obstacle_manager = obstacle_manager
        self.clearance = cast(float, rospy.get_param('/gnc/static_clearance', GNC_DEFAULTS['static_clearance']))
        self.dynamic_clearance = cast(float, rospy.get_param('/gnc/dynamic_clearance', GNC_DEFAULTS['dynamic_clearance']))
        self.scan_interval = cast(float, rospy.get_param('/gnc/scan_interval', GNC_DEFAULTS['scan_interval']))
        self.scan_fov_deg = cast(float, rospy.get_param('/gnc/scan_fov_deg', GNC_DEFAULTS['scan_fov_deg']))
        self._initial_scan_wait = cast(float, rospy.get_param('/gnc/initial_scan_wait', GNC_DEFAULTS['initial_scan_wait']))
        self._dr_yaw_tol_deg = cast(float, rospy.get_param('/gnc/direct_rotate_yaw_tol_deg', GNC_DEFAULTS['direct_rotate_yaw_tol_deg']))
        self._dr_pitch_tol_deg = cast(float, rospy.get_param('/gnc/direct_rotate_pitch_tol_deg', GNC_DEFAULTS['direct_rotate_pitch_tol_deg']))
        rospy.loginfo("Navigator params: clearance=%.2f, dynamic_clearance=%.2f, scan_interval=%.2f, scan_fov_deg=%.1f",
                      self.clearance, self.dynamic_clearance, self.scan_interval, self.scan_fov_deg)
        self._path_planner = path_planner or PathPlanner()
        self._pose_resolver = pose_resolver or PoseResolver(tf_buffer, executor)
        self._trajectory_follower = trajectory_follower or TrajectoryFollower(
            tf_buffer, executor, self._path_planner,
            obstacle_manager=obstacle_manager)

    def _pre_rotate_toward_goal(self, goal_iss, yaw_offset):
        """初回観測前にゴール直線方向へ機首を向ける。"""
        nav_now, nav_yaw, _ = self.executor.get_nav_pose()
        if nav_now is None:
            rospy.logwarn("Navigator: cannot pre-rotate (nav pose unavailable).")
            return
        goal_nav, q_goal = iss_wp_to_nav(goal_iss, self.tf_buffer, nav_now, nav_yaw, yaw_offset)
        if np.linalg.norm(goal_nav - nav_now) < 0.05:
            rospy.loginfo("Navigator: Skip pre-rotate (goal already near current position).")
            return
        rospy.loginfo("Navigator: Pre-rotate toward goal direction before first scan.")
        self.executor.move_to(nav_now, q_goal, timeout=30.0)

    def navigate(self, target_frame=None, goal=None, offset=(0, 0, 0)):
        """ターゲットへ経路計画を行い移動する."""
        nav_mode = rospy.get_param('/gnc/navigation_mode', GNC_DEFAULTS['navigation_mode'])
        if nav_mode == 'direct':
            return self._navigate_direct(target_frame, goal, offset)
        elif nav_mode == 'direct_rotate':
            return self._navigate_direct_rotate(target_frame, goal, offset)
        elif nav_mode == 'full':
            pass  # 以降の通常フローへ
        else:
            rospy.logerr("Navigator: Unknown navigation_mode '%s'. Valid: full / direct / direct_rotate", nav_mode)
            return False

        active_clearance = float(self.clearance)
        active_dynamic = float(self.dynamic_clearance)

        yaw_offset = 0.0
        try:
            start_iss, goal_iss = self._pose_resolver.resolve(target_frame, goal, offset)

            # 1. 初回観測先行: ゴール方向へ回転し、点群取得後に計画する
            if self.obstacle_manager is not None:
                self._pre_rotate_toward_goal(goal_iss, yaw_offset)
                rospy.loginfo("Navigator: Performing first scan before initial planning.")
                rospy.sleep(self._initial_scan_wait)
                scanned_path = self._path_planner.plan_with_retry(
                    start_iss, goal_iss, active_clearance,
                    dynamic_clearance=active_dynamic,
                    scan=True,
                    obstacle_manager=self.obstacle_manager,
                    scan_fov_deg=self.scan_fov_deg,
                )

                if scanned_path:
                    smooth_path = scanned_path
                elif self.obstacle_manager.last_captured_points is None:
                    rospy.logwarn("Navigator: Initial scan unavailable, fallback to static-only initial planning.")
                    smooth_path = self._path_planner.plan_with_retry(
                        start_iss, goal_iss, active_clearance,
                        dynamic_clearance=active_dynamic, scan=False)
                else:
                    rospy.logerr("Navigator: Initial planning failed even with first scan data.")
                    return False
            else:
                # ObstacleManager が無い構成では従来通り静的計画
                smooth_path = self._path_planner.plan_with_retry(
                    start_iss, goal_iss, active_clearance,
                    dynamic_clearance=active_dynamic, scan=False)

            if not smooth_path:
                rospy.logerr("Navigator: Initial planning failed.")
                return False

            cc = self._path_planner.current_cc
            if cc:
                cc.publish_debug_cloud(self._trajectory_follower._debug_pub)

            # 4. パス追従
            annotated_path = insert_virtual_waypoints(smooth_path, self.scan_interval)

            if not self._trajectory_follower.execute(annotated_path, yaw_offset, goal_iss, active_clearance, active_dynamic):
                return False

            rospy.loginfo("Navigator: full mode completed without final_adjust (policy: skip final orientation alignment).")
            return self._pose_resolver.check_result(goal_iss)
        except Exception as e:
            rospy.logerr("Navigator: Navigation failed: %s", e)
            import traceback
            rospy.logerr(traceback.format_exc())
            return False

    def _navigate_direct_rotate(self, target_frame=None, goal=None, offset=(0, 0, 0)):
        """直接TF追従 + 終端姿勢合わせモード（Pitch/Yawを反映）."""
        rospy.logwarn("Navigator: [DIRECT_ROTATE MODE] 経路計画・障害物回避・事前回転をスキップし、"
                      "Yaw/Pitch軸の姿勢合わせを実行します。")
        try:
            _, goal_iss = self._pose_resolver.resolve(target_frame, goal, offset)
            if goal_iss is None:
                rospy.logerr("Navigator: [DIRECT_ROTATE MODE] 目標座標の解決に失敗しました。")
                return False

            nav_pos, nav_yaw, _ = self.executor.get_nav_pose()
            if nav_pos is None:
                rospy.logerr("Navigator: [DIRECT_ROTATE MODE] 現在位置の取得に失敗しました。")
                return False

            # 初期値として方向ベースのクォータニオンを設定
            goal_nav, q_fallback = iss_wp_to_nav(goal_iss, self.tf_buffer, nav_pos, nav_yaw, 0.0)
            target_q = q_fallback
            target_yaw_iss = None
            target_pitch_iss = None

            if target_frame:
                try:
                    # 1. ISS座標系におけるターゲットの姿勢（回転）をルックアップ
                    # ここでターゲットの向きを丸ごと取得する
                    t_target = self.tf_buffer.lookup_transform("iss_body", target_frame, rospy.Time(0), rospy.Duration.from_sec(1.0))

                    # 2. 現在の機体の iss_body 内での向きを取得（補正用）
                    t_body = self.tf_buffer.lookup_transform("iss_body", "body", rospy.Time(0), rospy.Duration.from_sec(1.0))

                    # 3. ターゲットのクォータニオンからRoll, Pitch, Yawを取り出す
                    from tf.transformations import euler_from_quaternion, quaternion_from_euler
                    q = t_target.transform.rotation
                    target_roll_iss, target_pitch_iss, target_yaw_iss = euler_from_quaternion([q.x, q.y, q.z, q.w])

                    # 4. 機体の現在の iss_body 内での Yaw を取得
                    qb = t_body.transform.rotation
                    _, _, body_yaw_iss = euler_from_quaternion([qb.x, qb.y, qb.z, qb.w])

                    # 5. ナビゲーション座標系への変換計算
                    # Yawは相対関係を維持し、Roll/Pitchはターゲットの値をそのまま（あるいはISS基準で）適用
                    final_yaw_nav = (nav_yaw + body_yaw_iss) - target_yaw_iss

                    # ターゲットのPitchをそのまま反映させる
                    # (注: 競技環境の定義により、符号の反転が必要な場合はここで調整)
                    target_q = quaternion_from_euler(0, -target_pitch_iss, final_yaw_nav)

                except Exception as e:
                    rospy.logwarn("Navigator: [DIRECT_ROTATE MODE] failed to compute Full-Pose alignment (%s), fallback to direction-based.", e)
                    target_yaw_iss = None
                    target_pitch_iss = None
            else:
                rospy.logwarn("Navigator: [DIRECT_ROTATE MODE] target_frame is not specified; using fallback posture.")

            if not self.executor.move_to(goal_nav, target_q):
                return False

            if target_yaw_iss is not None:
                return self._verify_and_correct_pose(target_yaw_iss, target_pitch_iss)

            return True
        except Exception as e:
            rospy.logerr("Navigator: [DIRECT_ROTATE MODE] Navigation failed: %s", e)
            import traceback
            rospy.logerr(traceback.format_exc())
            return False

    def _verify_and_correct_pose(self, target_yaw_iss, target_pitch_iss):
        """direct_rotate 到着後にYaw/Pitch誤差を検証し、超過時に再調整する."""
        try:
            nav_pos, nav_yaw, curr_q = self.executor.get_nav_pose()
            if nav_pos is None:
                rospy.logwarn("Navigator: pose verify skipped (nav pose unavailable).")
                return True

            t_body = self.tf_buffer.lookup_transform("iss_body", "body", rospy.Time(0), rospy.Duration(1.0))
            qb = t_body.transform.rotation
            body_yaw_iss = _yaw_from_quaternion(qb.x, qb.y, qb.z, qb.w)

            from tf.transformations import euler_from_quaternion
            _, actual_pitch, _ = euler_from_quaternion(list(curr_q))

            expected_yaw_nav = (nav_yaw + body_yaw_iss) - target_yaw_iss
            expected_pitch_nav = -target_pitch_iss

            def _norm_rad(a):
                while a > math.pi:
                    a -= 2 * math.pi
                while a < -math.pi:
                    a += 2 * math.pi
                return a

            yaw_err_deg = math.degrees(abs(_norm_rad(nav_yaw - expected_yaw_nav)))
            pitch_err_deg = math.degrees(abs(_norm_rad(actual_pitch - expected_pitch_nav)))

            rospy.loginfo(
                "Navigator: pose verify yaw_err=%.1fdeg (tol=%.1f) pitch_err=%.1fdeg (tol=%.1f)",
                yaw_err_deg, self._dr_yaw_tol_deg, pitch_err_deg, self._dr_pitch_tol_deg,
            )

            if yaw_err_deg <= self._dr_yaw_tol_deg and pitch_err_deg <= self._dr_pitch_tol_deg:
                return True

            rospy.logwarn(
                "Navigator: pose verify FAILED (yaw_err=%.1fdeg pitch_err=%.1fdeg). Issuing corrective move.",
                yaw_err_deg, pitch_err_deg,
            )
            corrected_q = quaternion_from_euler(0, -target_pitch_iss, expected_yaw_nav)
            result = self.executor.move_to(nav_pos, corrected_q)
            if not result:
                rospy.logwarn("Navigator: corrective move also failed.")
            return result

        except Exception as e:
            rospy.logwarn("Navigator: pose verify skipped due to exception: %s", e)
            return True

    def _navigate_direct(self, target_frame=None, goal=None, offset=(0, 0, 0)):
        """直接TF追従モード（障害物回避・経路計画なし・ドックエリア向け）."""
        rospy.logwarn("Navigator: [DIRECT MODE] 障害物回避・経路計画・事前回転をすべてスキップします。"
                      " 動的障害物がないことを確認してから使用してください。")
        try:
            _, goal_iss = self._pose_resolver.resolve(target_frame, goal, offset)
            if goal_iss is None:
                rospy.logerr("Navigator: [DIRECT MODE] 目標座標の解決に失敗しました。")
                return False
            nav_pos, nav_yaw, curr_q = self.executor.get_nav_pose()
            if nav_pos is None:
                rospy.logerr("Navigator: [DIRECT MODE] 現在位置の取得に失敗しました。")
                return False
            goal_nav, _ = iss_wp_to_nav(goal_iss, self.tf_buffer, nav_pos, nav_yaw, 0.0)
            return self.executor.move_to(goal_nav, curr_q)
        except Exception as e:
            rospy.logerr("Navigator: [DIRECT MODE] Navigation failed: %s", e)
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
