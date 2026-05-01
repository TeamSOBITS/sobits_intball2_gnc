#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNC - Control Module: Smooth Action Executor
回転→並進の 2 フェーズを実行するが、中間 WP は緩い閾値で高速通過する。
コントローラ（type 40）は combined goal で姿勢を正しく処理しないため、
Phase 1（その場回転）が必須。
"""
import math
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from .action_handler import ActionExecutor


class SmoothActionExecutor(ActionExecutor):

    def move_to(self, pos, quat, timeout=60.0, is_waypoint=False):
        """
        1. 前回の目標位置でその場回転 → 緩い閾値で完了判定
        2. 並進を送信 → 収束を待つ

        is_waypoint=True:  中間WP用の緩い閾値（回転10deg / 位置10cm）
        is_waypoint=False: 最終WP用の厳密閾値（親クラスの _is_reached）
        """
        target_q = np.array(quat)
        _, _, target_yaw = euler_from_quaternion(target_q)

        # --- Phase 1: 回転（その場で姿勢だけ合わせる） ---
        rot_pos = self._current_target_pos
        if rot_pos is None:
            rot_pos, _, _ = self.get_nav_pose()

        if rot_pos is not None:
            self.update_target(rot_pos, quat)

            rot_tol = math.radians(10.0) if is_waypoint else math.radians(3.0)
            start_rot = rospy.Time.now()
            rate = rospy.Rate(10)

            while not rospy.is_shutdown():
                if (rospy.Time.now() - start_rot).to_sec() > 30.0:
                    rospy.logwarn("SmoothActionExecutor: Rotation timed out")
                    break

                _, curr_yaw, _ = self.get_nav_pose()
                if curr_yaw is not None:
                    yaw_err = target_yaw - curr_yaw
                    while yaw_err > math.pi: yaw_err -= 2 * math.pi
                    while yaw_err < -math.pi: yaw_err += 2 * math.pi
                    if abs(yaw_err) < rot_tol:
                        break

                rate.sleep()

        # --- Phase 2: 並進 ---
        self.update_target(pos, quat)

        target_pos = self._current_target_pos
        target_q = self._current_target_q
        _, _, target_yaw = euler_from_quaternion(target_q)

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("SmoothActionExecutor: Move timed out!")
                self.cancel_all_goals()
                return False

            curr_pos, curr_yaw, curr_q = self.get_nav_pose()
            if curr_pos is None:
                rate.sleep()
                continue

            if is_waypoint:
                dist_err = np.linalg.norm(target_pos - curr_pos)
                yaw_err = target_yaw - curr_yaw
                while yaw_err > math.pi: yaw_err -= 2 * math.pi
                while yaw_err < -math.pi: yaw_err += 2 * math.pi
                if dist_err < 0.10 and abs(yaw_err) < math.radians(10.0):
                    rospy.loginfo("SmoothActionExecutor: WP passed (dist: %.3fm, yaw: %.1fdeg)",
                                  dist_err, math.degrees(yaw_err))
                    return True
            else:
                reached, d_err, y_err = self._is_reached(
                    curr_pos, curr_yaw, curr_q, target_pos, target_yaw, target_q)
                if reached:
                    rospy.loginfo("SmoothActionExecutor: Tolerance reached (dist: %.3fm, yaw: %.1fdeg)",
                                  d_err, math.degrees(y_err))
                    break

            rate.sleep()

        if self.client.get_state() == 1:  # ACTIVE
            self.cancel_all_goals()

        return True
