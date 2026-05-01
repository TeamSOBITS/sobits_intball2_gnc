#!/usr/bin/env python3
"""座標解決・最終姿勢調整・到達判定をカプセル化する PoseResolver クラス."""
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler
from .tf_frame_resolver import (
    get_body_in_iss, get_frame_in_iss, iss_wp_to_nav, wait_for_tf, _yaw_from_quaternion
)


class PoseResolver:
    """TF 経由での目標座標確定・最終姿勢調整・到達判定を担う."""

    def __init__(self, tf_buffer, executor):
        self._tf_buffer = tf_buffer
        self._executor = executor
        self._final_adjust_sample_interval = rospy.get_param('/gnc/final_adjust_sample_interval', 0.1)
        rospy.loginfo("PoseResolver params: final_adjust_sample_interval=%.2f",
                      self._final_adjust_sample_interval)

    def resolve(self, target_frame=None, goal=None, offset=(0, 0, 0)):
        """目標地点の ISS 座標を確定させる.

        Returns:
            (start_iss, goal_iss) のタプル
        """
        if not wait_for_tf(self._tf_buffer, timeout=5.0):
            raise RuntimeError("TF not available")

        start_iss = get_body_in_iss(self._tf_buffer)
        if target_frame:
            goal_iss = get_frame_in_iss(self._tf_buffer, target_frame, offset=tuple(offset))
            rospy.loginfo("PoseResolver: Goal (TF:%s): %.3f, %.3f, %.3f", target_frame, *goal_iss)
        else:
            goal_iss = np.array(goal) + np.array(offset)
            rospy.loginfo("PoseResolver: Goal (Coords): %.3f, %.3f, %.3f", *goal_iss)

        return start_iss, goal_iss

    def final_adjust(self, goal_iss, target_frame, yaw_offset):
        """ターゲット姿勢への精密な収束（時系列 TF サンプリング）."""
        if rospy.is_shutdown() or not target_frame:
            return
        rospy.loginfo("PoseResolver: Settle & Sampling for final pose...")
        rospy.sleep(1.0)

        # 複数回サンプリングしてTFの微細な揺れを平均化
        yaws_iss, yaws_nav, poses_nav = [], [], []
        for _ in range(5):
            try:
                t_b = self._tf_buffer.lookup_transform(
                    "iss_body", "body", rospy.Time(0), rospy.Duration(0.1))
                yaws_iss.append(_yaw_from_quaternion(
                    t_b.transform.rotation.x, t_b.transform.rotation.y,
                    t_b.transform.rotation.z, t_b.transform.rotation.w))
                n_p, n_y, _ = self._executor.get_nav_pose()
                if n_p is not None:
                    poses_nav.append(n_p)
                    yaws_nav.append(n_y)
                rospy.sleep(self._final_adjust_sample_interval)
            except Exception:
                continue

        if not yaws_iss or not poses_nav:
            return
        body_yaw_iss = sum(yaws_iss) / len(yaws_iss)
        nav_now = sum(poses_nav) / len(poses_nav)
        yaw_now = sum(yaws_nav) / len(yaws_nav)

        try:
            t_target = self._tf_buffer.lookup_transform(
                "iss_body", target_frame, rospy.Time(0), rospy.Duration(1.0))
            target_yaw_iss = _yaw_from_quaternion(
                t_target.transform.rotation.x, t_target.transform.rotation.y,
                t_target.transform.rotation.z, t_target.transform.rotation.w)
            # 現在のNav姿勢から目標Iss姿勢への差分を反映
            final_yaw_nav = (yaw_now + body_yaw_iss) - target_yaw_iss + yaw_offset
            final_q_nav = quaternion_from_euler(0, 0, final_yaw_nav)
            final_wp_nav, _ = iss_wp_to_nav(goal_iss, self._tf_buffer, nav_now, yaw_now, yaw_offset)
            self._executor.move_to(final_wp_nav, final_q_nav, timeout=20.0)
        except Exception as e:
            rospy.logerr("PoseResolver: final_adjust failed: %s", e)

    def check_result(self, goal_iss):
        """到達判定（ISS 座標の誤差を計測）."""
        error = np.linalg.norm(get_body_in_iss(self._tf_buffer) - goal_iss)
        rospy.loginfo("PoseResolver: Navigation Complete. Error: %.3f m", error)
        return error < 0.3
