#!/usr/bin/env python3
"""
GNC - Control Module: Action Executor
ロボットへの移動指令の送信と、実行中の監視を担当する。
"""
import rospy
import actionlib
import numpy as np
import math
from ib2_msgs.msg import CtlCommandAction, CtlCommandGoal, Navigation
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from .base_executor import BaseExecutor

class ActionExecutor(BaseExecutor):
    def __init__(self):
        # Action Client の初期化
        self.client = actionlib.SimpleActionClient("/ctl/command", CtlCommandAction)
        
        # 二重送信防止および状態管理用
        self._current_target_pos = None
        self._current_target_q = None

        rospy.loginfo("ActionExecutor: Waiting for /ctl/command server...")
        if not self.client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.logerr("ActionExecutor: Control server not found!")

    def get_nav_pose(self):
        """現在の Nav 座標系の位置、Yaw角(float)、およびクォータニオンを取得."""
        try:
            msg = rospy.wait_for_message("/sensor_fusion/navigation", Navigation, timeout=2.0)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            
            # クォータニオンを保持
            curr_q = np.array([o.x, o.y, o.z, o.w])
            
            # クォータニオンから Yaw (Z軸回転) を計算
            siny_cosp = 2 * (o.w * o.z + o.x * o.y)
            cosy_cosp = 1 - 2 * (o.y * o.y + o.z * o.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return np.array([p.x, p.y, p.z]), yaw, curr_q
        except rospy.ROSException:
            rospy.logerr("ActionExecutor: Failed to get navigation message.")
            return None, None, None

    def _is_reached(self, curr_pos, curr_yaw, curr_q, target_pos, target_yaw, target_q):
        """位置、Yaw、および3D姿勢(Quaternion)の収束を判定 (特異点対策版)."""

        # 1. 位置誤差の計算
        dist_err = np.linalg.norm(target_pos - curr_pos)
        
        # 2. Yaw誤差の計算 (正規化)
        yaw_err = target_yaw - curr_yaw
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi
        
        # 3. 3D姿勢誤差 (Quaternionの内積: 1.0に近いほど一致)
        q_dot = abs(np.dot(curr_q, target_q)) if curr_q is not None and target_q is not None else 0.0
        
        # --- 4. 特異点判定 (Pitch > 85度) ---
        # ターゲットの姿勢からPitch成分を確認
        _, p_rad, _ = euler_from_quaternion(target_q)
        is_near_singularity = abs(math.degrees(p_rad)) > 85.0

        # --- 5. 判定ロジック ---
        # 位置は常に 5cm 以内であることを要求
        pos_ok = dist_err < 0.05
        
        if is_near_singularity:
            # 特異点付近では数学的にYaw/3D姿勢の厳密な一致が困難なため、
            # 位置さえ合っていれば「到達」とみなしてループを抜ける。
            angle_ok = True
            rospy.logwarn_throttle(5.0, "ActionExecutor: Near singularity (Pitch: %.1f), skipping angle check.", math.degrees(p_rad))
        else:
            # 通常時は Yaw 3度以内、かつクォータニオン内積 0.999 以上を要求
            angle_ok = abs(yaw_err) < math.radians(3.0) and q_dot > 0.999
        
        if pos_ok and angle_ok:
            return True, dist_err, yaw_err
            
        return False, dist_err, yaw_err

    def update_target(self, pos, quat):
        """
        現在の目標地点を更新する。Navigatorから移管された特異点対策を含む。
        """
        # --- Navigator から移管された特異点 (Singularity) 対策 ---
        _, p_rad, y_rad = euler_from_quaternion(quat)
        p_deg = math.degrees(p_rad)
        
        final_quat = quat
        if abs(p_deg) > 85.0:
            # ピッチが ±85度 を超える場合、Yawを現在の値に固定して再構成
            curr_pos, curr_yaw, _ = self.get_nav_pose()
            if curr_yaw is not None:
                rospy.logwarn_throttle(2.0, "ActionExecutor: Near Singularity (Pitch: %.1f). Fixing Yaw to current.", p_deg)
                final_quat = quaternion_from_euler(0, p_rad, curr_yaw)

        # 重複命令の抑制 (前回の命令とほぼ同じならスキップ)
        if self._current_target_pos is not None and self._current_target_q is not None:
            d_pos = np.linalg.norm(np.array(pos) - self._current_target_pos)
            d_q = abs(np.dot(final_quat, self._current_target_q))
            if d_pos < 0.01 and d_q > 0.9999:
                return

        self._current_target_pos = np.array(pos)
        self._current_target_q = np.array(final_quat)
        self.send_goal_async(pos, final_quat)

    def send_goal_async(self, pos, quat):
        """非同期で移動指令を送信する."""
        goal = CtlCommandGoal()
        goal.target.header.frame_id = "" 
        goal.type.type = 40 
        goal.target.pose.position.x, goal.target.pose.position.y, goal.target.pose.position.z = pos
        goal.target.pose.orientation.x, goal.target.pose.orientation.y, \
        goal.target.pose.orientation.z, goal.target.pose.orientation.w = quat
        
        rospy.loginfo("ActionExecutor: Sending goal to (%.3f, %.3f, %.3f)", pos[0], pos[1], pos[2])
        self.client.send_goal(goal)

    def cancel_all_goals(self):
        """現在のすべての移動指令をキャンセルする."""
        self.client.cancel_goal()
        self._current_target_pos = None
        self._current_target_q = None

    def move_to(self, pos, quat, timeout=60.0):
        """
        指定した座標と姿勢へ移動指令を送る（ブロッキング）。
        """
        # update_target を通すことで特異点対策を適用
        self.update_target(pos, quat)
        
        target_pos = self._current_target_pos
        target_q = self._current_target_q
        _, _, target_yaw = euler_from_quaternion(target_q)
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(5)
        
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            
            if (now - start_time).to_sec() > timeout:
                rospy.logwarn("ActionExecutor: Move timed out!")
                self.cancel_all_goals()
                return False

            state = self.client.get_state()
            if state in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
                rospy.logerr("ActionExecutor: Goal failed (state: %d)", state)
                return False

            curr_pos, curr_yaw, curr_q = self.get_nav_pose()
            if curr_pos is not None and curr_yaw is not None:
                reached, d_err, y_err = self._is_reached(curr_pos, curr_yaw, curr_q, target_pos, target_yaw, target_q)
                if reached:
                    rospy.loginfo("ActionExecutor: Tolerance reached (dist: %.3fm, yaw: %.1fdeg)", 
                                  d_err, math.degrees(y_err))
                    break

            rate.sleep()
            
        if self.client.get_state() == actionlib.GoalStatus.ACTIVE:
             self.cancel_all_goals()
             
        return True

    def is_reached(self, target_pos, target_quat):
        """
        BaseExecutorのインターフェース実装。
        """
        curr_pos, curr_yaw, curr_q = self.get_nav_pose()
        if curr_pos is None or curr_yaw is None:
            return False
            
        target_q = np.array(target_quat)
        _, _, target_yaw = euler_from_quaternion(target_quat)
        
        reached, _, _ = self._is_reached(curr_pos, curr_yaw, curr_q, target_pos, target_yaw, target_q)
        return reached

# --------------- 単体動作確認用 ---------------
if __name__ == "__main__":
    rospy.init_node("test_executor", anonymous=True)
    
    executor = ActionExecutor()
    
    print("Fetching current pose for test...")
    curr_pos, curr_yaw, curr_q = executor.get_nav_pose()
    
    if curr_pos is not None:
        test_goal = curr_pos + np.array([0, 0, 0.2])
        q = quaternion_from_euler(0, 0, curr_yaw)
        
        success = executor.move_to(test_goal, q, timeout=20.0)
        
        if success:
            rospy.loginfo("Test Result: SUCCESS")
        else:
            rospy.logerr("Test Result: FAILED")