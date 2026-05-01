#!/usr/bin/env python3
"""
TF フレーム名 → 各種座標系での位置を解決するユーティリティ。
ISS Z-down 定義を Nav Z-up 座標系へ「絶対座標変換」としてマッピングする。
"""
import math
import numpy as np
import rospy
import tf2_ros
from ib2_msgs.msg import Navigation
from tf.transformations import quaternion_from_euler

# グローバルな TF キャッシュ
_tf_buffer = None
_tf_listener = None

def _initialize_tf():
    global _tf_buffer, _tf_listener
    if _tf_buffer is None:
        _tf_buffer = tf2_ros.Buffer()
        _tf_listener = tf2_ros.TransformListener(_tf_buffer)
        rospy.sleep(1.0) # TF の蓄積待ち
    return _tf_buffer

def _yaw_from_quaternion(qx, qy, qz, qw):
    """quaternion → yaw (rad)"""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)

def wait_for_tf(tf_buffer, timeout=10.0):
    """iss_body→body の TF が利用可能になるまで待機する."""
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            tf_buffer.lookup_transform("iss_body", "body",
                                       rospy.Time(0), rospy.Duration(1))
            return True
        except Exception as e:
            rospy.logwarn("wait_for_tf: %s", e)
        rospy.sleep(0.5)
    rospy.logerr("wait_for_tf timed out after %.1fs", timeout)
    return False

def get_body_in_iss(tf_buffer):
    """TF: iss_body → body の translationを取得."""
    t = tf_buffer.lookup_transform("iss_body", "body", rospy.Time(0), rospy.Duration(5))
    tr = t.transform.translation
    return np.array([tr.x, tr.y, tr.z])

def get_frame_in_iss(tf_buffer, frame_name, offset=(0, 0, 0)):
    """TF lookup で任意フレームの iss_body 座標を取得."""
    t = tf_buffer.lookup_transform("iss_body", frame_name, rospy.Time(0), rospy.Duration(5))
    tr = t.transform.translation
    return np.array([tr.x + offset[0], tr.y + offset[1], tr.z + offset[2]])

def iss_wp_to_nav(wp_iss, tf_buffer, nav_pos, nav_yaw, yaw_offset=0.0):
    """
    ISSフレームのWPを、現在のNavフレームの絶対座標に変換する。
    相対的な差分ではなく、フレーム間の回転オフセットを算出して適用する。
    """
    # 1. 現在のフレーム間の回転オフセットを算出
    # 成功した相対ロジック (target = nav_y - (goal_iss - body_iss_y)) から導出
    # これは NavフレームとISSフレームの「方位の対応関係」を固定する計算
    t = tf_buffer.lookup_transform("iss_body", "body", rospy.Time(0), rospy.Duration(5.0))
    body_yaw_iss = _yaw_from_quaternion(t.transform.rotation.x, t.transform.rotation.y, 
                                        t.transform.rotation.z, t.transform.rotation.w)
    
    # ISSフレームの0度が、Navフレームで何度に相当するかを決定する定数
    # (揺れているISSに合わせて、この瞬間の値で固定)
    frame_map_offset = nav_yaw + body_yaw_iss

    # 2. ISS内での現在地と移動ベクトル
    body_iss = get_body_in_iss(tf_buffer)
    dx_iss = wp_iss[0] - body_iss[0]
    dy_iss = wp_iss[1] - body_iss[1]
    dz_iss = wp_iss[2] - body_iss[2]
    
    dist_xy = math.sqrt(dx_iss**2 + dy_iss**2)
    goal_angle_iss = math.atan2(dy_iss, dx_iss)

    # 3. 【絶対マッピング】ISSの方位をNavの方位へ直接変換
    # 鏡面反転しているため、オフセットから引く
    target_nav_yaw = frame_map_offset - goal_angle_iss
    
    # 4. Navフレームでの変位ベクトル（絶対方位から算出）
    nav_dx = dist_xy * math.cos(target_nav_yaw)
    nav_dy = dist_xy * math.sin(target_nav_yaw)
    nav_dz = -dz_iss # Z反転 (ISS Z-down -> Nav Z-up)
    
    target_nav_pos = nav_pos + np.array([nav_dx, nav_dy, nav_dz])

    # 5. 姿勢決定 (絶対方位 + 各種補正)
    final_yaw = target_nav_yaw + yaw_offset
    # 実験結果に基づき、上に進む(nav_dz > 0)ときに負のPitchを送る
    final_pitch = math.atan2(-nav_dz, max(dist_xy, 1e-6))

    q = quaternion_from_euler(0, final_pitch, final_yaw)
    return target_nav_pos, q

def resolve_frame(frame_name, offset=(0, 0, 0), timeout=5.0):
    """
    TFフレーム名からNav座標系での絶対位置・姿勢を直接解決する。
    """
    buffer = _initialize_tf()
    try:
        nav_msg = rospy.wait_for_message("/sensor_fusion/navigation", Navigation, timeout=2.0)
        nav_p = np.array([nav_msg.pose.pose.position.x, 
                          nav_msg.pose.pose.position.y, 
                          nav_msg.pose.pose.position.z])
        nav_y = _yaw_from_quaternion(nav_msg.pose.pose.orientation.x, 
                                     nav_msg.pose.pose.orientation.y, 
                                     nav_msg.pose.pose.orientation.z, 
                                     nav_msg.pose.pose.orientation.w)

        # ターゲットのISS座標を取得
        wp_iss = get_frame_in_iss(buffer, frame_name, offset)
        
        # 共通の絶対座標変換ロジックを呼び出す
        return (*iss_wp_to_nav(wp_iss, buffer, nav_p, nav_y)[0], 
                *iss_wp_to_nav(wp_iss, buffer, nav_p, nav_y)[1])

    except Exception as e:
        rospy.logerr(f"Error in resolve_frame: {e}")
        raise