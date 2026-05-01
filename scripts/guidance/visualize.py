#!/usr/bin/env python3
"""経路の RViz 可視化ユーティリティ（リソース最適化版）."""
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# モジュールレベルで Publisher インスタンスを保持（重複生成を防止）
_publishers = {}

def publish_path(waypoints, topic="/planned_path", frame_id="map", latch=True):
    """ウェイポイントリストを nav_msgs/Path として publish する.

    Args:
        waypoints: [(x,y,z), ...] のリスト、または None/空リスト
        topic: publish 先トピック名
        frame_id: Path の座標フレーム
        latch: True なら latched publisher（後から接続しても最新経路が見える）
    Returns:
        None
    """
    global _publishers

    # 1. Publisher の取得または新規作成（トピックごとに1つだけ保持）
    if topic not in _publishers:
        # 初回呼び出し時のみ Publisher を作成
        _publishers[topic] = rospy.Publisher(topic, Path, queue_size=1, latch=latch)
    
    pub = _publishers[topic]

    # 2. メッセージの構築
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id

    # ウェイポイントが存在する場合のみ poses を追加
    if waypoints:
        for wp in waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

    # 3. Publish
    try:
        pub.publish(msg)
        if waypoints:
            rospy.loginfo("Published path (%d waypoints) to %s", len(waypoints), topic)
        else:
            rospy.loginfo("Cleared path on topic %s", topic)
    except Exception as e:
        rospy.logwarn("Failed to publish path to %s: %s", topic, e)

def clear_path(topic, frame_id="map"):
    """指定したトピックのパス表示を空（クリア）にする."""
    publish_path([], topic=topic, frame_id=frame_id)