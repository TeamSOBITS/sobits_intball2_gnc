#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import yaml
import tf2_ros
import rospkg
from geometry_msgs.msg import TransformStamped

class LocationBroadcaster:
    def __init__(self):
        rospy.init_node('location_broadcaster', anonymous=True)

        # 設定
        self.reference_frame = "iss_body"  # 親フレーム
        self.child_prefix = ""         # RVizで見やすくするための接頭辞
        
        # 保存先パスの特定
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('sobits_intball2_gnc')
            self.location_path = os.path.join(pkg_path, 'maps', 'iss_locations.yaml')
        except Exception:
            self.location_path = os.path.expanduser("~/catkin_ws/src/sobits_intball2_gnc/maps/iss_locations.yaml")

        self.br = tf2_ros.TransformBroadcaster()
        self.locations_data = {}
        self.last_mtime = 0  # ファイルの最終更新日時保持用

        rospy.loginfo(f"Broadcaster started. Watching: {self.location_path}")

    def load_yaml_if_updated(self):
        """ファイルが更新されていたら再読み込みする"""
        if not os.path.exists(self.location_path):
            return

        current_mtime = os.path.getmtime(self.location_path)
        if current_mtime > self.last_mtime:
            try:
                with open(self.location_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and "location_pose" in data:
                        self.locations_data = data["location_pose"] or {}
                        self.last_mtime = current_mtime
                        rospy.loginfo(f"YAML reloaded. {len(self.locations_data)} locations found.")
            except Exception as e:
                rospy.logwarn(f"Failed to reload YAML: {e}")

    def publish_transforms(self):
        """メモリ内の全地点をTFとして配信"""
        now = rospy.Time.now()
        
        for name, pose in self.locations_data.items():
            t = TransformStamped()
            
            # ヘッダー情報
            t.header.stamp = now
            t.header.frame_id = self.reference_frame
            t.child_frame_id = self.child_prefix + name
            
            # 位置 (Translation)
            t.transform.translation.x = pose["translation"]["x"]
            t.transform.translation.y = pose["translation"]["y"]
            t.transform.translation.z = pose["translation"]["z"]
            
            # 姿勢 (Rotation/Quaternion)
            t.transform.rotation.x = pose["rotation"]["x"]
            t.transform.rotation.y = pose["rotation"]["y"]
            t.transform.rotation.z = pose["rotation"]["z"]
            t.transform.rotation.w = pose["rotation"]["w"]
            
            self.br.sendTransform(t)

    def run(self):
        rate = rospy.Rate(10) # 10Hzで配信
        while not rospy.is_shutdown():
            self.load_yaml_if_updated()
            self.publish_transforms()
            rate.sleep()

if __name__ == '__main__':
    try:
        broadcaster = LocationBroadcaster()
        broadcaster.run()
    except rospy.ROSInterruptException:
        pass
