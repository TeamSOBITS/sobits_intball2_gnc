#!/usr/bin/env python3
"""点群トピックから動的障害物を CollisionChecker に焼き込むモジュール."""
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import quaternion_matrix


def _transform_stamped_to_matrix(ts):
    """geometry_msgs/TransformStamped → 4x4 numpy 変換行列."""
    t = ts.transform.translation
    q = ts.transform.rotation
    mat = quaternion_matrix([q.x, q.y, q.z, q.w])
    mat[0, 3] = t.x
    mat[1, 3] = t.y
    mat[2, 3] = t.z
    return mat


class ObstacleManager:
    """PointCloud2 トピックを一度取得し、CollisionChecker の動的レイヤーに反映する."""

    def __init__(self, tf_buffer, topic="/depth/points"):
        self._tf_buffer = tf_buffer
        self._topic = topic
        self._min_points = rospy.get_param('/gnc/scan_min_points', 100)
        self._sampling_step = rospy.get_param('/gnc/scan_sampling_step', 5)
        self._scan_timeout = rospy.get_param('/gnc/scan_timeout', 5.0)
        self._wall_filter_dist = rospy.get_param('/gnc/wall_filter_dist', 0.20)
        self._last_captured_points = None
        rospy.loginfo("ObstacleManager params: min_points=%d, sampling_step=%d, scan_timeout=%.1f",
                      self._min_points, self._sampling_step, self._scan_timeout)

    @property
    def last_captured_points(self):
        """最後にキャプチャした ISS 座標点群 (N,3) ndarray。未取得時は None."""
        return self._last_captured_points

    def reset_captured_points(self):
        """保持している点群をリセットする（clear_dynamic() 前に呼ぶ）."""
        self._last_captured_points = None

    def capture_once(self, collision_checker, timeout=None, fov_deg=20.0):
        """点群を一枚取得し、動的レイヤーをクリアしてから焼き込む."""
        collision_checker.clear_dynamic()
        # 取得失敗時に古い点群状態を残さない
        self._last_captured_points = None
        transformed = self._fetch_and_transform(fov_deg=fov_deg, timeout=timeout or self._scan_timeout)
        if transformed is not None:
            self._last_captured_points = transformed[::self._sampling_step]
            rospy.loginfo(f"DEBUG: transformed_points_shape: {transformed.shape}, transformed_points_len: {len(transformed)}")
            self._bake(collision_checker, transformed)
        else:
            rospy.logwarn("ObstacleManager: capture_once failed, dynamic layer remains empty.")

    def capture_incremental(self, collision_checker, timeout=None, fov_deg=20.0):
        """点群を一枚取得し、既存の動的レイヤーに追加焼き込みする（クリアしない）."""
        transformed = self._fetch_and_transform(fov_deg=fov_deg, timeout=timeout or self._scan_timeout)
        if transformed is not None:
            self._last_captured_points = transformed[::self._sampling_step]
            rospy.loginfo(f"DEBUG: transformed_points_shape: {transformed.shape}, transformed_points_len: {len(transformed)}")
            self._bake(collision_checker, transformed)

    def capture_incremental_fov_clear(self, collision_checker, fov_deg=20.0, timeout=None):
        """点群を取得し、FOVコーン内の旧動的ボクセルをクリアしてから追加焼き込みする."""
        transformed = self._fetch_and_transform(fov_deg=fov_deg, timeout=timeout or self._scan_timeout)
        if transformed is None:
            return
        self._last_captured_points = transformed[::self._sampling_step]
        # ロボット現在位置をiss_bodyフレームで取得
        try:
            robot_tf = self._tf_buffer.lookup_transform(
                "iss_body", "body", rospy.Time(0), rospy.Duration(2.0)
            )
            t = robot_tf.transform.translation
            robot_pos = np.array([t.x, t.y, t.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("ObstacleManager: failed to get robot pos for FOV-clear (%s), using origin", e)
            robot_pos = np.zeros(3)
        collision_checker.clear_dynamic_in_fov(transformed, robot_pos, fov_deg=fov_deg)
        rospy.loginfo(f"DEBUG: transformed_points_shape: {transformed.shape}, transformed_points_len: {len(transformed)}")
        self._bake(collision_checker, transformed)

    def _fetch_and_transform(self, fov_deg=20.0, timeout=5.0):
        """点群取得・TF変換・フィルタを行い、iss_body座標の点群 (N,3) を返す。

        取得失敗・点数不足の場合は None を返す。
        """
        # 1. トピックフラッシュ: 古いバッファを破棄してから新鮮なデータを取得
        try:
            rospy.loginfo("ObstacleManager: Waiting for message on %s (timeout=%.1fs)...", self._topic, timeout)
            # 空読み: バッファに残っている古いメッセージを破棄
            rospy.wait_for_message(self._topic, PointCloud2, timeout=timeout)
            # 本読み: 新鮮なメッセージを取得
            msg = rospy.wait_for_message(self._topic, PointCloud2, timeout=timeout)
        except rospy.ROSException:
            rospy.logwarn("ObstacleManager: timeout waiting for %s, skipping dynamic obstacles", self._topic)
            return None
        except Exception as e:
            rospy.logwarn("ObstacleManager: failed to receive point cloud: %s", e)
            return None

        # 2. TF 変換行列を取得 (frame_id → iss_body)
        try:
            transform = self._tf_buffer.lookup_transform(
                "iss_body", msg.header.frame_id, rospy.Time(0), rospy.Duration(2.0)
            )
            tf_mat = _transform_stamped_to_matrix(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("ObstacleManager: TF transform failed (%s -> iss_body): %s",
                          msg.header.frame_id, e)
            return None

        # 3. 点群を numpy 配列に変換 (NaN は skip_nans=True で除去)
        raw_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                                    skip_nans=True)), dtype=np.float64)
        rospy.loginfo(f"DEBUG: raw_points_length: {len(raw_points)}, raw_points_shape: {raw_points.shape}")
        if len(raw_points) == 0:
            rospy.logwarn("ObstacleManager: no valid points after NaN filtering")
            return None

        # # [DEBUG FILTER] 中心10度（半角30度）以内のみを抽出
        # norms = np.linalg.norm(raw_points, axis=1)
        # valid_norms = norms > 0.001
        # raw_points = raw_points[valid_norms]
        # norms = norms[valid_norms]

        # # 光軸(0,0,1)との内積による角度判定
        # cos_limit = np.cos(np.radians(fov_deg))
        # # z / norm が光軸方向の一致度
        # fov_mask = (raw_points[:, 2] / norms) > cos_limit
        # raw_points = raw_points[fov_mask]

        if len(raw_points) == 0:
            rospy.logwarn("ObstacleManager: no points left after %.0f-deg FOV filtering", fov_deg)
            return None

        # 4. Inf を除去
        finite_mask = np.all(np.isfinite(raw_points), axis=1)
        raw_points = raw_points[finite_mask]
        if len(raw_points) == 0:
            rospy.logwarn("ObstacleManager: no valid points after Inf filtering")
            return None

        # 5. 変換行列を一括適用 (N,3) → (N,3)
        ones = np.ones((len(raw_points), 1))
        homo = np.hstack([raw_points, ones])  # (N, 4)
        transformed = (tf_mat @ homo.T).T[:, :3]  # (N, 3)

        # 6. 低点群数チェック: ノイズ回避のため閾値未満は書き込みスキップ
        if len(transformed) < self._min_points:
            rospy.logwarn("ObstacleManager: too few points (%d < %d) after FOV filter",
                          len(transformed), self._min_points)
            return None
        rospy.loginfo(f"DEBUG: ObstacleManager return transformed type {type(transformed)}")

        return transformed

    def _bake(self, collision_checker, transformed):
        """変換済み点群 (N,3) を動的レイヤーに焼き込む."""
        sampled_points = transformed[::self._sampling_step]
        edt_static = getattr(collision_checker, '_edt_static', None)
        wall_dist = self._wall_filter_dist
        n_written = 0
        n_filtered = 0
        for pt in sampled_points:
            idx = collision_checker.pos_to_idx(pt)
            if idx is None:
                continue
            if edt_static is not None and wall_dist > 0 and edt_static[idx] < wall_dist:
                n_filtered += 1
                continue
            collision_checker.set_dynamic_occupied(*idx)
            n_written += 1

        rospy.loginfo("ObstacleManager: captured %d points, wrote %d to dynamic layer (wall-filtered %d)",
                      len(transformed), n_written, n_filtered)