#!/usr/bin/env python3
"""OctoMap ベースの衝突判定モジュール."""
import os
import tempfile
from typing import Dict, Optional, Tuple, cast

import numpy as np
import rospy
import octomap
from scipy.ndimage import distance_transform_edt
from octomap_msgs.msg import Octomap
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2
_OCCUPIED_CORE = 2    # 追加: 動的レイヤーの中心点
_OCCUPIED_MARGIN = 3  # 追加: 動的レイヤーの膨張分

def _octomap_msg_to_octree(msg):
    """octomap_msgs/Octomap → octomap.OcTree."""
    raw = np.array(msg.data, dtype=np.int8).view(np.uint8).tobytes()
    header = (
        "# Octomap OcTree binary file\n"
        "id {}\n"
        "size 99999999\n"
        "res {}\n"
        "data\n"
    ).format(msg.id, msg.resolution).encode()
    with tempfile.NamedTemporaryFile(suffix=".bt", delete=False) as f:
        f.write(header + raw)
        tmp_path = f.name
    tree = octomap.OcTree(msg.resolution)
    tree.readBinary(tmp_path.encode())
    os.unlink(tmp_path)
    return tree

class CollisionChecker:
    """OctoMap を購読し、点・線分の衝突判定を提供する."""

    def __init__(self, topic="/octomap_binary", timeout=10.0,
                 robot_radius=0.10, safety_margin=0.05,
                 dynamic_safety_margin=0.05, # 追加: 動的用のマージン
                 bbox_min=None, bbox_max=None, unknown_as_free=False,
                 octree=None, resolution=None):
        if octree is not None:
            self._tree = octree
            self._resolution = resolution if resolution is not None else octree.getResolution()
        else:
            rospy.loginfo("CollisionChecker: waiting for %s ...", topic)
            msg = cast(Octomap, rospy.wait_for_message(topic, Octomap, timeout=timeout))
            self._tree = _octomap_msg_to_octree(msg)
            self._resolution = msg.resolution
        
        # 静的用マージン
        self._margin = robot_radius + safety_margin
        # 動的用マージン (セル数に変換して保持)
        self._dynamic_margin_cells = int(np.ceil(dynamic_safety_margin / self._resolution))

        if bbox_min is None or bbox_max is None:
            raise ValueError("bbox_min and bbox_max are required")
        self._bbox_min = np.asarray(bbox_min, dtype=float)
        self._bbox_max = np.asarray(bbox_max, dtype=float)
        self._unknown_as_free = unknown_as_free
        self._oob_total = 0
        self._oob_counts: Dict[str, int] = {
            'min_x': 0,
            'max_x': 0,
            'min_y': 0,
            'max_y': 0,
            'min_z': 0,
            'max_z': 0,
        }
        self._build_grid()
        
    def check_collision_detailed(self, pos):
        """衝突の種類と座標を詳細に返す (0:なし, 1:静的, 2:動的)"""
        idx = self.pos_to_idx(pos)
        if idx is None:
            return 1, pos # 範囲外は静的扱い
        
        # 静的衝突 (壁とその膨らみ)
        if self._grid_static[idx] != _FREE:
            return 1, self.idx_to_pos(idx)
        
        # 動的衝突 (点群の蓄積とその膨らみ)
        if self._grid_dynamic[idx] != _FREE:
            return 2, self.idx_to_pos(idx)
            
        return 0, None

    def _build_grid(self):
        """OcTree から 3D occupancy grid を構築し、occupied のみ margin 分膨張."""
        res = self._resolution
        pad = self._margin + res
        bb_min = self._bbox_min - pad
        bb_max = self._bbox_max + pad

        self._origin = bb_min
        grid_size = np.ceil((bb_max - bb_min) / res).astype(int) + 1

        grid = np.zeros(grid_size, dtype=np.uint8)

        for it in self._tree.begin_leafs_bbx(bb_min, bb_max):
            center = np.array([it.getX(), it.getY(), it.getZ()])
            size = it.getSize()
            half = size / 2.0
            imin = np.maximum(np.floor((center - half - self._origin) / res).astype(int), 0)
            imax = np.minimum(np.ceil((center + half - self._origin) / res).astype(int), grid_size - 1)
            val = _OCCUPIED if self._tree.isNodeOccupied(it) else _FREE
            grid[imin[0]:imax[0]+1, imin[1]:imax[1]+1, imin[2]:imax[2]+1] = val

        # 膨張前の生グリッドを保存（EDT 計算用）
        self._grid_static_raw = grid.copy()

        margin_cells = int(np.ceil(self._margin / res))
        if margin_cells > 0:
            occ_idx = np.argwhere(grid == _OCCUPIED)
            rospy.loginfo("CollisionChecker: dilating %d occupied cells (margin_cells=%d)",
                          len(occ_idx), margin_cells)
            offsets = []
            for di in range(-margin_cells, margin_cells + 1):
                for dj in range(-margin_cells, margin_cells + 1):
                    for dk in range(-margin_cells, margin_cells + 1):
                        if di*di + dj*dj + dk*dk <= margin_cells * margin_cells:
                            offsets.append((di, dj, dk))
            for di, dj, dk in offsets:
                shifted = occ_idx + np.array([di, dj, dk])
                valid = np.all((shifted >= 0) & (shifted < grid_size), axis=1)
                shifted = shifted[valid]
                grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = _OCCUPIED

        if self._unknown_as_free:
            n_converted = np.sum(grid == _UNKNOWN)
            grid[grid == _UNKNOWN] = _FREE
            rospy.loginfo("CollisionChecker: unknown_as_free=True, converted %d unknown -> free",
                          n_converted)

        self._grid_static = grid
        self._grid_dynamic = np.full(grid_size, _FREE, dtype=np.uint8)
        self._margin_cells = margin_cells
        self._edt_static = None
        self._edt_dynamic = None

        n_free = np.sum(grid == _FREE)
        n_occ = np.sum(grid == _OCCUPIED)
        n_unk = np.sum(grid == _UNKNOWN)
        rospy.loginfo(
            "CollisionChecker: grid=%s free=%d(%.1f%%) occ=%d unk=%d",
            grid_size, n_free, 100.0 * n_free / grid.size, n_occ, n_unk,
        )

    @property
    def resolution(self):
        return self._resolution

    @property
    def margin(self):
        return self._margin

    def pos_to_idx(self, pos) -> Optional[Tuple[int, int, int]]:
        pos_arr = np.asarray(pos, dtype=float)
        idx = np.floor((pos_arr - self._origin) / self._resolution).astype(int)
        shape = self._grid_static.shape
        out_sides = []
        for d in range(3):
            if idx[d] < 0 or idx[d] >= shape[d]:
                if idx[d] < 0:
                    out_sides.append(('min_x', 'min_y', 'min_z')[d])
                else:
                    out_sides.append(('max_x', 'max_y', 'max_z')[d])

        if out_sides:
            for side in out_sides:
                self._oob_counts[side] += 1
            self._oob_total += 1

            nonzero_counts = {k: v for k, v in self._oob_counts.items() if v > 0}
            dominant_side = max(nonzero_counts.items(), key=lambda kv: kv[1])[0]
            hint_param = {
                'min_x': 'bbox_extra_min_x',
                'max_x': 'bbox_extra_max_x',
                'min_y': 'bbox_extra_min_y',
                'max_y': 'bbox_extra_max_y',
                'min_z': 'bbox_extra_min_z',
                'max_z': 'bbox_extra_max_z',
            }[dominant_side]

            rospy.logwarn_throttle(
                1.0,
                "CollisionChecker: pos out of bbox. pos=%s idx=%s shape=%s origin=%s res=%.3f sides=%s oob_total=%d oob_counts=%s hint=increase /gnc/%s",
                np.round(pos_arr, 3).tolist(),
                idx.tolist(),
                list(shape),
                np.round(self._origin, 3).tolist(),
                self._resolution,
                out_sides,
                self._oob_total,
                nonzero_counts,
                hint_param,
            )
            return None
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def idx_to_pos(self, idx):
        return self._origin + (np.array(idx, dtype=float) + 0.5) * self._resolution

    def set_dynamic_occupied(self, ix, iy, iz):
        """動的レイヤーの指定セルを occupied に設定し、マージン分膨張させる."""
        shape = self._grid_dynamic.shape
        m = self._dynamic_margin_cells
        
        if m <= 0:
            if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                self._grid_dynamic[ix, iy, iz] = _OCCUPIED_CORE
            return

        # 動的マージン分の矩形膨張
        imin = np.maximum([ix-m, iy-m, iz-m], 0)
        imax = np.minimum([ix+m, iy+m, iz+m], np.array(shape) - 1)
        
        # まず範囲全体を MARGIN(3) で塗り、中心を CORE(2) で上書き
        self._grid_dynamic[imin[0]:imax[0]+1, imin[1]:imax[1]+1, imin[2]:imax[2]+1] = _OCCUPIED_MARGIN
        self._grid_dynamic[ix, iy, iz] = _OCCUPIED_CORE

    def clear_dynamic(self):
        self._grid_dynamic[:] = _FREE

    def clear_dynamic_in_fov(self, points, robot_pos, fov_deg=20.0, margin_factor=1.2):
        """FOVコーン内に入る動的占有ボクセルをクリアする。

        Args:
            points: iss_body座標の点群 (N, 3) ndarray。FOVフィルタ・TF変換済み。
            robot_pos: iss_body座標でのロボット現在位置 (3,) ndarray。
            fov_deg: FOV半角 [deg]。スキャン時のフィルタ角度と合わせる。
            margin_factor: 最大距離に掛ける余裕係数。
        """
        if len(points) == 0:
            rospy.logwarn("CollisionChecker: clear_dynamic_in_fov skipped (empty points)")
            return

        robot_pos = np.asarray(robot_pos, dtype=float)

        # ロボット基準の相対座標
        rel_points = points - robot_pos  # (N, 3)

        # 光軸方向 = ロボット基準の点群平均方向（正規化）
        mean_dir = rel_points.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-9:
            rospy.logwarn("CollisionChecker: clear_dynamic_in_fov skipped (zero mean direction)")
            return
        axis = mean_dir / norm

        # スキャン最大距離（ロボット基準・余裕係数付き）
        max_dist = np.linalg.norm(rel_points, axis=1).max() * margin_factor

        # FOV半角のコサイン閾値（設計書どおり fov_deg を使用）
        cos_limit = np.cos(np.radians(fov_deg))

        # 動的レイヤーの占有ボクセルインデックスを一括取得
        occ_idx = np.argwhere(self._grid_dynamic >= _OCCUPIED_CORE)
        if len(occ_idx) == 0:
            return

        # ボクセルのロボット基準相対座標
        voxel_pos = self._origin + (occ_idx + 0.5) * self._resolution  # (M, 3)
        rel_voxel = voxel_pos - robot_pos  # (M, 3)

        dists = np.linalg.norm(rel_voxel, axis=1)  # (M,)
        in_range = dists <= max_dist

        cos_angles = np.where(
            dists > 1e-9,
            (rel_voxel @ axis) / np.maximum(dists, 1e-9),
            0.0,
        )
        in_cone = cos_angles >= cos_limit

        to_clear = occ_idx[in_range & in_cone]
        if len(to_clear) > 0:
            self._grid_dynamic[to_clear[:, 0], to_clear[:, 1], to_clear[:, 2]] = _FREE
        rospy.loginfo("CollisionChecker: clear_dynamic_in_fov cleared %d / %d dynamic voxels",
                      len(to_clear), len(occ_idx))

    def check_point(self, pos, margin=None):
        idx = self.pos_to_idx(pos)
        if idx is None:
            return False
        # 静的が FREE かつ 動的が FREE (Core/Marginいずれでもない) ことを確認
        return self._grid_static[idx] == _FREE and self._grid_dynamic[idx] == _FREE

    def get_distance(self, pos):
        if not self.check_point(pos):
            return 0.0
        
        idx = self.pos_to_idx(pos)
        if idx is None:
            return 0.0
        search_range = int(np.ceil(self._margin / self._resolution)) + 1
        min_dist_sq = float('inf')
        found = False

        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                for dz in range(-search_range, search_range + 1):
                    n_idx = (idx[0]+dx, idx[1]+dy, idx[2]+dz)
                    if all(0 <= n_idx[i] < self._grid_static.shape[i] for i in range(3)):
                        # 静的 OCCUPIED または 動的 (CORE/MARGIN) のいずれかがある場合
                        if self._grid_static[n_idx] == _OCCUPIED or self._grid_dynamic[n_idx] >= _OCCUPIED_CORE:
                            dist_sq = dx**2 + dy**2 + dz**2
                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq
                                found = True
        
        if not found:
            return self._margin
            
        return np.sqrt(min_dist_sq) * self._resolution

    def compute_edt(self):
        """膨張前の生障害物から EDT を計算.

        静的は膨張前（_grid_static_raw）、動的は CORE のみを使用する。
        これにより EDT 値は実障害物からの距離となり、cc_margin をそのまま
        閾値として使える（膨張済みグリッドだと二重カウントになる）。
        """
        obstacle = (self._grid_static_raw != _FREE) | (self._grid_dynamic >= _OCCUPIED_CORE)
        self._edt = distance_transform_edt(~obstacle) * self._resolution
        rospy.loginfo("CollisionChecker: EDT computed, grid=%s", self._edt.shape)

    def get_distance_edt(self, pos):
        """EDT ベースの O(1) 距離クエリ."""
        idx = self.pos_to_idx(pos)
        if idx is None:
            return 0.0
        return float(self._edt[idx])

    # ---- デュアル EDT（use_dual_edt=True 時のみ使用） ----

    def compute_edt_static(self):
        """静的レイヤーのみで EDT を計算し _edt_static に保持."""
        obstacle = (self._grid_static_raw == _OCCUPIED)
        self._edt_static = distance_transform_edt(~obstacle) * self._resolution
        rospy.loginfo("CollisionChecker: EDT (static) computed, grid=%s", self._edt_static.shape)

    def compute_edt_dynamic(self):
        """動的レイヤーの CORE セルのみで EDT を計算し _edt_dynamic に保持.

        動的セルが存在しない場合は _edt_dynamic = None のまま。
        """
        has_dynamic = np.any(self._grid_dynamic >= _OCCUPIED_CORE)
        if not has_dynamic:
            self._edt_dynamic = None
            rospy.loginfo("CollisionChecker: EDT (dynamic) skipped (no dynamic cells)")
            return
        obstacle = (self._grid_dynamic >= _OCCUPIED_CORE)
        self._edt_dynamic = distance_transform_edt(~obstacle) * self._resolution
        rospy.loginfo("CollisionChecker: EDT (dynamic) computed, grid=%s", self._edt_dynamic.shape)

    def get_distance_edt_static(self, pos):
        """静的 EDT ベースの O(1) 距離クエリ."""
        if self._edt_static is None:
            return float('inf')
        idx = self.pos_to_idx(pos)
        if idx is None:
            return 0.0
        return float(self._edt_static[idx])

    def get_distance_edt_dynamic(self, pos):
        """動的 EDT ベースの O(1) 距離クエリ."""
        if self._edt_dynamic is None:
            return float('inf')
        idx = self.pos_to_idx(pos)
        if idx is None:
            return 0.0
        return float(self._edt_dynamic[idx])

    def check_line(self, p1, p2, margin=None):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        diff = p2 - p1
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            ok = self.check_point(p1)
            if not ok:
                rospy.logwarn_throttle(
                    1.0,
                    "CollisionChecker: check_line rejected degenerate segment at point=%s",
                    np.round(p1, 3).tolist(),
                )
            return ok
        step = self._resolution * 0.5
        n_steps = max(int(np.ceil(dist / step)), 1)
        for i in range(n_steps + 1):
            t = i / n_steps
            pt = p1 + t * diff
            if not self.check_point(pt):
                rospy.logwarn_throttle(
                    1.0,
                    "CollisionChecker: check_line blocked at step=%d/%d point=%s",
                    i,
                    n_steps,
                    np.round(pt, 3).tolist(),
                )
                return False
        return True

    def publish_debug_cloud(self, pub):
        """蓄積された動的障害物(Core/Margin)を可視化用にパブリッシュ."""
        # CORE(2) または MARGIN(3) のインデックスを取得
        occ_indices = np.argwhere(self._grid_dynamic >= _OCCUPIED_CORE)
        if len(occ_indices) == 0:
            return

        # 物理座標に変換
        points_xyz = self._origin + (occ_indices + 0.5) * self._resolution
        # Intensity として値を保持 (2: Core, 3: Margin)
        intensities = self._grid_dynamic[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2]]
        
        # [x, y, z, intensity] の形式でデータを結合
        cloud_data = np.zeros((len(points_xyz), 4), dtype=np.float32)
        cloud_data[:, :3] = points_xyz
        cloud_data[:, 3] = intensities

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "iss_body" # 自己位置推定のフレームに合わせて調整してください

        # PointCloud2 メッセージの作成 (フィールドに intensity を追加)
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('intensity', 12, pc2.PointField.FLOAT32, 1),
        ]
        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        pub.publish(cloud_msg)