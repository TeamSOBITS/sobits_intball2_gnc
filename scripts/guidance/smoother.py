#!/usr/bin/env python3
"""経路後処理: ショートカット、壁面押し出し、micro-WP補間."""
import math
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler

_EDT_TOLERANCE = 0.98  # 離散化誤差バッファ (2%)


def shortcut(path, collision_checker, shortcut_margin=None, **kwargs):
    if len(path) <= 2:
        return list(path)

    # 動的レイヤーの解像度よりも細かいステップでチェックするための設定
    # 解像度の1/3程度のステップにすれば、薄い動的障害物も踏み抜かなくなる
    step_size = collision_checker.resolution * 0.3

    # セグメントマージン判定: shortcut_margin > 0 なら EDT を事前計算
    cc_margin = shortcut_margin if shortcut_margin is not None else 0.0
    if cc_margin > 0:
        collision_checker.compute_edt()

    result = [np.asarray(path[0], dtype=float)]
    i = 0
    while i < len(path) - 1:
        farthest = i + 1
        # 最遠候補から順に試行（最初の3候補のみデバッグログ出力）
        tried = 0
        for j in range(len(path) - 1, i + 1, -1):
            label = "WP%d->%d" % (i, j) if tried < 3 else ""
            tried += 1
            if safe_check_line(path[i], path[j], collision_checker, step_size,
                               margin=cc_margin, debug_label=label):
                farthest = j
                break
        rospy.loginfo("Shortcut step: WP%d -> WP%d (tried %d candidates)", i, farthest, tried)
        result.append(np.asarray(path[farthest], dtype=float))
        i = farthest

    rospy.loginfo("Shortcut: %d -> %d waypoints (margin=%.3f)", len(path), len(result), cc_margin)
    return result

def safe_check_line(p1, p2, cc, step_size, margin=0.0, debug_label=""):
    """標準の check_line よりも高密度にチェックするヘルパー.

    margin > 0 の場合、占有判定に加え EDT 距離が margin 以上であることも検証する。
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-9:
        return cc.check_point(p1)

    effective_margin = margin * _EDT_TOLERANCE
    n_steps = int(math.ceil(dist / step_size))
    min_edt = float('inf')
    for k in range(n_steps + 1):
        pt = p1 + (p2 - p1) * (float(k) / n_steps)
        if not cc.check_point(pt):
            if debug_label:
                rospy.loginfo("  %s REJECT(occupied) at step %d/%d pt=[%.2f,%.2f,%.2f]",
                              debug_label, k, n_steps, pt[0], pt[1], pt[2])
            return False
        if effective_margin > 0:
            d = cc.get_distance_edt(pt)
            if d < min_edt:
                min_edt = d
            if d < effective_margin:
                if debug_label:
                    rospy.loginfo("  %s REJECT(margin) at step %d/%d pt=[%.2f,%.2f,%.2f] "
                                  "edt=%.3f < %.3f",
                                  debug_label, k, n_steps, pt[0], pt[1], pt[2],
                                  d, effective_margin)
                return False
    if debug_label:
        rospy.loginfo("  %s OK min_edt=%.3f (thresh=%.3f)",
                      debug_label, min_edt, effective_margin)
    return True


def push_from_walls(path, collision_checker, push_step=0.05, max_iter=10, **kwargs):
    """壁に近すぎるウェイポイントを法線方向に押し出す.

    各 WP について、6 方向（±x, ±y, ±z）の最近障害物方向から離れるよう移動する。
    start と goal（先頭・末尾）は移動しない。
    """
    if len(path) <= 2:
        return list(path)

    result = [np.asarray(p, dtype=float) for p in path]
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)

    # プロパティ経由でマージンを取得（デフォルト 0.15）
    cc_margin = getattr(collision_checker, "margin", 0.15)

    for _ in range(max_iter):
        moved = False
        for idx in range(1, len(result) - 1):
            pt = result[idx]
            
            # 現在の点から障害物までの距離を取得 (ESDFインターフェースの活用)
            dist = collision_checker.get_distance(pt)
            
            # マージン内に障害物がある場合のみ斥力を計算
            if dist < cc_margin:
                push_dir = np.zeros(3)
                near_obstacle = False
                
                for d in directions:
                    # 周囲のマージン圏内をチェック
                    if not collision_checker.check_point(pt + d * cc_margin):
                        push_dir -= d
                        near_obstacle = True
                
                if near_obstacle:
                    norm = np.linalg.norm(push_dir)
                    if norm > 1e-9:
                        new_pt = pt + (push_dir / norm) * push_step
                        if collision_checker.check_point(new_pt):
                            result[idx] = new_pt
                            moved = True
        if not moved:
            break

    return result


def interpolate_path(path, step=0.05, yaw_lookahead=3):
    """smoother出力パスを ~step m 間隔の micro-WP 列に補間する.

    各 micro-WP は (position, quaternion) のタプル。
    姿勢は yaw_lookahead 個先の WP 方向から先行算出する。
    path は ISS 座標系の [[x,y,z], ...] を想定。
    """
    if len(path) < 2:
        return [(np.asarray(path[0], dtype=float), quaternion_from_euler(0, 0, 0))]

    # 1. 位置の線形補間
    positions = []
    for i in range(len(path) - 1):
        a = np.asarray(path[i], dtype=float)
        b = np.asarray(path[i + 1], dtype=float)
        seg_len = np.linalg.norm(b - a)
        if seg_len < 1e-6:
            continue
        n_pts = max(1, int(math.ceil(seg_len / step)))
        for j in range(n_pts):
            t = j / n_pts
            positions.append(a + t * (b - a))
    positions.append(np.asarray(path[-1], dtype=float))

    # 2. 姿勢の先行制御 (yaw_lookahead 先の方向ベクトルから yaw を算出)
    micro_wps = []
    n = len(positions)
    for i in range(n):
        look_idx = min(i + yaw_lookahead, n - 1)
        diff = positions[look_idx] - positions[i]
        dist_xy = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        dist_3d = np.linalg.norm(diff)

        if dist_3d > 1e-6:
            yaw = math.atan2(diff[1], diff[0]) if dist_xy > 1e-6 else 0.0
            pitch = math.atan2(-diff[2], max(dist_xy, 1e-6))
        elif i > 0:
            # 最終付近で差分がほぼゼロ → 直前の姿勢を引き継ぐ
            _, prev_q = micro_wps[-1]
            micro_wps.append((positions[i], prev_q))
            continue
        else:
            yaw, pitch = 0.0, 0.0

        q = quaternion_from_euler(0, pitch, yaw)
        micro_wps.append((positions[i], q))

    rospy.loginfo("interpolate_path: %d -> %d micro-WPs (step=%.3fm)", len(path), len(micro_wps), step)
    return micro_wps