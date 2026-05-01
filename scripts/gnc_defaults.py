"""GNC パラメータのデフォルト値.

gnc_params.yaml が読み込まれている場合はパラメータサーバーの値が優先される。
このモジュールは rospy.get_param() の第2引数として使用し、
複数クラス間でデフォルト値が一致することを保証する。

使用例:
    from gnc_defaults import GNC_DEFAULTS
    clearance = rospy.get_param('/gnc/static_clearance', GNC_DEFAULTS['static_clearance'])
"""

GNC_DEFAULTS = {
    # ロボット基本パラメータ
    'robot_radius': 0.10,

    # 障害物回避クリアランス
    'static_clearance': 0.35,
    'min_clearance': 0.25,
    'dynamic_clearance': 0.05,

    # 再計画制御
    'max_replan_retries': 3,
    'dynamic_clear_mode': 'fov',
    'dynamic_clearance_recheck_enabled': False,
    'dynamic_clearance_min_distance': 0.30,
    'dynamic_clearance_recheck_max_retries': 3,

    # スキャン・仮想ウェイポイント
    'scan_interval': 1.6,
    'scan_fov_deg': 20.0,
    'scan_timeout': 5.0,
    'scan_min_points': 100,
    'scan_sampling_step': 5,

    # A* プランナー
    'grid_resolution': 0.15,
    'bbox_pad': 2.2,
    'bbox_extra_min_x': 0.0,
    'bbox_extra_max_x': 0.0,
    'bbox_extra_min_y': 0.0,
    'bbox_extra_max_y': 0.30,
    'bbox_extra_min_z': 0.0,
    'bbox_extra_max_z': 0.30,
    'astar_max_iter': 12000,

    # パス後処理（Smoother）
    'path_filters': ['push_from_walls', 'shortcut'],
    'shortcut_margin': 0.0,
    'push_step': 0.05,
    'push_max_iter': 10,

    # SafetyAwareAStarPlanner（ポテンシャル場 A*）
    'use_potential_astar': False,
    'safety_weight': 1.5,
    'safety_threshold': 0.30,

    # デュアル EDT（静的・動的分離コスト計算）
    'use_dual_edt': False,
    'dynamic_threshold': 0.30,
    'static_weight': 1.5,
    'dynamic_weight': 5.0,
    'wall_filter_dist': 0.20,

    # direct_rotate 姿勢検証許容誤差
    'direct_rotate_yaw_tol_deg': 5.0,
    'direct_rotate_pitch_tol_deg': 5.0,

    # タイミング制御
    'wp_move_timeout': 45.0,
    'stabilize_wait': 2.0,
    'initial_scan_wait': 0.5,
    'final_adjust_sample_interval': 0.1,

    # 実行モード・トピック設定
    'executor_mode': 'steady',
    'obstacle_topic': '/depth/points',

    # ナビゲーションモード
    # 'full': 通常モード（点群・経路計画・回転すべて有効）
    # 'direct': 直接TF追従モード（障害物なし前提、ドックエリア向け）
    # 'direct_rotate': 直接TF追従 + 終端姿勢合わせ（single move_to）
    'navigation_mode': 'full',
}
