# Guidance

waypoint列から、時間の関数としての滑らかな目標軌道（位置・速度・加速度・目標姿勢）を生成し、`/gnc/move_to`アクションとしてgoal駆動で機体を動かすモジュールです。実際に使われている軌道生成は`HermiteSplineTrajectoryGenerator`（C1連続のみ保証する劣化版）。min-snapのコアロジックは実装しない方針（2026-08-24決定）で、`min_snap_trajectory_generator.py`はスケルトンのまま残しています。それ以外（`guidance_node`本体・move-to統合含む）は実装済みです。

## 目次

- [構成](#構成)
- [`guidance.py`（GuidanceNode）の使い方](#guidance-node-usage)
- [トピック・アクション・サービス](#topics-actions-services)
- [パラメータ](#parameters)

<a id="構成"></a>
## 構成

```
guidance/
├── guidance.py                           # GuidanceNode（唯一のROSノード、1file1node）
│                                          # /gnc/move_to（ib2_msgs/action/CtlCommand）を提供
├── global_planner/                       # 大域経路計画（waypoint列の生成）
│   ├── base_global_planner.py                # 共通インターフェース
│   ├── astar_planner.py
│   └── rrt_planner.py
├── ros/                                  # ROS 入出力ラッパ
│   ├── path_publisher.py                     # nav_msgs/Path をRVizへ可視化publish（/gnc/trajectory_path）
│   ├── speed_path_publisher.py                # 速度で色分けしたLINE_STRIP MarkerをRVizへ可視化publish（表示のみ、制御には無関係）
│   ├── multi_dof_joint_trajectory_publisher.py  # /gnc/trajectory_setpoint へ発行（Control側が購読）
│   ├── checkpoint_publisher.py               # /gnc/checkpoints へ発行（事前/到着時整列の静止保持）
│   ├── move_to_client.py                     # move_to_client CLI（名前付きTF地点へgoal送信、手動検証用）
│   └── ctl_command_action_server.py          # ib2_msgs/action/CtlCommand（目標姿勢へのgoal駆動）
├── segment_time/                         # 区間時間配分
│   ├── base_segment_time_allocator.py
│   ├── heuristic_segment_time_allocator.py   # distance/target_speed + 台形速度プロファイルの時間下限
│   └── optimal_segment_time_allocator.py     # シグネチャ確定のみ、本体は未実装（min-snap前提のためスコープ外）
├── trajectory_generation/                # waypoints+区間時間 -> 多項式係数
│   ├── base_trajectory_generator.py
│   ├── hermite_spline_trajectory_generator.py  # 劣化版（C1連続のみ保証）、実装済み・実際に使用中
│   └── min_snap_trajectory_generator.py      # スケルトンのみ、コアロジックは実装しない方針（2026-08-24決定）
└── utils/                                # ROS非依存のロジック
    ├── polynomial.py                         # 多項式（微分）評価
    ├── trajectory.py                         # Trajectory: sample(t) -> (p, v, a, q_des)
    ├── attitude_reference.py                 # v_des(t) -> q_des(t)（進行方向を向く姿勢参照）
    └── guidance_executor.py                  # GuidanceExecutor: 1件のCtlCommand goalを
                                               # pre-align→軌道追従→arrival-alignで駆動（cancel対応）
```

[↑ 目次に戻る](#目次)

<a id="guidance-node-usage"></a>
## `guidance.py`（GuidanceNode）の使い方

`/gnc/move_to`（`ib2_msgs/action/CtlCommand`）を提供する唯一のROSノードです。Control側（`control_node`）が別途起動済みで、`/tf`（`iss_body <- body`）が生きていることが前提です。

### 起動

```sh
export ROS_DOMAIN_ID=54   # 環境に合わせて設定
source /root/colcon_ws/install/setup.bash
ros2 run sobits_intball2_gnc guidance --ros-args --params-file \
  /root/colcon_ws/src/sobits_intball2_gnc/config/gnc_params.yaml
```

パラメータは`config/gnc_params.yaml`の`guidance`セクションと、Control側と共有する`tf_correction.reference_frame`/`target_frame`・`trajectory_controller.max_force`/`mass`（区間時間配分が機体の加速度能力を超えないようにするため、Control側と同じ値を使う）から読む。詳細は次節参照。


### goalを送る（`move_to_client` CLI、手動検証用）

`navigation/`パッケージがTF配信する名前付き地点（`maps/iss_location.yaml`、例: `near_dock`・`above_dock_2`・`nav_entry`）を指定するだけで、その位置・姿勢をgoalとして送信できる:

```sh
ros2 run sobits_intball2_gnc move_to_client near_dock
```

内部で`iss_body <- near_dock`をTFで解決し、`/gnc/move_to`へgoal送信して完了までfeedback（`time_to_go`・`pose_to_go`）をログ表示する。**注意**: `pose_to_go`は計画軌道（open-loop）の残差であり実TF追従の証明にはならない。実際に到達したかは`ros2 run tf2_ros tf2_echo iss_body body`で直接確認すること（詳細: `docs/archive/achieved/2026-08-20_main_plan_completed_phases.md`のGuidanceノード統合の項）。

### goalを送る（標準の`ros2 action` CLI、任意の座標へ）

```sh
ros2 action send_goal /gnc/move_to ib2_msgs/action/CtlCommand \
  "{target: {header: {frame_id: 'iss_body'}, \
     pose: {position: {x: 10.936, y: -3.636, z: 4.121}, \
            orientation: {x: 0.7071067811865476, y: -0.7071067811865475, z: 0.0, w: 0.0}}}, \
    type: {type: 40}}" --feedback
```

`--feedback`を付けたまま`Ctrl-C`（`SIGINT`）を送ると、標準のaction cancelリクエストが送信され、`GuidanceExecutor`が該当ループの次tickで中断する。中断後、Control側は`trajectory_controller.timeout`（既定0.2秒）後にその場（当時の平滑化された現在位置）でのcheckpoint holdへ自動フォールバックする（ファンは止まらない）。中断直後に新しいgoalを送っても正常に受理される。

### 実行の流れ（`GuidanceExecutor.execute()`）

1. 現在姿勢と経路初期進行方向のズレが大きい場合、事前整列（`/gnc/checkpoints`で静止保持、最大`align_timeout`秒）
2. `HeuristicSegmentTimeAllocator`→`HermiteSplineTrajectoryGenerator`→`Trajectory`で生成した軌道を`/gnc/trajectory_setpoint`へ追従再生。計画所要時間が経過しても、実TF位置の誤差が`align_pos_tolerance_m`以下に`align_pos_settle_time`秒連続して収まるまで（最大`align_pos_timeout`秒）は次段へ進まない（`docs/archive/achieved/2026-08-24_align_at_arrival_position_based.md`）
3. 到着後、目標姿勢とのズレが大きければ整列（同じく`/gnc/checkpoints`、最大`align_timeout`秒）

現状`face_travel=True`・`face_travel_camera="main"`・`align_at_arrival=True`固定（`CtlCommand.action`にオプションを渡すフィールドが無いため、インターフェース拡張待ち）。

[↑ 目次に戻る](#目次)

<a id="topics-actions-services"></a>
## トピック・アクション・サービス

### 入力トピック

| トピック名 | 型 | 説明 |
|---|---|---|
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | `iss_body <- body`のTF（`TfClient`経由、到着判定・整列判定に使用） |

### 出力トピック

| トピック名 | 型 | 説明 |
|---|---|---|
| `/gnc/trajectory_setpoint` | `trajectory_msgs/MultiDOFJointTrajectory` | 軌道追従の目標位置・速度・加速度（Control側が購読） |
| `/gnc/checkpoints` | `geometry_msgs/PoseArray` | 事前整列・到着時整列での静止保持目標（Control側が購読） |
| `/gnc/trajectory_path_speed` | `visualization_msgs/Marker` | 速度で色分けしたLINE_STRIPのRViz可視化（表示のみ、制御には無関係） |

### アクション

| アクション名 | 型 | 説明 |
|---|---|---|
| `/gnc/move_to` | `ib2_msgs/action/CtlCommand` | 目標姿勢へのgoal駆動move-to（`GuidanceExecutor`が実行） |

`path_publisher`（`/gnc/trajectory_path`、`nav_msgs/Path`）は`console_scripts`登録済みの単体デバッグ用ラッパのみで、`guidance_node`本体からは配線されていない（実際に使われているのは`speed_path_publisher`）。

[↑ 目次に戻る](#目次)

<a id="parameters"></a>
## パラメータ

分類の考え方（固定/動的）の詳細は[docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md](../../docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md)を参照。

### 固定パラメータ（起動時のみ、実行中は変更不可）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.target_speed` | 巡航速度（区間時間配分用）[m/s] | `0.5` |
| `guidance.attitude_speed_threshold` | 進行方向姿勢参照を更新する速度の下限 [m/s] | `0.02` |
| `guidance.rate` | `/gnc/trajectory_setpoint`発行レート [Hz] | `50.0` |
| `guidance.camera_forward_axis.main` | メインカメラの前方軸（機体座標系） | `[1.0, 0.0, 0.0]` |
| `guidance.camera_forward_axis.stereo` | ステレオカメラの前方軸（機体座標系） | `[0.0, 1.0, 0.0]` |
| `tf_correction.reference_frame` | 自己位置の親フレーム（Control側と共有） | `iss_body` |
| `tf_correction.target_frame` | 機体フレーム（Control側と共有） | `body` |
| `trajectory_controller.max_force` | 区間時間配分の加速度上限算出に使う力 [N]（Control側と共有、軸別ベクトル。加速度上限は最も厳しい軸=min値を採用） | `[0.181, 0.0996, 0.122]` |
| `trajectory_controller.mass` | 区間時間配分の加速度上限算出に使う質量 [kg]（Control側と共有） | `3.216` |
| `guidance.wrench_envelope_safety_margin` | `static`モード（TOPP-RA）の達成可能ウレンチ包絡域（`wrench_envelope_halfspaces`）を原点中心にこの係数で縮小し、フィードバック補正の余力を計画段階から確保する（`docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その5/6） | `0.7` |
| `guidance.align_traj_publish_rate_hz` | SLERP+台形整列ランプの中間目標publishレート [Hz] | `20.0` |
| `guidance.velocity_estimate_rate` | Guidance側TF速度推定器の更新レート [Hz]（`TrajectoryController`自身の推定器とは独立） | `10.0` |
| `guidance.replan_rate_hz` | `replanning`モードの再計画レート [Hz]（`velocity_estimate_rate`と一致させる） | `10.0` |

### 動的パラメータ（`ros2 param set`で実行中に変更可能）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.align_tolerance_deg` | 事前/事後アラインメントの収束判定角度 [deg] | `3.0` |
| `guidance.align_timeout` | 事前/事後アラインメントの安全カットオフ [s] | `60.0` |
| `guidance.align_pos_tolerance_m` | 軌道追従（`_run_trajectory`）の到着判定に使う位置誤差許容値 [m]。計画所要時間経過後、この誤差以下に収まるまで待つ（`align_tolerance_deg`の位置版、独立パラメータ） | `0.05` |
| `guidance.align_pos_settle_time` | 上記の許容値内に連続して留まるべき最小時間 [s]（オーバーシュート通過の誤検知防止、`align_settle_time`と同じ目的） | `0.5` |
| `guidance.align_pos_timeout` | 位置収束待ちの安全カットオフ [s]。超えると警告ログを出して進行（`align_timeout`より短い——並進残差を詰めるだけの待ちのため） | `10.0` |
| `guidance.attitude_reference_mode` | 移動中の姿勢参照モード（`fixed`/`face_travel`/`look_at`、goal受理時にラッチ）。`look_at`は未実装で`face_travel`にフォールバック（警告ログ） | `face_travel` |
| `guidance.pre_align` | 出発前の事前整列を行うか（goal受理時にラッチ、`attitude_reference_mode=face_travel`のときのみ効く） | `true` |
| `guidance.align_at_arrival` | 到着後、再整列するか（目標は`align_at_arrival_camera`で決まる。goal受理時にラッチ） | `true` |
| `guidance.face_travel_camera` | `attitude_reference_mode=face_travel`のとき進行方向に向けるカメラ軸（`main`/`stereo`、goal受理時にラッチ） | `main` |
| `guidance.look_at_target_frame` | `attitude_reference_mode=look_at`（未実装）で見る対象のTFフレーム名（goal受理時にラッチ） | `""` |
| `guidance.align_at_arrival_camera` | 到着後どのカメラ軸を基準に整列するか。`main`はgoalの`q_target`そのまま、他のカメラ（例: `stereo`）は「`q_target`のときメインカメラが見ていたはずの方向」をそのカメラの軸で向くよう計算（`compute_camera_relative_quat`） | `main` |
| `guidance.align_settle_time` | 事前/事後アラインメントで角度閾値内が連続してこの時間続いたら収束とみなす（`align_pos_settle_time`とは別、姿勢版） | `0.5` |
| `guidance.align_angular_speed_deg` | SLERP+台形整列ランプの巡航角速度 [deg/s] | `15.0` |
| `guidance.align_angular_accel_deg` | 同ランプの角加速度 [deg/s^2] | `2.4` |
| `guidance.tf_staleness_timeout` | このノード自身のsimクロック基準で、TFのstampがこの時間止まったらTF断とみなす [s] | `1.0` |
| `guidance.velocity_estimate_alpha` | Guidance側TF速度推定のEMA係数（1.0で無フィルタ） | `0.3` |
| `guidance.trajectory_tracking_mode` | 軌道追従方式（`static`=開ループ単一軌道 / `replanning`=実TFから再計画、goal受理時にラッチ） | `static` |
| `guidance.max_angular_rate_deg` | `q_des`のレート制限 [deg/s]（両trackingモード共通、未チューニング） | `90.0` |
| `guidance.distance_fallback_m` | `replanning`モードのみ: 残距離がこの値未満になったら残りは再計画を打ち切る [m] | `0.3` |

### その他のROS I/Oラッパ

`path_publisher`・`multi_dof_joint_trajectory_publisher`・`checkpoint_publisher`・`ctl_command_action_server`は`console_scripts`登録済みだが、通常は`guidance_node`から利用するライブラリとしての位置づけで、単体`ros2 run`はデバッグ用途のみ（各ファイルの`main()`docstring参照）。

min-snapのコアロジックが未実装（実装しない方針）な状態での追加の動作確認手段として、`test/manual/`のスタンドインスクリプトで軌道を直接publishする方法もある（詳細: `test/manual/README.md`）。

[↑ 目次に戻る](#目次)
