# Guidance

waypoint列から、時間の関数としての滑らかな目標軌道（位置・速度・加速度・目標姿勢）を生成し、`/gnc/move_to`アクションとしてgoal駆動で機体を動かすモジュールです。軌道はファンで出せる力・トルクの範囲（wrench envelope）に収まるように作ります: TOPP-RA（`static_toppra`、既定）またはMINCO（`static_minco`・`replan_minco`）。軌道を作れないときはgoalをabortし、その場で止まります。

## 目次

- [構成](#構成)
- [move_toの使い方](#guidance-node-usage)
- [よく使う設定](#common-settings)
- [実行の流れ](#execution-flow)
- [トピック・アクション・サービス](#topics-actions-services)
- [パラメータ一覧（参照用）](#parameters)

<a id="構成"></a>
## 構成

```
guidance/
├── guidance.py                           # GuidanceNode（唯一のROSノード、1file1node）
│                                          # /gnc/move_to（ib2_msgs/action/CtlCommand）を提供
├── guidance_params.py                    # GuidanceNodeのパラメータの既定値・読み取り専用の一覧・goalごとの読み取り
├── ros/                                  # ROS 入出力ラッパ
│   ├── path_publisher.py                     # nav_msgs/Path をRVizへ可視化publish（/gnc/trajectory_path）
│   ├── speed_path_publisher.py               # 速度で色分けしたLINE_STRIP MarkerをRVizへ可視化publish（表示のみ、制御には無関係）
│   ├── multi_dof_joint_trajectory_publisher.py  # /gnc/trajectory_setpoint へ発行（Control側が購読）
│   ├── checkpoint_publisher.py               # /gnc/checkpoints へ発行（事前/到着時整列の静止保持）
│   ├── marker_array_subscriber.py            # /guidance/virtual_obstacles（MarkerArrayのCUBE）を箱の変更にして渡す
│   ├── marker_array_publisher.py             # 今の箱の一覧を /guidance/obstacles_active へ発行（RViz表示）
│   ├── move_to_client.py                     # move_to_client CLI（名前付きTF地点へgoal送信、手動検証用）
│   └── ctl_command_action_server.py          # ib2_msgs/action/CtlCommand（目標姿勢へのgoal駆動）
├── executor/                             # 1つのgoalの流れ
│   ├── guidance_executor.py                  # GuidanceExecutor: pre-align→軌道追従→arrival-align、cancel・計画失敗後はbrake()で制動
│   ├── tracker_builder.py                    # goalのtrajectory_tracking_modeに応じてtrackerを組み立てる（作れなければTrajectoryBuildError）
│   └── cancel_brake.py                       # 停止プロファイルに沿って止め、止まった点で静止保持
├── global_planner/                       # 大域経路計画（本番では未使用、単体テストのみ）
│   ├── base_global_planner.py                # 共通インターフェース
│   ├── astar_planner.py
│   └── rrt_planner.py
├── local_planner/                        # 障害物を見たlocalの計画
│   ├── minco_local_planner.py                # replan_mincoのglobal/localの作り方（EGO-Planner v2のplanner_manager）
│   └── obstacle_map.py                       # 静的な地図（OctoMap）＋仮想の箱から格子地図を作り直す
├── trajectory/                           # 軌道の表現と生成
│   ├── minco_trajectory.py                   # MINCO姿勢/トルク統合軌道（minco_native_py拡張のPythonラッパ）
│   ├── toppra_trajectory.py                  # TOPP-RAによる力/トルク制約付き時間割当済み軌道
│   └── generation/                           # waypoints+区間時間 -> 多項式係数
│       ├── base_trajectory_generator.py
│       ├── hermite_spline_trajectory_generator.py  # C1連続のHermiteスプライン（TOPP-RAの幾何経路に使用）
│       └── min_snap_trajectory_generator.py      # スケルトンのみ、コアロジックは実装しない方針（2026-08-24決定）
├── trajectory_tracking/                  # 生成済み軌道の追従方式
│   ├── base_trajectory_tracker.py            # 共通インターフェース
│   ├── static_trajectory_tracker.py          # 開ループ単一軌道を最後まで追従（static_toppra・static_minco）
│   └── replan_minco_tracker.py               # global MINCO軌道を一度だけ解き、local区間を一定周期で再計画しながら追従（衝突確認・非常停止）
├── align/                                # 事前/事後アラインメント（SLERP+台形角速度ランプ）
│   ├── angular_trajectory.py                 # 角度台形プロファイル（角速度・角加速度上限からランプ軌道を生成）
│   └── attitude_aligner.py                   # 現在姿勢->目標姿勢のSLERP+台形ランプ整列を駆動
├── estimation/                           # 状態推定
│   ├── velocity_estimator.py                 # TF位置列からのGuidance側速度推定（EMA平滑化）
│   └── model_kf_estimator.py                 # 指令加速度で予測・観測位置で補正する定加速度カルマンフィルタ（現在未使用）
├── constraints/                          # 機体の制約
│   ├── actuation_envelope.py                 # 機体の達成可能wrench包絡域（wrench_envelope_halfspaces等）の算出
│   └── wrench_envelope_constraint.py         # TOPP-RA用の経路非依存wrench包絡域制約（ToppraTrajectoryが使用）
└── utils/                                # 数学の小道具
    ├── polynomial.py                         # 多項式（微分）評価
    ├── attitude_reference.py                 # v_des(t) -> q_des(t)（進行方向を向く姿勢参照）
    └── quintic_hermite.py                    # 両端の位置/速度/加速度から5次多項式を解析的に解く（現在未使用）
```

cancel後の制動プロファイルは`guidance/`の外、`common/utils/stopping_profile.py`（ROS非依存、JAXA `ctl_only`の`stoppingProfile()`の移植）にある。将来Control側からも使うため共通の場所に置いている。

[↑ 目次に戻る](#目次)

<a id="guidance-node-usage"></a>
## move_toの使い方

`/gnc/move_to`（`ib2_msgs/action/CtlCommand`）で、機体を目標の位置・姿勢へ動かします。

### 1. 起動

2つを別々に起動します。起動前に`ros2 node list`で、既に動いているノードがないか確認してください（多重起動すると機体が暴れます）。

```sh
export ROS_DOMAIN_ID=54   # 環境に合わせて設定
source /root/colcon_ws/install/setup.bash

ros2 launch sobits_intball2_gnc gnc_bringup.launch.py   # control_node・TF・機体モデル・RViz・名前付き地点のTF配信
ros2 launch sobits_intball2_gnc guidance.launch.py      # guidance_node（gnc_params.yamlを読む）
```

`control_node`を別に起動する場合（sim/bridgeの再起動前に`control_node`だけ止めたいときなど）は、`gnc_bringup.launch.py use_control:=false`と`control.launch.py`を起動します。
`guidance_node`は`gnc_bringup.launch.py`には含まれず、`guidance.launch.py`で単独起動します。`ros2 run sobits_intball2_gnc guidance`で直接起動すると`gnc_params.yaml`が読まれず、コード内の既定値で動くので使わないでください。

### 2. goalを送る

名前付き地点（`maps/iss_location.yaml`のTFフレーム名、例: `nav_entry`・`inspection_entry_1`・`above_dock_2`）を指定します。

```sh
ros2 run sobits_intball2_gnc move_to_client nav_entry
```

その地点の位置・姿勢をTFで解決してgoalを送り、完了まで`time_to_go`・`pose_to_go`を表示します。`pose_to_go`は計画上の残りで、実際に着いたかどうかは分かりません（次の「3. 結果を確かめる」で確認します）。

任意の座標へ送る場合は標準の`ros2 action`を使います:

```sh
ros2 action send_goal /gnc/move_to ib2_msgs/action/CtlCommand \
  "{target: {header: {frame_id: 'iss_body'}, \
     pose: {position: {x: 10.936, y: -3.636, z: 4.121}, \
            orientation: {x: 0.7071067811865476, y: -0.7071067811865475, z: 0.0, w: 0.0}}}, \
    type: {type: 40}}" --feedback
```

`--feedback`付きで`Ctrl-C`するとgoalがキャンセルされます。結果（canceled）はすぐ返り、そのあとGuidanceが現在の速度・角速度から止まれる地点を計算して減速し、止まった地点で静止保持します（下の「実行の流れ」4）。制動中に送ったgoalは拒否されます。

### 3. 結果を確かめる

```sh
python3 gnc/test/manual/get_pose.py                          # 今の位置・姿勢（iss_body <- body）
python3 gnc/test/manual/move_to_full_analysis.py nav_entry   # goalを送り、追従誤差・duty飽和・wrenchをまとめて表示
python3 gnc/test/manual/move_to_cancel_brake_test.py inspection_entry_2   # 途中でcancelし、止まり方（行き過ぎ・戻り・静定時間）を表示
```

`move_to_full_analysis.py`は`move_to_client`の代わりにgoalを送り、走行中の記録からレポートとCSVを出します（詳細: `gnc/test/manual/README.md`）。

[↑ 目次に戻る](#目次)

<a id="common-settings"></a>
## よく使う設定

どれも`ros2 param set /guidance_node <名前> <値>`で変更でき、次に送るgoalから効きます（走行中のgoalには効きません）。

| やりたいこと | パラメータ | 値 |
|---|---|---|
| 追従の方式を選ぶ | `guidance.trajectory_tracking_mode` | `static_toppra`（既定、一度だけ計画した軌道を追従）/ `static_minco` / `replan_minco`（1秒ごとに再計画） |
| 経由点を通る | `guidance.via_waypoints` | 地点名の配列、例: `"['nav_entry']"`。**使い終わったら`"['']"`に戻す**（残すと以降の全goalが経由する） |
| 進行方向を向いて移動する | `guidance.attitude_reference_mode` | `face_travel`（既定）/ `fixed` |
| 同上（`replan_minco`のとき） | `guidance.minco_replan_face_travel`、`guidance.minco_planning_horizon_m` | 既定で`true`と`4.0`（姿勢を固定するなら`false`。face travelのまま先読みを2mにすると角を曲がりきれない） |
| 出発前・到着時の姿勢合わせ | `guidance.pre_align`、`guidance.align_at_arrival` | `true`（既定）/ `false` |

例: `replan_minco`で進行方向を向き、`nav_entry`を経由して`inspection_entry_1`へ行く

```sh
ros2 param set /guidance_node guidance.trajectory_tracking_mode replan_minco
ros2 param set /guidance_node guidance.minco_replan_face_travel true
ros2 param set /guidance_node guidance.minco_planning_horizon_m 4.0
ros2 param set /guidance_node guidance.via_waypoints "['nav_entry']"
ros2 run sobits_intball2_gnc move_to_client inspection_entry_1
ros2 param set /guidance_node guidance.via_waypoints "['']"
```

[↑ 目次に戻る](#目次)

<a id="execution-flow"></a>
## 実行の流れ（`GuidanceExecutor.execute()`）

1. **出発前の姿勢合わせ**（`pre_align`、`attitude_reference_mode=face_travel`のとき）: 最初の進行方向へ向きを合わせる。`/gnc/checkpoints`で静止保持、最大`align_timeout`秒
2. **軌道の追従**: `/gnc/trajectory_setpoint`へ参照を出す。方式は`trajectory_tracking_mode`で決まる
   - `static_toppra`: 力・トルクの制約付きで一度だけ計画した軌道（TOPP-RA）
   - `static_minco`: MINCOで一度だけ計画した軌道
   - `replan_minco`: ゴールまでのglobal軌道を一度だけ作り、そこから先読み距離先までのlocal軌道を1秒ごとに作り直す（EGO-Planner v2と同じ構成）。最初のglobal軌道を作れなければ`static_toppra`で作り直す
   - 軌道を作れないとき（TOPP-RA・MINCOが解けない、wrench envelope・質量・慣性・`max_angular_rate`が未設定）はgoalを`TERMINATE_ABORTED`で返し、下の4と同じ制動で止まる
   - 計画時間が過ぎても、位置誤差が`align_pos_tolerance_m`以下に`align_pos_settle_time`秒続くまで待つ（最大`align_pos_timeout`秒）
3. **到着時の姿勢合わせ**（`align_at_arrival`）: 目標姿勢へ合わせる。`/gnc/checkpoints`で静止保持、最大`align_timeout`秒
4. **cancelされたとき・軌道を作れなかったとき**（`GuidanceExecutor.brake()`）: 結果を返したあと別スレッドで実行する。JAXA `ctl_only`と同じく、先に回転を止め（その間の並進は等速）、次に並進を一定の減速度で止める軌道を`/gnc/trajectory_setpoint`へ出す。減速度はファン1基あたり`wrench_envelope_safety_margin`倍までの推力で出せる値で、さらに各軸`hover_control.max_force`以下に抑える。軌道を最後まで出し、停止点から`stopping.tolerance_pos`・`stopping.tolerance_att`以内に`stopping.duration_goal`秒いたら（最大は軌道時間＋`stopping.wait_cancel`秒）、停止点を`/gnc/checkpoints`で静止保持にする。停止距離は速度の2乗に比例する（0.5 m/sから3〜6 m）。

`CtlCommand.action`にはオプションを渡すフィールドが無いため、goalごとの設定はすべてROSパラメータで渡す。

[↑ 目次に戻る](#目次)

<a id="topics-actions-services"></a>
## トピック・アクション・サービス

### 入力トピック

| トピック名 | 型 | 説明 |
|---|---|---|
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | `iss_body <- body`のTF（`TfClient`経由、到着判定・整列判定に使用） |
| `/imu/imu` | `ib2_msgs/IMU` | 角速度（cancel後の制動の初期角速度に使用） |
| `/guidance/virtual_obstacles` | `visualization_msgs/MarkerArray` | 地図にない障害物（仮想の箱、CUBE）の追加・削除。`test/manual/virtual_obstacle.py`で送れる。あとでセンサーからの入力に差し替える前提の入口 |

### 出力トピック

| トピック名 | 型 | 説明 |
|---|---|---|
| `/gnc/trajectory_setpoint` | `trajectory_msgs/MultiDOFJointTrajectory` | 軌道追従の目標位置・速度・加速度（Control側が購読） |
| `/gnc/checkpoints` | `geometry_msgs/PoseArray` | 事前整列・到着時整列での静止保持目標（Control側が購読） |
| `/gnc/trajectory_path_speed` | `visualization_msgs/Marker` | 速度で色分けした軌道のRViz表示（表示のみ、制御には無関係） |
| `/gnc/trajectory_path_speed_local` | `visualization_msgs/Marker` | `replan_minco`のlocal軌道のRViz表示（表示のみ） |
| `/guidance/obstacles_active` | `visualization_msgs/MarkerArray` | 今の障害物の地図に入っている仮想の箱の一覧（RViz表示、transient local） |

### アクション

| アクション名 | 型 | 説明 |
|---|---|---|
| `/gnc/move_to` | `ib2_msgs/action/CtlCommand` | 目標姿勢へのgoal駆動move-to（`GuidanceExecutor`が実行） |

`path_publisher`（`/gnc/trajectory_path`、`nav_msgs/Path`）は`console_scripts`登録済みの単体デバッグ用ラッパのみで、`guidance_node`本体からは配線されていない。

[↑ 目次に戻る](#目次)

<a id="parameters"></a>
## パラメータ一覧（参照用）

普段は[よく使う設定](#common-settings)だけで足ります。

### goalごとの設定（`ros2 param set`で変更、次のgoalから有効）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.trajectory_tracking_mode` | 軌道追従方式（`static_toppra` / `static_minco` / `replan_minco`、[実行の流れ](#execution-flow)参照） | `static_toppra` |
| `guidance.via_waypoints` | 経由点のTFフレーム名の配列（順に経由）。`['']`で経由なし | `['']` |
| `guidance.attitude_reference_mode` | 移動中の姿勢参照（`fixed`/`face_travel`/`look_at`）。`look_at`は未実装で`face_travel`にフォールバック（警告ログ） | `face_travel` |
| `guidance.face_travel_camera` | `face_travel`で進行方向に向けるカメラ軸（`main`/`stereo`） | `main` |
| `guidance.look_at_target_frame` | `look_at`（未実装）で見る対象のTFフレーム名 | `""` |
| `guidance.pre_align` | 出発前の姿勢合わせを行うか（`face_travel`のときのみ効く） | `true` |
| `guidance.align_at_arrival` | 到着後に姿勢合わせを行うか | `true` |
| `guidance.align_at_arrival_camera` | 到着後どのカメラ軸を基準に合わせるか。`main`はgoalの姿勢そのまま、他のカメラは「goalの姿勢でメインカメラが見ていた方向」をそのカメラで向く（`compute_camera_relative_quat`） | `main` |

### MINCO（`static_minco`・`replan_minco`）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.minco_via_half_width` | 経由点・分割点を±この幅[m]の箱の中で動かせる。`0.0`で厳密に通過 | `0.0` |
| `guidance.minco_attitude_resample_spacing_m` | 経路をこの間隔[m]で分割して姿勢の経由点を置く。`0.0`で分割しない | `0.3` |
| `guidance.minco_freetime` | `replan_minco`のglobalを、区間時間も最適化する方式（`plan_minco`）で解く。`false`は`target_speed`・加速度上限から区間時間を決める方式 | `false` |
| `guidance.minco_local_replan_period` | `replan_minco`のlocal再計画周期[s] | `1.0` |
| `guidance.minco_planning_horizon_m` | `replan_minco`のlocalの先読み距離[m]（global上で直線距離がこの値以上になる最初の点を目標にする） | `4.0` |
| `guidance.minco_replan_face_travel` | `replan_minco`で進行方向を向く（`attitude_reference_mode=face_travel`も必要）。localを2回solveし、wrenchを機体座標で評価する。先読みは`4.0`程度にする | `true` |
| `guidance.minco_local_max_vel` | `minco_replan_face_travel`のときのlocalの速度上限[m/s] | `0.15` |
| `guidance.minco_obstacle_avoidance` | `replan_minco`で障害物の地図（JEMの壁＋仮想の箱）を避ける（`minco_replan_face_travel`も必要）。避けきれないときは停止プロファイルで非常停止し、静止から再計画する | `false` |
| `guidance.minco_local_piece_length_m` | 障害物を避けるときのlocalの1区間の長さ[m]（EGO-Planner v2の`polyTraj_piece_length`） | `1.5` |
| `guidance.minco_obstacle_clearance_soft` | 障害物を避けるときの緩い余裕[m]（ぶつかった障害物から離す距離） | `0.2` |

### 姿勢合わせ・到着判定

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.align_tolerance_deg` | 姿勢合わせの収束判定角度[deg] | `3.0` |
| `guidance.align_settle_time` | 角度が閾値内に連続してこの時間続いたら収束とみなす[s] | `0.5` |
| `guidance.align_timeout` | 姿勢合わせの打ち切り時間[s] | `60.0` |
| `guidance.align_pos_tolerance_m` | 軌道追従の到着判定の位置誤差許容値[m] | `0.05` |
| `guidance.align_pos_settle_time` | 位置誤差が許容値内に連続して留まるべき時間[s] | `0.5` |
| `guidance.align_pos_timeout` | 位置収束待ちの打ち切り時間[s]（超えると警告ログを出して進む） | `10.0` |

### その他（実行中に変更可能）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.tf_staleness_timeout` | TFのstampがこの時間[s]止まったらTF断とみなす（simクロック基準） | `1.0` |
| `guidance.velocity_estimate_alpha` | Guidance側TF速度推定のEMA係数（1.0で無フィルタ） | `0.3` |

### 起動時のみ（実行中に変えても反映されない）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `guidance.target_speed` | 巡航速度[m/s]（区間時間配分、`replan_minco`のglobal） | `0.5` |
| `guidance.attitude_speed_threshold` | 進行方向の姿勢参照を更新する速度の下限[m/s] | `0.02` |
| `guidance.rate` | `/gnc/trajectory_setpoint`発行レート[Hz] | `50.0` |
| `guidance.velocity_estimate_rate` | Guidance側TF速度推定の更新レート[Hz] | `10.0` |
| `guidance.max_angular_rate_deg` | `q_des`のレート制限[deg/s]（未チューニング） | `90.0` |
| `guidance.align_angular_speed_deg` | 姿勢合わせランプの巡航角速度[deg/s] | `15.0` |
| `guidance.align_angular_accel_deg` | 姿勢合わせランプの角加速度[deg/s^2]（control_nodeのゲインを変えたら手で再計算） | `2.4` |
| `guidance.align_traj_publish_rate_hz` | 姿勢合わせランプの中間目標のpublishレート[Hz] | `20.0` |
| `guidance.wrench_envelope_safety_margin` | 達成可能なwrenchの範囲をこの係数で縮め、フィードバックの余力を残す（全モード共通） | `0.7` |
| `guidance.camera_forward_axis.main` | メインカメラの前方軸（機体座標系） | `[1.0, 0.0, 0.0]` |
| `guidance.camera_forward_axis.stereo` | ステレオカメラの前方軸（機体座標系） | `[0.0, 1.0, 0.0]` |
| `guidance.stopping.x_threshold` | cancel後の制動: 停止距離がこれ未満なら即その場で静止保持[m] | `0.05` |
| `guidance.stopping.theta_threshold` | 同、回転量がこれ未満なら即停止[rad] | `0.01745` |
| `guidance.stopping.f_max`・`t_max` | 最大減速度を探すときの基準の力[N]・トルク[Nm]（結果は変わらない） | `0.181`・`0.0081904` |
| `guidance.stopping.tolerance_pos`・`tolerance_att` | 制動終了とみなす停止点からの距離[m]・角度[rad] | `0.30`・`1.0` |
| `guidance.stopping.duration_goal` | 上の範囲にこの秒数いたら制動終了[s] | `3.0` |
| `guidance.stopping.wait_cancel` | 軌道時間＋この秒数で打ち切って静止保持[s] | `10.0` |
| `guidance.obstacle_map_file` | 障害物の地図（OctoMap `.bt`）。相対パスは`share/sobits_intball2_gnc/maps/`から。`""`で静的な地図なし（仮想の箱だけ） | `jem_octomap.bt` |
| `guidance.obstacle_grid_resolution` | 障害物の格子の解像度[m] | `0.1` |
| `guidance.obstacle_grid_inflation` | 障害物の格子の膨張[m]（機体半径0.1m＋余裕0.1m）。マス単位で効く（解像度0.1mなら0.15は0.2と同じ） | `0.2` |

### Control側と共有（起動時のみ、`gnc_params.yaml`の各セクションと同じ値を使う）

| パラメータ名 | 役割 | デフォルト値（コード / `gnc_params.yaml`） |
|---|---|---|
| `tf_correction.reference_frame` | 自己位置の親フレーム | `iss_body` |
| `tf_correction.target_frame` | 機体フレーム | `body` |
| `trajectory_controller.max_force` | 加速度上限の算出に使う力[N]（軸別、最も厳しい軸の値を使う） | `[0.181, 0.0996, 0.122]` |
| `trajectory_controller.mass` | 加速度上限（区間時間配分・MINCOのheuristic-time）・TOPP-RA・cancel後の制動の質量[kg] | `3.216` |
| `trajectory_controller.inertia` | TOPP-RAの慣性[kg·m²]（等方） | `0.0136` |
| `thrust_allocator.*` | ファンの配置・推力上限（`kj`・`fj_max`・`cg`・`fan_positions`・`fan_vectors`など）。wrenchの範囲の算出と、cancel後の制動の最大減速度の算出に使う | `gnc_params.yaml`の`thrust_allocator`セクション |
| `hover_control.max_force` | Control側の出力の各軸上限[N]。cancel後の制動の減速度をこれ以下に抑える | `0.1` |

`minco_solver.cpp`（MINCO）の質量・慣性はC++の定数（3.216 kg・0.0136 kg·m²）で、上の`trajectory_controller.*`の影響は受けない。

### その他のROS I/Oラッパ

`path_publisher`・`multi_dof_joint_trajectory_publisher`・`checkpoint_publisher`・`ctl_command_action_server`は`console_scripts`登録済みだが、通常は`guidance_node`から利用するライブラリとしての位置づけで、単体`ros2 run`はデバッグ用途のみ（各ファイルの`main()`docstring参照）。

[↑ 目次に戻る](#目次)
