# Control

目標軌道を追従する force/torque を計算し、8基のファンへ配分するモジュールです。IMU姿勢制御をベースに、必要に応じてTF補正・経路チェックポイント・軌道追従を重ねます。

## 構成

```
control/
├── control.py    # 統括ノード（唯一の rclpy ノード）
├── ros/          # ROS 入出力ラッパ
│   ├── fan_duty_publisher.py
│   ├── imu_subscriber.py
│   ├── pose_array_subscriber.py                  # /gnc/checkpoints 購読
│   └── multi_dof_joint_trajectory_subscriber.py   # /gnc/trajectory_setpoint 購読
└── utils/        # ROS非依存のロジック（単体テスト可能）
    ├── quat_math.py
    ├── pose_control_law.py                       # (target, current) -> (force, torque)の純粋な誤差則。PoseCorrector等から再利用される
    ├── hover_law.py
    ├── pose_corrector.py
    ├── hover_controller.py       # 全体を束ねるオーケストレーション
    ├── trajectory_controller.py
    ├── thrust_allocator.py
    ├── translation_direction_controller.py
    └── singleton_lock.py
```

TF自己位置取得（`TfClient`）は`control/`と`guidance/`で共有するため`common/ros/tf_client.py`にあります。`common/ros/pose_relay_client.py`は同じ`iss_body <- body`フィードバックを、ブリッジ負荷対策として専用の間引きトピック経由で取得する`TfClient`互換の代替実装（`control.py`のみで使用、`docs/recording_cpu_load_control_degradation.md`参照）。


## ホバリング制御（起動方法）

```sh
# パラメータファイルを与えて起動（TF補正が有効）
ros2 run sobits_intball2_gnc control --ros-args \
  --params-file $(ros2 pkg prefix sobits_intball2_gnc)/share/sobits_intball2_gnc/config/gnc_params.yaml

# パラメータファイルなしで起動（純IMUホバリング）
ros2 run sobits_intball2_gnc control
```

- モードは`config/gnc_params.yaml`の`hover_control.mode`で切替（`imu`=純IMU / `tf_imu`=TF補正あり、既定）。
- IMUのみの場合は絶対参照がなく、姿勢・位置はゆっくりドリフトします。TF補正はこのドリフトを抑えます。
- TFの配信が`timeout`秒止まると自動的に純IMUホバリングへ縮退し、復帰すると再度捕捉します。
- TF（`iss_body`<-`body`）はシミュレータ限定のオラクルで、実機には存在しません。

## ファン直接制御

Navigation OFFの状態で、`/ctl/duty`へ直接publishして8基のファンを個別に駆動できます。

```sh
ros2 run sobits_intball2_gnc fan_duty_publisher --help
```

## 経路チェックポイントIF

`/gnc/checkpoints`（`geometry_msgs/PoseArray`、frame_idは`iss_body`）に配列をpublishすると、先頭のポーズが保持目標になります。空配列で現在位置を再捕捉します。

```sh
ros2 topic pub --once /gnc/checkpoints geometry_msgs/msg/PoseArray \
  "{header: {frame_id: iss_body}, poses: [{position: {x: 0.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}]}"
```

## 軌道追従IF

`/gnc/trajectory_setpoint`（`trajectory_msgs/MultiDOFJointTrajectory`）に目標位置・速度・加速度をpublishすると、`TrajectoryController`がフィードフォワード＋フィードバックで追従します。前提は`hover_control.mode: tf_imu`。

setpointが届いている間はcheckpointホールドより優先され、途切れると自動的にcheckpointホールドへ戻ります。

Guidanceは未実装のため、現状は`test/manual/`のスタンドインスクリプト（`send_trajectory.py`等）で動作確認します（詳細: `test/manual/README.md`）。


## パラメータ

パラメータは全て[config/gnc_params.yaml](../../config/gnc_params.yaml)で管理します。

分類の考え方（固定/動的）の詳細は[docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md](../../docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md)を参照。

### 固定パラメータ（起動時のみ、実行中は変更不可）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `hover_control.mode` | ホバリング方式（`imu`=純IMU / `tf_imu`=TF補正あり） | `tf_imu` |
| `hover_control.control_rate` | 制御ループ周期 [Hz] | `50.0` |
| `tf_correction.reference_frame` | 自己位置の親フレーム | `iss_body` |
| `tf_correction.target_frame` | 機体フレーム | `body` |
| `tf_correction.poll_rate` | TF取得レート [Hz] | `50.0` |
| `tf_correction.smooth_window` | 平滑化ウィンドウ幅 [サンプル数] | `5` |
| `tf_correction.smooth_sigma` | 平滑化のガウス重みσ | `2.0` |
| `tf_correction.checkpoint_topic` | チェックポイント配列のトピック名 | `/gnc/checkpoints` |
| `trajectory_controller.mass` | 機体質量（フィードフォワード用）[kg] | `4.5` |
| `thrust_allocator.kj` | 推力→duty変換係数 | `4.082482905` |
| `thrust_allocator.fj_max` | ファン1基あたりの最大推力 [N] | `0.06` |
| `thrust_allocator.cg` | 重心位置 [m] | `[0.001489, 0.001363, 0.000249]` |
| `thrust_allocator.fan_positions` | 8ファンの搭載位置（機体座標系）[m] | `config/gnc_params.yaml`参照 |
| `thrust_allocator.fan_vectors` | 8ファンの推力方向単位ベクトル | `config/gnc_params.yaml`参照 |
| `control.status_log_period` | ステータスログの出力間隔 [s]（0で無効） | `2.0` |

### 動的パラメータ（`ros2 param set`で実行中に変更可能）

| パラメータ名 | 役割 | デフォルト値 |
|---|---|---|
| `hover_control.kd_w` | 角速度ダンピングゲイン [Nm/(rad/s)] | `[0.02, 0.02, 0.02]` |
| `hover_control.kp_a` | 加速度外乱抑制ゲイン [N/(m/s²)] | `[0.5, 0.5, 0.5]` |
| `hover_control.deadband_w` | ジャイロ不感帯 [rad/s] | `0.01` |
| `hover_control.deadband_a` | 加速度残差不感帯 [m/s²] | `0.02` |
| `hover_control.acc_bias_alpha` | 加速度バイアス推定のEMA係数 | `0.01` |
| `hover_control.max_force` | IMU則の出力力クランプ [N] | `0.1` |
| `hover_control.max_torque` | IMU則の出力トルククランプ [Nm] | `0.02` |
| `tf_correction.kp_pos` | 位置誤差→力ゲイン [N/m] | `[0.89, 0.89, 0.89]` |
| `tf_correction.kd_pos` | 速度→力ゲイン [N/(m/s)] | `[3.6, 3.6, 3.6]` |
| `tf_correction.kp_att_align` | 姿勢誤差→トルクゲイン、align中 [Nm] | `[0.01, 0.01, 0.01]` |
| `tf_correction.kd_att_align` | 相対角速度誤差→トルクゲイン、align中 [Nm/(rad/s)] | `[0.0, 0.0, 0.0]` |
| `tf_correction.kp_att_hold` | 姿勢誤差→トルクゲイン、hold中 [Nm] | `[0.01, 0.01, 0.01]` |
| `tf_correction.kd_att_hold` | 相対角速度誤差→トルクゲイン、hold中 [Nm/(rad/s)] | `[0.0, 0.0, 0.0]` |
| `tf_correction.align_tolerance_deg` | align→hold切替の角度閾値 [deg] | `3.0` |
| `tf_correction.align_settle_time` | 角度閾値内が連続してこの時間続いたらhold gainに切替 [s] | `0.5` |
| `tf_correction.align_gain_max_duration` | align gainを使う時間の保険上限 [s] | `30.0` |
| `tf_correction.vel_filter_alpha` | 速度推定のEMA係数 | `0.3` |
| `tf_correction.att_filter_alpha` | 角速度誤差推定のEMA係数 | `1.0` |
| `tf_correction.max_corr_force` | TF補正の出力力クランプ [N] | `0.05` |
| `tf_correction.max_corr_torque` | TF補正の出力トルククランプ [Nm] | `0.01` |
| `tf_correction.timeout` | TFステール判定の閾値 [s] | `1.0` |
| `trajectory_controller.kp_pos` | 軌道追従の位置誤差→力ゲイン [N/m] | `[0.89, 0.89, 0.89]` |
| `trajectory_controller.kd_pos` | 軌道追従の速度誤差→力ゲイン [N/(m/s)] | `[3.6, 3.6, 3.6]` |
| `trajectory_controller.vel_filter_alpha` | 軌道追従の速度推定EMA係数 | `0.3` |
| `trajectory_controller.max_force` | 軌道追従の出力力クランプ [N] | `0.1` |
| `trajectory_controller.kp_att` | 軌道追従の姿勢誤差→トルクゲイン [Nm] | `[0.60, 0.60, 0.60]` |
| `trajectory_controller.kd_att` | 軌道追従の角速度誤差→トルクゲイン [Nm/(rad/s)] | `[0.20, 0.20, 0.20]` |
| `trajectory_controller.att_filter_alpha` | 軌道追従の角速度誤差推定EMA係数 | `1.0` |
| `trajectory_controller.max_torque` | 軌道追従の出力トルククランプ [Nm] | `0.32` |
| `trajectory_controller.timeout` | setpointステール判定の閾値 [s] | `0.2` |
| `thrust_allocator.force_weight_ref` | 配分の力チャンネル重み参照値 [N] | `0.1` |
| `thrust_allocator.torque_weight_ref` | 配分のトルクチャンネル重み参照値 [Nm] | `0.32` |

`translation_direction_control.*`（`force_magnitude`/`max_force`/`control_rate`）は`TranslationDirectionController`が宣言・保持するパラメータだが、現状どのノードにも配線されていないため上表からは省略（詳細: `docs/main_plan.md`）。
