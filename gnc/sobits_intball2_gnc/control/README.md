# Control

guidance_nodeから受け取った目標（軌道のsetpoint、止まる場所のcheckpoint）を追従するforce/torqueを計算し、8基のファンへ配分するモジュールです。IMUによる姿勢制御をベースに、TFの自己位置で補正します。

## 目次

- [起動](#launch)
- [入出力](#io)
- [デバッグ用](#debug)
- [構成](#構成)
- [パラメータ](#parameters)

<a id="launch"></a>
## 起動

```sh
ros2 launch sobits_intball2_gnc control.launch.py
```

`control_node`だけを起動し、[config/gnc_params.yaml](../../config/gnc_params.yaml)を読みます（別のファイルは`params_file:=<path>`）。
何も届いていない間は、起動した位置・姿勢でホバリングします。切り替えはyamlで行います（どちらも起動時のみ）:

| パラメータ | 値 |
|---|---|
| `hover_control.mode` | `tf_imu`（既定、TFで補正）/ `imu`（IMUだけ。絶対参照がないので位置・姿勢はゆっくりドリフトする） |
| `control.thrust_allocation` | `builtin`（既定、このノードが推力配分して`/ctl/duty`を出す）/ `jaxa_fsm`（`/ctl/wrench`を出し、推力配分はJAXAの`fsm`に任せる） |

- TF（`iss_body`<-`body`）はシミュレータ限定のオラクルで、実機にはない。TFが`tf_correction.timeout`秒止まるとIMUだけのホバリングに落ち、戻ると再び補正する
- `jaxa_fsm`はJAXAの`ctl_only`がSTAND_BY（NAV_OFF）のときだけ`/ctl/wrench`を出す（下の[出力](#io)）

[↑ 目次に戻る](#目次)

<a id="io"></a>
## 入出力

### 入力

| トピック名 | 型 | 送り元 | 説明 |
|---|---|---|---|
| `/imu/imu` | `ib2_msgs/IMU` | シム | 角速度・加速度 |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | シム | `iss_body <- body`の自己位置（`TfClient`） |
| `/gnc/trajectory_setpoint` | `trajectory_msgs/MultiDOFJointTrajectory` | guidance_node | 移動中の目標（位置・速度・加速度・姿勢・角速度）。届いている間はcheckpointより優先し、`trajectory_controller.timeout`秒途切れるとcheckpointの保持に戻る |
| `/gnc/checkpoints` | `geometry_msgs/PoseArray` | guidance_node | 止まる場所（frame_id: `iss_body`）。先頭のポーズを保持する。空配列で今の位置を保持し直す |
| `/ctl/status` | `ib2_msgs/CtlStatus` | JAXAの`ctl_only` | `jaxa_fsm`のときだけ。`ctl_only`が止まっているかの確認 |

### 出力

| トピック名 | 型 | 説明 |
|---|---|---|
| `/ctl/duty` | `std_msgs/Float64MultiArray` | 8基のファンへのduty（`builtin`のときだけ） |
| `/ctl/wrench` | `geometry_msgs/WrenchStamped` | 推力配分に渡す合計のforce/torque（`jaxa_fsm`のときだけ）。bridgeでJAXAの`fsm`に届く |
| `/ctl/wrench_correction` | `geometry_msgs/WrenchStamped` | 診断用: 制御則が要求する補正force/torque（上限で切る前） |
| `/ctl/wrench_total` | `geometry_msgs/WrenchStamped` | 診断用: 推力配分に渡す合計のforce/torque |
| `/ctl/wrench_achieved` | `geometry_msgs/WrenchStamped` | 診断用: dutyで実際に出せるforce/torque（`builtin`のときだけ） |

JAXAの`fsm`はNAV_OFFでも止まらず、`/ctl/wrench`が届けば`/ctl/duty`を出す。そのため`builtin`では`/ctl/wrench`を出さず、
`jaxa_fsm`でも`/ctl/status`が3秒以内に届いていて`ctl_only`が自分のwrenchを出していない状態のときだけ出す。

[↑ 目次に戻る](#目次)

<a id="debug"></a>
## デバッグ用

普段は使わない。

```sh
# 止まる場所を手で送る（普段はguidance_nodeが送る）
ros2 topic pub --once /gnc/checkpoints geometry_msgs/msg/PoseArray \
  "{header: {frame_id: iss_body}, poses: [{position: {x: 0.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}]}"

# checkpoint配列を1つ先へ進める
ros2 service call /gnc/advance_checkpoint std_srvs/srv/Trigger

# control_nodeを止めた状態で、ファンを直接回す（NAV_OFFで）
ros2 run sobits_intball2_gnc fan_duty_publisher --help
```

`gnc/test/manual/`の検証スクリプトは[gnc/test/manual/README.md](../../test/manual/README.md)を参照。

[↑ 目次に戻る](#目次)

<a id="構成"></a>
## 構成

```
control/
├── control.py    # 統括ノード（唯一の rclpy ノード）。ros/のI/OラッパとutilsのROS非依存ロジックをDIで束ね、
│                 # IMU(+TFポーズ) -> HoverController -> ThrustAllocator -> FanDutyPublisher の制御ループを回す
├── ros/          # ROS 入出力ラッパ（Nodeは継承せず、渡されたnodeにpub/subをぶら下げるだけ）
│   ├── ctl_status_subscriber.py                   # /ctl/status（JAXA ctl_onlyの状態）購読。jaxa_fsm時の安全確認用
│   ├── fan_duty_publisher.py                      # /ctl/duty へ8基分のduty配列をpublish。負推力は出せないため[0,1]にクランプ
│   ├── imu_subscriber.py                          # /imu/imu（ib2_msgs/IMU）を購読し最新のジャイロ・加速度を保持
│   ├── pose_array_subscriber.py                   # /gnc/checkpoints（止まる場所の配列）購読
│   ├── multi_dof_joint_trajectory_subscriber.py   # /gnc/trajectory_setpoint（軌道追従の目標点）購読
│   └── wrench_publisher.py                        # /ctl/wrench_correction・/ctl/wrench_total・/ctl/wrench_achieved へ
│                                                   # 要求/合成/実現wrenchをpublish（可観測性強化用）。jaxa_fsm時は/ctl/wrenchも
└── utils/        # ROS非依存のロジック（単体テスト可能）
    ├── quat_math.py                        # クォータニオン/ベクトルの純粋関数群（回転・誤差角度など）
    ├── pose_control_law.py                 # (target, current) -> (force, torque)の純粋な誤差則。PoseCorrector等から再利用される
    ├── hover_law.py                        # IMUのみのホバリング制御則（角速度ダンピング＋加速度外乱抑制）
    ├── pose_corrector.py                   # TFポーズに基づくホバリング補正（TF取得・平滑化・チェックポイント保持）
    ├── hover_controller.py                 # 全体を束ねるオーケストレーション（HoverLaw/PoseCorrector/TrajectoryController/ThrustAllocatorを結線）
    ├── trajectory_controller.py            # Guidanceの移動目標(p_des/v_des/a_des/q_des)へのフィードフォワード＋フィードバック追従
    ├── thrust_allocator.py                 # body系wrench -> 8基の非負ファンduty配分（拘束付き最小二乗/ミニマックス）
    ├── translation_direction_controller.py # 並進のみの方向ベクトル指令をwrenchに変換（現状未配線、将来のテレオペ用部品）
    └── singleton_lock.py                   # flockによるcontrol_nodeの多重起動防止
```

TF自己位置取得（`TfClient`）は`control/`と`guidance/`で共有するため`common/ros/tf_client.py`にあります。
`common/ros/pose_relay_client.py`（間引きトピック`/gnc/body_pose_raw`経由の`TfClient`互換の代替）は現在どのノードからも使われていません。

[↑ 目次に戻る](#目次)

<a id="parameters"></a>
## パラメータ

パラメータは全て[config/gnc_params.yaml](../../config/gnc_params.yaml)で管理します。

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
| `trajectory_controller.mass` | 機体質量（フィードフォワード用）[kg] | `3.216` |
| `trajectory_controller.inertia` | 機体慣性、等方性（フィードフォワード用）[kg*m^2] | `0.0136` |
| `thrust_allocator.kj` | 推力→duty変換係数 | `4.082482905` |
| `thrust_allocator.fj_max` | ファン1基あたりの最大推力 [N] | `0.06` |
| `thrust_allocator.cg` | 重心位置 [m] | `[0.001489, 0.001363, 0.000249]` |
| `thrust_allocator.fan_positions` | 8ファンの搭載位置（機体座標系）[m] | `config/gnc_params.yaml`参照 |
| `thrust_allocator.fan_vectors` | 8ファンの推力方向単位ベクトル | `config/gnc_params.yaml`参照 |
| `control.status_log_period` | ステータスログの出力間隔 [s]（0で無効） | `2.0` |
| `control.thrust_allocation` | 推力配分をどこでするか。`builtin`（このノード）/ `jaxa_fsm`（`/ctl/wrench`を出してJAXAの`fsm`に任せる） | `builtin` |

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
| `tf_correction.kp_pos` | 位置誤差→力ゲイン [N/m] | `[0.635, 0.635, 0.635]` |
| `tf_correction.kd_pos` | 速度→力ゲイン [N/(m/s)] | `[2.573, 2.573, 2.573]` |
| `tf_correction.kp_att_align` | 姿勢誤差→トルクゲイン、align中 [Nm] | `[0.20, 0.20, 0.20]` |
| `tf_correction.kd_att_align` | 相対角速度誤差→トルクゲイン、align中 [Nm/(rad/s)] | `[0.2816, 0.2816, 0.2816]` |
| `tf_correction.kp_att_hold` | 姿勢誤差→トルクゲイン、hold中 [Nm] | `[0.20, 0.20, 0.20]` |
| `tf_correction.kd_att_hold` | 相対角速度誤差→トルクゲイン、hold中 [Nm/(rad/s)] | `[0.1408, 0.1408, 0.1408]` |
| `tf_correction.align_tolerance_deg` | align→hold切替の角度閾値 [deg] | `3.0` |
| `tf_correction.align_settle_time` | 角度閾値内が連続してこの時間続いたらhold gainに切替 [s] | `0.5` |
| `tf_correction.align_gain_max_duration` | align gainを使う時間の保険上限 [s] | `30.0` |
| `tf_correction.vel_filter_alpha` | 速度推定のEMA係数 | `0.3` |
| `tf_correction.att_filter_alpha` | 角速度誤差推定のEMA係数 | `0.3` |
| `tf_correction.max_corr_force` | TF補正の出力力クランプ [N] | `0.05` |
| `tf_correction.max_corr_torque` | TF補正の出力トルククランプ [Nm] | `0.3` |
| `tf_correction.timeout` | TFステール判定の閾値 [s] | `1.0` |
| `tf_correction.torque_direction_preserving` | 大角度補正時、各軸独立クランプの代わりに指令回転軸の方向を維持したままクランプする | `false` |
| `trajectory_controller.kp_pos` | 軌道追従の位置誤差→力ゲイン [N/m] | `[0.635, 0.635, 0.635]` |
| `trajectory_controller.kd_pos` | 軌道追従の速度誤差→力ゲイン [N/(m/s)] | `[2.573, 2.573, 2.573]` |
| `trajectory_controller.max_force` | 軌道追従の出力力クランプ [N]（軸別、ファンモデル理論値） | `[0.181, 0.0996, 0.122]` |
| `trajectory_controller.kp_att` | 軌道追従の姿勢誤差→トルクゲイン [Nm] | `[0.20, 0.20, 0.20]` |
| `trajectory_controller.kd_att` | 軌道追従の角速度誤差→トルクゲイン [Nm/(rad/s)] | `[0.0626, 0.0626, 0.0626]` |
| `trajectory_controller.att_filter_alpha` | 軌道追従の角速度誤差推定EMA係数 | `1.0` |
| `trajectory_controller.max_torque` | 軌道追従の出力トルククランプ [Nm] | `0.32` |
| `trajectory_controller.timeout` | setpointステール判定の閾値 [s] | `0.2` |
| `trajectory_controller.torque_direction_preserving` | 大角度補正時、各軸独立クランプの代わりに指令回転軸の方向を維持したままクランプする | `false` |
| `thrust_allocator.force_weight_ref` | 配分の力チャンネル重み参照値 [N] | `0.1` |
| `thrust_allocator.torque_weight_ref` | 配分のトルクチャンネル重み参照値 [Nm] | `0.32` |
| `thrust_allocator.torque_axis_balance` | 軸ごとの物理トルク上限で重み付け（非推奨、下記参照） | `false` |
| `thrust_allocator.minimax_objective` | L2最小二乗の代わりにミニマックス目的関数を使用 | `true` |

`thrust_allocator.torque_axis_balance`は**非推奨**: duty飽和緩和を狙って`true`でsim検証したが、
同一ルート・同一条件で
duty≥0.95飽和頻度は`false`と同等（48%→48.3%、改善なし）、`t_des`最大は悪化（0.30Nm→0.35Nm）、
所要時間も悪化（約51〜54秒→約62秒）した。

`thrust_allocator.minimax_objective`は要求ピーク値（`t_des`/`f_des`最大）を下げる効果があり
（`t_des`最大0.30Nm→0.22Nm）、`wrench_envelope_safety_margin`（guidance側）
との併用でduty飽和頻度を最も改善できたためデフォルト`true`に変更済み。

`translation_direction_control.*`（`force_magnitude`/`max_force`/`control_rate`）は`TranslationDirectionController`が宣言・保持するパラメータだが、現状どのノードにも配線されていないため上表からは省略。

[↑ 目次に戻る](#目次)
