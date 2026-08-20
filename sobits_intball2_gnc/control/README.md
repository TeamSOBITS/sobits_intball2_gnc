# control （Control: GNCの「C」）

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
    ├── pose_control_law.py
    ├── hover_law.py
    ├── pose_corrector.py
    ├── hover_controller.py       # 全体を束ねるオーケストレーション
    ├── trajectory_controller.py
    ├── thrust_allocator.py
    ├── direction_controller.py
    └── singleton_lock.py
```

TF自己位置取得（`TfClient`）は`control/`と`guidance/`で共有するため`common/ros/tf_client.py`にあります。

パラメータは全て[config/gnc_params.yaml](../../config/gnc_params.yaml)で管理します。

## ファン直接制御

Navigation OFFの状態で、`/ctl/duty`へ直接publishして8基のファンを個別に駆動できます。

```sh
ros2 run sobits_intball2_gnc fan_duty_publisher --help
```

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
