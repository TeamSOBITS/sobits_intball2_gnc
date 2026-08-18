# guidance （Guidance: GNCの「G」）

**未実装（骨格のみ）**。waypoint列から、時間の関数としての滑らかな目標軌道（位置・速度・加速度・目標姿勢）を生成する、ROS非依存のロジック層を作る予定です。詳細は`docs/main_plan.md`のPhase 2を参照してください（このファイルは`docs/`同様プッシュ対象外のため、社内共有時は別途内容を転記してください）。

```
guidance/
├── ros/                        # 現状は空。ROS I/Oが必要になった場合に備えた置き場（YAGNIで中身は未着手）
└── utils/                      # ROS非依存のロジック
    ├── min_snap.py              # waypoints + 時間配分 -> 各軸・区間の多項式係数
    ├── trajectory.py            # Trajectoryクラス: sample(t) -> (p, v, a, q_des)
    └── attitude_reference.py    # v_des(t) -> q_des(t)、低速時の閾値処理
```

実装が入り次第、このREADMEも`control/README.md`・`navigation/README.md`と同水準の内容に更新します。
