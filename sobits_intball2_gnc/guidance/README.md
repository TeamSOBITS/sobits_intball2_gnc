# guidance （Guidance: GNCの「G」）

waypoint列から、時間の関数としての滑らかな目標軌道（位置・速度・加速度・目標姿勢）を生成するモジュールです。**`min_snap.py`のコアロジックのみ未実装**（別担当者、詳細: `docs/main_plan.md`）で、それ以外は実装済みです。

## 構成

```
guidance/
├── global_planner/                       # 大域経路計画（waypoint列の生成）
│   ├── base_global_planner.py                # 共通インターフェース
│   ├── astar_planner.py
│   └── rrt_planner.py
├── ros/                                  # ROS 入出力ラッパ
│   ├── path_publisher.py                     # nav_msgs/Path をRVizへ可視化publish
│   ├── multi_dof_joint_trajectory_publisher.py  # /gnc/trajectory_setpoint へ発行（Control側が購読）
│   └── ctl_command_action_server.py          # ib2_msgs/action/CtlCommand（目標姿勢へのgoal駆動）
└── utils/                                # ROS非依存のロジック
    ├── polynomial.py                         # 多項式（微分）評価
    ├── trajectory.py                         # Trajectory: sample(t) -> (p, v, a, q_des)
    ├── attitude_reference.py                 # v_des(t) -> q_des(t)（進行方向を向く姿勢参照）
    └── min_snap.py                           # waypoints + 時間配分 -> 多項式係数（コアロジック未実装）
```

## 現状の使い方

Control側の受け口（`/gnc/trajectory_setpoint`、`/gnc/checkpoints`）は実装済みなので、Guidance本体（`min_snap.py`）が未完成の間は`test/manual/`のスタンドインスクリプトで軌道を直接publishして動作確認します（詳細: `test/manual/README.md`）。

`path_publisher`・`multi_dof_joint_trajectory_publisher`・`ctl_command_action_server`は`console_scripts`登録済みですが、通常はGuidanceノードから利用するライブラリとしての位置づけで、単体で`ros2 run`する想定ではありません。
