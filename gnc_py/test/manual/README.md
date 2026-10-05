# test/manual/

シミュレータを動かして手で確認するスクリプト置き場です。pytestのテストではなく、`colcon test`では実行されません。
`gnc_bringup.launch.py` で起動したシミュレータとcontrolが必要で、`python3` で直接実行します。

```sh
export PYTHONPATH="/root/colcon_ws/src/sobits_intball2_gnc/gnc_py:$PYTHONPATH"
python3 test/manual/<script>.py --help
```

各スクリプトの目的・引数・出力は、先頭のdocstringと `--help` を見てください。ここには一覧を書きません。

## 目的別の入口

- `move_to` の検証: `move_to_full_analysis.py`（追従誤差・ファンの飽和・指令と実現のwrenchを1回で出す）
- 自己位置の確認: `get_pose.py`
- 障害物の配置: `spawn_obstacle.py`、`virtual_obstacle.py`
- ホバー・姿勢の安定性: `measure_*`、`log_pose_drift.py`、`analyze_pose_drift.py`

## 守ること

- 自己位置は `TfClient`（`control/ros/tf_client.py`）で取る。生の `tf2_ros.Buffer` を自前で作らない。
- 機体を実際に動かすので、初めて回すときは短い `timeout` を付ける。
