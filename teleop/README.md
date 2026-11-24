# teleop （パッケージ `sobits_intball2_teleop`）

キーボードまたは Joy-Con で機体を今の位置からの相対で動かすためのパッケージです（C++・ament_cmake）。入力から目標の基準を作る backend ノードと、RViz の Panel/Tool、Joy-Con の入力ノードで構成します。

起動方法・操作・設定は，トップの[README.md](../README.md#キーボードテレオペ)を参照してください．

## 構成

```
teleop/
├── include/sobits_intball2_teleop/, src/   # 公開ヘッダと実装（同じ並び）
│   ├── core/      # ROS・Qt に依存しない処理（基準の生成、誤差の上限、入力の排他など）
│   ├── codec/     # backend と Panel が std_msgs/String で交換する JSON 形式
│   ├── backend/   # teleop ノード
│   ├── rviz/      # RViz の Panel・Tool とキー入力
│   └── joycon/    # Joy-Con の入力（ROS 非依存の処理は core/、ノードは nodes/）
├── launch/        # teleop.launch.py、joycon.launch.py
├── config/        # RViz の設定（teleop.rviz）、Joy-Con の設定（joycon_*.yaml、teleop_bridge.yaml）
├── test/          # 単体・結合テスト、期待値の CSV（test/data）、手動の確認（test/manual）
└── rviz_plugins.xml
```

## Joy-Con

Joy-Con（左右 1 本ずつ、縦持ち）で操作できます。パッケージ内の 4 つのノードで構成します（evdev は `linux/input.h` を直接使うので、追加の依存はありません）。

```
Joy-Con → joycon_driver → joycon_input → shared_control → teleop_bridge → /gnc/teleop/key_state → teleop_node
```

起動（teleop_node を先に起動しておく）:

```sh
ros2 launch sobits_intball2_teleop joycon.launch.py
```

| 操作 | ボタン |
| --- | --- |
| 前後・左右 | R スティック |
| 上・下 | `ZR`・`R` |
| 回転（ジャイロ） | `ZL` を押している間、L を傾ける |
| 回転（スティック） | L スティック（ピッチ・ヨー）、`ZL`・`L`（ロール） |
| 回転の方式の切り替え（ジャイロ ⇄ スティック） | `CAPTURE`（L のスクショボタン） |
| 開始・終了 | `+` |
| 非常停止 | `−`（操作の権利がなくても効く） |
| 速度の段階 | 十字キー 上・下 |
| 加速度の段階 | 十字キー 右・左 |
| 入力元の切り替え（Joy-Con ⇄ キーボード） | R の `HOME` を 1 秒長押し |

- 回転の方式は、RViz の Panel の回転の欄に表示されます（`GYRO MODE` / `STICK MODE`）。`/joycon/rotation_mode`（`std_msgs/String`、`gyro` か `stick`、reliable + transient_local）から受け取ります。
- Joy-Con のデバイスを読むために、ホストに udev ルールが必要です（`/etc/udev/rules.d/99-joycon.rules`、vendor 057e・product 2006/2007 の event を 0666）。
- テストは `test_joycon`（gtest）です。Python の実装の出力を記録した CSV（`test/data/joycon_*_golden.csv`）と比べています。
