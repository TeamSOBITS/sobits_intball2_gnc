# teleop （パッケージ `sobits_intball2_teleop`）

キーボードで機体を今の位置からの相対で動かすためのパッケージです（C++・ament_cmake）。キー入力から目標の基準を作る backend ノードと、RViz の Panel/Tool で構成します。

起動方法・操作・設定は，トップの[README.md](../README.md#キーボードテレオペ)を参照してください．

## 構成

```
teleop/
├── include/sobits_intball2_teleop/, src/   # 公開ヘッダと実装（同じ並び）
│   ├── core/      # ROS・Qt に依存しない処理（基準の生成、誤差の上限、入力の排他など）
│   ├── codec/     # backend と Panel が std_msgs/String で交換する JSON 形式
│   ├── backend/   # teleop ノード
│   └── rviz/      # RViz の Panel・Tool とキー入力
├── launch/        # teleop.launch.py
├── config/        # RViz の設定（teleop.rviz）
├── test/          # 単体・結合テスト、期待値の CSV（test/data）、手動の確認（test/manual）
└── rviz_plugins.xml
```
