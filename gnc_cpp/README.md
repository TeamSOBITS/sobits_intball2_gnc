# gnc_cpp （パッケージ `sobits_intball2_gnc_cpp`）

`gnc_py`（`sobits_intball2_gnc`）から呼ぶ計算部分のC++実装と、そのpybind11拡張です。ROSに依存するのは `perception/virtual_camera_node.cpp`（仮想カメラのノード）だけで、それ以外はROS非依存です。

## 構成

`include/sobits_intball2_gnc_cpp/`（公開ヘッダ）と `src/`（実装）は同じ並びです。

```
gnc_cpp/
├── include/sobits_intball2_gnc_cpp/, src/
│   ├── common/        # so3.hpp（回転ベクトルまわり）、trace.hpp（環境変数で有効になる診断出力）
│   ├── mapping/       # occupancy_grid（静的層＋深度で更新する層の占有格子）、octomap_io（OctoMapの読み込み・切り出し）
│   ├── perception/    # depth_renderer（占有格子から深度画像を生成）、mesh_voxels（.daeメッシュのボクセル化）、
│   │                  # virtual_camera_node（仮想深度カメラのROSノード）
│   ├── guidance/
│   │   ├── minco/     # MINCO軌道生成（minco_planner、制約点、時間配分、wrench envelope）
│   │   ├── rebound/   # 衝突した区間を押し出す rebound（EGO-Planner系）
│   │   ├── search/    # global_a_star（出発前の共通経路）、local_a_star（reboundの誘導経路）
│   │   └── jaxa/      # JAXA方式の局所経路計画（OMPL RRT*）
│   └── control/       # JAXA制御器（位置・姿勢・推力配分）の移植
├── python/            # pybind11バインディング（bindings.cpp、jaxa_planner_bindings.cpp）
├── config/            # wrench_envelope.csv
├── cmake/             # config.hpp.in
├── test/              # C++の単体テストと実験用実行ファイル
└── third_party/       # GCOPTER（install.shで取得）
```

## Pythonから使う

`python/` のバインディングが、Pythonモジュール `sobits_intball2_gnc_cpp` として公開します。`gnc_py` 側は `import sobits_intball2_gnc_cpp` で使います。公開されるものは `python/*.cpp` を見てください。

## 外部由来のコード

- `control/jaxa_*` は [JAXA Int-Ball2 simulator](https://github.com/jaxa/int-ball2_simulator)（Apache-2.0）の `ctl_only`・`fsm` の移植です。ROS型を外し、パラメータはコンストラクタで受けます。
- `guidance/minco/` と `guidance/rebound/` は GCOPTER（MIT）と EGO-Planner の考え方に基づきます。GCOPTERのヘッダは `third_party/gcopter` に置きます。
- 出典とライセンスの一覧は、ルートの[README.md](../README.md)にあります。
