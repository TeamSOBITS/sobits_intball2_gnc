<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# sobits_intball2_gnc

<!-- 目次 -->
<details>
  <summary>目次</summary>
  <ol>
    <li>
      <a href="#概要">概要</a>
    </li>
    <li>
      <a href="#パッケージ構成">パッケージ構成</a>
    </li>
    <li>
      <a href="#セットアップ">セットアップ</a>
      <ul>
        <li><a href="#環境条件">環境条件</a></li>
        <li><a href="#インストール方法">インストール方法</a></li>
      </ul>
    </li>
    <li><a href="#実行操作方法">実行・操作方法</a></li>
    <li><a href="#マイルストーン">マイルストーン</a></li>
  </ol>
</details>

## 概要

Int-Ball2 シミュレータでロボットを自律移動させるためのパッケージです．
ROS2 Humble に対応しています．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## パッケージ構成

制御系は `ros/`（ROS 入出力ラッパ）・`utils/`（ROS 非依存のロジック）・`control.py`（統括ノード）の3層に分離しています．ROS の I/O とロジックと設定読み込みを分け，各ロジックを単体テスト可能にしています．制御系のノードは統括ノード `control.py` の**唯一の1ノード**で動作します（1ファイル1ノード）．

```
sobits_intball2_gnc/
├── config/
│   └── gnc_params.yaml              # GNC パラメータ（ROS2 param 形式：ファン配置・推力モデル・制御ゲイン）
├── maps/
│   └── iss_location.yaml            # 登録済みロケーション一覧（27地点）
├── sobits_intball2_gnc/
│   ├── navigation/
│   │   ├── location_broadcaster.py  # YAML のロケーションを TF にパブリッシュ（10Hz）
│   │   └── location_setting.py      # ロケーション登録 GUI（Tkinter + Zenity）
│   └── control/
│       ├── control.py               # 統括ノード（唯一の rclpy ノード）：入出力とロジックを結線し制御ループを回す
│       ├── ros/                     # ROS 入出力ラッパ（Node を継承せず、渡されたノードに購読/配信を張る）
│       │   ├── fan_duty_publisher.py    # 8基ファンの duty を /ctl/duty へ配信
│       │   ├── imu_subscriber.py        # /imu/imu（ib2_msgs/IMU）を購読
│       │   ├── tf_client.py             # TF からの自己位置取得（iss_body <- body）
│       │   └── path_subscriber.py       # /gnc/checkpoints（PoseArray）を購読
│       └── utils/                   # ROS 非依存のロジック（素値コンストラクタで単体テスト可能）
│           ├── thrust_allocator.py      # wrench（力・トルク）→ 8ファン duty 配分
│           ├── hover_controller.py      # IMU ホバリング則 + Nav 補正（ドリフト抑制）
│           └── direction_controller.py  # 進行方向ベクトル → wrench（将来の自由経路移動用）
├── test/                            # 各ロジックの単体テスト（ROS 不要）
├── package.xml
├── setup.py
└── setup.cfg
```

> [!NOTE]
> パラメータは **ROS2 パラメータ**として [config/gnc_params.yaml](config/gnc_params.yaml) で管理します．各パラメータは、それを使うモジュール自身が宣言・取得するため，モジュール単体でも構築・テストできます（将来の dynamic parameter 対応も見据えた構成）．ファン幾何は ROS2 param が「マップの配列」を表現できないため，`fan_positions`／`fan_vectors` のフラットな数値配列で保持します．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## セットアップ

### 環境条件

まず，以下の環境を整えてから，次のインストール方法に進んでください．

| System  | Version |
| --- | --- |
| Ubuntu | 22.04 (Jammy Jellyfish) |
| ROS    | Humble Hawksbill |
| Python | 3.10 |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### インストール方法

1. ROS2 ワークスペースの `src` フォルダに移動します．
   ```sh
   cd ~/colcon_ws/src/
   ```
2. 本レポジトリをクローンします．
   ```sh
   git clone -b humble-devel https://github.com/TeamSOBITS/sobits_intball2_gnc.git
   ```
3. レポジトリの中へ移動します．
   ```sh
   cd sobits_intball2_gnc
   ```
4. 依存パッケージをインストールします．
   ```sh
   bash install.sh
   ```
5. パッケージをビルドします．
   ```sh
   cd ~/colcon_ws/
   colcon build --packages-select sobits_intball2_gnc
   source ~/colcon_ws/install/setup.bash
   ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 実行・操作方法

はじめに Int-Ball2 シミュレータを起動する．**Navigation は OFF のままで構いません**（本パッケージは自己位置を TF から取得します）．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 地点登録方法

1. YAML に登録された地点を TF としてパブリッシュする
   ```sh
   ros2 run sobits_intball2_gnc location_broadcaster
   ```
   - `maps/iss_location.yaml` の地点を `iss_body` 座標系の TF フレームとして 10Hz で配信します
   - YAML ファイルの変更をリアルタイムで検知し，再起動なしに反映します

2. ロケーション登録 GUI を起動する
   ```sh
   ros2 run sobits_intball2_gnc location_setting
   ```
   - Zenity のファイル選択ダイアログが開きます（デフォルト: [iss_location.yaml](maps/iss_location.yaml)）
   - シミュレータ内でロボットを登録したい地点・姿勢に移動させ，GUI で地点名を入力して **SNAP CURRENT POS** を押すと登録されます
   - 登録済みの地点の削除・リネームも GUI から行えます

### ファン直接制御方法
Navigationを使わず，`/ctl/duty`へ直接publishして8基のファンを個別に駆動できます．自律移動・ホバリングをユーザプログラムで実装するための最低レベル制御です（`ros/fan_duty_publisher.py` の手動テスト用 CLI）．
```sh
ros2 run sobits_intball2_gnc fan_duty_publisher [引数]
```

> [!NOTE]
> - ファン番号は `1`〜`8`，デューティ比は `0.0`〜`1.0`（範囲外は自動でクランプ）．
> - 逆回転（逆推力）はできません．負のdutyを送るとファンが停止するだけです．逆方向の力が必要な場合は逆向きペアのファン（fan1↔fan8, fan2↔fan5, fan3↔fan6, fan4↔fan7）を駆動してください．
> - Navigationが**OFF**の状態で使用してください（ON時は制御器と競合します）．

#### 引数一覧

| 引数 | 説明 | デフォルト値 |
| --- | --- | --- |
| `--fan` | 制御するファン番号（1-8）。`--duty`と併用 | なし |
| `--duty` | デューティ比（0.0-1.0）。`--fan`と併用 | `0.0` |
| `--set` | ファン毎に指定（`FAN:DUTY`形式を複数）例: `1:0.5 3:0.2` | なし |
| `--all` | 全8基を同一デューティ比に設定 | なし |
| `--duration` | publishを継続する秒数 | `1.0` |

> `--fan` / `--set` / `--all` は排他です（いずれか1つを指定）．引数なしで起動するとヘルプを表示します．

#### 使用例

```sh
# fan1 を duty=0.5 で 2秒間 駆動
ros2 run sobits_intball2_gnc fan_duty_publisher --fan 1 --duty 0.5 --duration 2

# ファン毎に個別指定（fan1=0.5, fan3=0.2）
ros2 run sobits_intball2_gnc fan_duty_publisher --set 1:0.5 3:0.2 --duration 2

# 全ファンを duty=0.3 で 1秒間 駆動
ros2 run sobits_intball2_gnc fan_duty_publisher --all 0.3 --duration 1

# 逆向きペアで往復（並進をおおよそ初期位置に戻す: 押す→2倍戻す→止める）
ros2 run sobits_intball2_gnc fan_duty_publisher --fan 1 --duty 0.3 --duration 1 && \
ros2 run sobits_intball2_gnc fan_duty_publisher --fan 8 --duty 0.3 --duration 2 && \
ros2 run sobits_intball2_gnc fan_duty_publisher --fan 1 --duty 0.3 --duration 1
```

他のノードから利用する場合は，`FanDutyPublisher`（ROS 入出力ラッパ）に自分の `rclpy` ノードを渡して使えます（Node は継承しません）．
```python
import rclpy
from rclpy.node import Node
from sobits_intball2_gnc.control.ros.fan_duty_publisher import FanDutyPublisher

rclpy.init()
node = Node("my_node")
fan = FanDutyPublisher(node)      # 渡したノードに /ctl/duty パブリッシャを張る
fan.set_duties({1: 0.5, 3: 0.2})  # ファン毎に一括設定
fan.set_all_duty(0.3)             # 全ファン一括
fan.set_duty_array([0.1]*8)       # 8基まとめて設定（配分結果の publish に使用）
duty = fan.force_to_duty(0.02)    # 推力[N] → デューティ比 換算
```

推力配分（wrench → 8 duty）のロジックは ROS 非依存の `ThrustAllocator` として単体で使えます．
```python
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

alloc = ThrustAllocator()                       # 既定のファン幾何/係数で構築
duties = alloc.allocate([0.05, 0, 0], [0, 0, 0])  # +X 並進力 → 8ファン duty
```

> [!NOTE]
> ファン関連パラメータ（推力換算係数 `kj`，ファン配置，制御ゲイン等）はすべて **ROS2 パラメータ**として [config/gnc_params.yaml](config/gnc_params.yaml) に集約されています．各モジュールが自分の使うパラメータを宣言・取得するため，`kj` はファン配信と推力配分で共有される単一情報源です（`config` を渡さない場合は各モジュールの既定値で動作します）．


### 進行方向制御（ライブラリ）

進みたい**進行方向ベクトル（機体座標系）**を，推力配分（`ThrustAllocator`）により8基のファンへの並進力に変換するロジックです．Navigation を使わない自前の自由経路移動の土台で，現在は独立ノードではなく ROS 非依存の `utils/direction_controller.py` として提供されます（正規化 → `force_magnitude` 倍 → `max_force` でクランプ → wrench）．

```python
from sobits_intball2_gnc.control.utils.direction_controller import direction_to_force

force = direction_to_force([1.0, 0.0, 0.0], force_magnitude=0.02, max_force=0.1)  # +X 並進力
```

`DirectionController(allocator, fan_publisher, ...)` に `ThrustAllocator` と `FanDutyPublisher` を注入すれば，`step(direction)` で「方向 → 配分 → `/ctl/duty` 配信」まで実行できます．将来の自由経路飛行プログラムから利用します．

### ホバリング制御（統括ノード `control`）

制御系の中心となる統括ノードです．IMU（`/imu/imu`）のジャイロ・加速度で姿勢を安定させ（角速度減衰＝無回転維持・並進加速度外乱抑制），必要に応じて Nav 補正・経路チェックポイントを重ねます．入出力ラッパとロジックを結線する**唯一の rclpy ノード**です．

```sh
# パラメータファイルを与えて起動（Nav 補正が有効）
ros2 run sobits_intball2_gnc control --ros-args \
  --params-file $(ros2 pkg prefix sobits_intball2_gnc)/share/sobits_intball2_gnc/config/gnc_params.yaml

# パラメータファイルなしで起動（各モジュールの既定値＝純 IMU ホバリング）
ros2 run sobits_intball2_gnc control
```
- `ib2_msgs/msg/IMU` の `/imu/imu` を購読し，`config/gnc_params.yaml` のゲイン（`hover_control.kd_w`, `hover_control.kp_a` 等）で補正 wrench を計算 → 推力配分 → `/ctl/duty` に出力します．
- パラメータは ROS2 パラメータとして与えます（`ros2 param list /control_node` で確認可能）．

> [!NOTE]
> - IMU のみの場合は姿勢・位置の絶対参照がなく，**「無回転・外乱抑制の維持」**です（姿勢・位置はゆっくりドリフトします）．下記の **TF 補正**を有効にするとドリフトを抑制できます．
> - **将来の自由経路移動**は，`HoverController` のフィードフォワード並進力フック（`/gnc/feedforward_force`，または `compute(..., feedforward_force=...)`）に進行方向の力を与えることで，ホバリング制御の上に積み上げて実装できます．

#### TF 補正（自己位置によるドリフト抑制）

`config/gnc_params.yaml` の `hover_control.mode: tf_imu`（同梱の設定ファイルの既定）で，TF ツリー
（`iss_body` <- `body`）から取得した自己位置によるドリフト補正が有効になります．**IMU 制御が主力**のまま，
平滑化した位置・姿勢の保持目標からの誤差を低ゲインの補正 wrench として加算します
（補正は `max_corr_force`/`max_corr_torque` で独立にクランプされ，IMU 項を上回りません）．

- **前提**: Navigation は **OFF** のままで構いません．TF は Nav OFF でも配信されます（実測済み）．
  JAXA の制御器（`ctl_only`）は STAND_BY のままなので `/ctl/duty` の競合も起きません．
- 位置補正は **P 項のみ**です．TF からは速度を作らず，減衰は IMU の加速度項 `kp_a` と
  角速度項 `kd_w` が担います．
- TF は `poll_rate` で参照され，**ガウシアンフィルタ**（窓長 `smooth_window`・σ `smooth_sigma`）で
  平滑化されます．`smooth_window: 1` で平滑化を無効にできます．
- TF は pull 型なので，配信が止まってもルックアップはバッファの値を返し続けます．このため
  **スタンプが進んでいるか**で生死を判定します．スタンプが `timeout` 秒進まないと自動で
  **純 IMU ホバリングへ縮退**し，復帰すると保持目標を再捕捉して補正を再開します．
- `hover_control.mode: imu` にすると TF を一切参照しない純 IMU ホバリングになります．

> [!NOTE]
> シミュレータは `/clock` を publish しないため，`use_sim_time` は設定しません．TF のスタンプは
> スタンプ同士でのみ比較され，ノードのクロックとは突き合わせません．

#### 経路チェックポイントIF（将来の自由経路飛行の受け口）

`/gnc/checkpoints`（`geometry_msgs/PoseArray`，`tf_correction.reference_frame` 座標系＝既定 `iss_body`）に経路のチェックポイント配列を publish すると，先頭のポーズが保持目標に切り替わります．空配列でクリアされ，現在位置を保持目標として再捕捉します．

`header.frame_id` は基準フレームと照合され，不一致の配列は**全体が破棄**されます（空文字列は「未指定」として受理）．

```sh
# 例: チェックポイント1点を保持目標に設定
ros2 topic pub --once /gnc/checkpoints geometry_msgs/msg/PoseArray \
  "{header: {frame_id: iss_body}, poses: [{position: {x: 0.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}]}"
```

将来の自由経路飛行プログラムは，配列を publish → 到達判定しつつ `ControlNode.advance_checkpoint()` を呼ぶことで経路を進みます（到達判定・軌道生成はこのパッケージのスコープ外）．


<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## マイルストーン

現時点のバグや新規機能の依頼を確認するために[Issueページ](https://github.com/TeamSOBITS/sobits_intball2_gnc/issues)をご覧ください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 参考文献

- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [tf2_ros (ROS2)](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[contributors-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[forks-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/network/members
[stars-shield]: https://img.shields.io/github/stars/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[stars-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/stargazers
[issues-shield]: https://img.shields.io/github/issues/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[issues-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/issues
[license-shield]: https://img.shields.io/github/license/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[license-url]: LICENSE
