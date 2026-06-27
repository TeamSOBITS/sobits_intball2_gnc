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

```
sobits_intball2_gnc/
├── maps/
│   └── iss_location.yaml            # 登録済みロケーション一覧（27地点）
├── sobits_intball2_gnc/
│   └── navigation/
│       ├── location_broadcaster.py  # YAML のロケーションを TF にパブリッシュ（10Hz）
│       └── location_setting.py      # ロケーション登録 GUI（Tkinter + Zenity）
├── package.xml
├── setup.py
└── setup.cfg
```

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

はじめに Int-Ball2 シミュレータを起動し，GSE で Navigation を ON にする．

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
Navigationを使わず，`/ctl/duty`へ直接publishして8基のファンを個別に駆動できます．自律移動・ホバリングをユーザプログラムで実装するための最低レベル制御です．
```sh
ros2 run sobits_intball2_gnc fan_control [引数]
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
ros2 run sobits_intball2_gnc fan_control --fan 1 --duty 0.5 --duration 2

# ファン毎に個別指定（fan1=0.5, fan3=0.2）
ros2 run sobits_intball2_gnc fan_control --set 1:0.5 3:0.2 --duration 2

# 全ファンを duty=0.3 で 1秒間 駆動
ros2 run sobits_intball2_gnc fan_control --all 0.3 --duration 1

# 逆向きペアで往復（並進をおおよそ初期位置に戻す: 押す→2倍戻す→止める）
ros2 run sobits_intball2_gnc fan_control --fan 1 --duty 0.3 --duration 1 && \
ros2 run sobits_intball2_gnc fan_control --fan 8 --duty 0.3 --duration 2 && \
ros2 run sobits_intball2_gnc fan_control --fan 1 --duty 0.3 --duration 1
```

他のPythonプログラムから利用する場合は`FanControlNode`をimportして使えます．
```python
from intball2_programs.fan_control import FanControlNode

fan = FanControlNode()
fan.set_duty(1, 0.5)            # fan1 を duty=0.5
fan.set_duties({1: 0.5, 3: 0.2})  # ファン毎に一括設定
fan.set_all_duty(0.3)          # 全ファン一括
duty = fan.force_to_duty(0.02)  # 推力[N] → デューティ比 換算
```


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
