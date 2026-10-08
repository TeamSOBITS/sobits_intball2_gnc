<a name="readme-top"></a>

[JA](README.md) | [EN](README.en.md)

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
    <li><a href="#概要">概要</a></li>
    <li><a href="#パッケージ構成">パッケージ構成</a></li>
    <li>
      <a href="#セットアップ">セットアップ</a>
      <ul>
        <li><a href="#環境条件">環境条件</a></li>
        <li><a href="#インストール方法">インストール方法</a></li>
      </ul>
    </li>
    <li>
      <a href="#実行操作方法">実行・操作方法</a>
      <ul>
        <li><a href="#静的octomap配信方法">静的OctoMap配信方法</a></li>
        <li><a href="#地点登録方法">地点登録方法</a></li>
        <li><a href="#自律移動方法">自律移動方法</a></li>
        <li><a href="#パラメータ">パラメータ</a></li>
      </ul>
    </li>
    <li><a href="#マイルストーン">マイルストーン</a></li>
    <li><a href="#参考文献">参考文献</a></li>
  </ol>
</details>

## 概要
Int-Ball2 シミュレータ上でロボットを自律移動させるためのパッケージです．
OctoMap による ISS 内の静的地図と，深度カメラの点群による動的障害物をもとに A* で経路を計画し，ロボットを目的地まで移動させます．

<p align="center">
  <img src="docs/images/person_avoidance.gif" alt="Int-Ball2 が人を避けて移動する様子（4.5倍速）．緑の線が計画した経路" width="720">
</p>

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## パッケージ構成

```
sobits_intball2_gnc/
├── config/
│   └── gnc_params.yaml          # 自律移動パラメータ設定
├── launch/
│   └── iss_static_map_server.launch
├── maps/
│   ├── iss_locations.yaml       # 登録済みロケーション一覧
│   └── iss_octomap.bt           # ISS 静的 OctoMap データ
└── scripts/
    ├── gnc_manager.py           # GNC プロセス全体を統括．経路計画とロボットの移動を管理
    ├── gnc_defaults.py          # デフォルトパラメータ定義
    ├── navigator.py             # 目的地の解決・経路計画・実行を一貫して行うファサード
    ├── control/                 # ロボットへの移動コマンド送信
    │   ├── action_handler.py
    │   ├── base_executor.py
    │   ├── smooth_executor.py   # 平滑化軌道追従エグゼキュータ
    │   └── trajectory_follower.py
    ├── guidance/                # A* 経路計画・衝突検出・経路平滑化・可視化
    │   ├── astar_planner.py
    │   ├── base_planner.py
    │   ├── collision_checker.py
    │   ├── obstacle_manager.py
    │   ├── path_planner.py
    │   ├── safety_astar_planner.py
    │   ├── smoother.py
    │   ├── test_planner.py      # 経路計画の結合テスト用スクリプト
    │   └── visualize.py
    └── navigation/              # TF フレーム解決・座標変換・ロケーション登録
        ├── location_broadcaster.py  # YAML のロケーションを TF にパブリッシュ
        ├── location_setting.py      # ロケーション登録 GUI
        ├── pose_resolver.py
        ├── save_current_location.py # 現在位置を YAML に保存
        └── tf_frame_resolver.py
```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## セットアップ

### 環境条件
まず，以下の環境を整えてから，次のインストール方法に進んでください．

| System  | Version |
| --- | --- |
| Ubuntu | 20.04 (Focal Fossa) |
| ROS    | Noetic Ninjemys |
| Python | 3.8 |

また，Int-Ball2 シミュレータ（`ib2_msgs` を含む）が動作する環境が必要です．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### インストール方法

1. ROSの`src`フォルダに移動します．
   ```sh
   cd ~/catkin_ws/src/
   ```
2. 本レポジトリをcloneします．
   ```sh
   git clone https://github.com/TeamSOBITS/sobits_intball2_gnc.git
   ```
3. レポジトリの中へ移動します．
   ```sh
   cd sobits_intball2_gnc
   ```
4. 依存パッケージをインストールします．
   ```sh
   bash install.sh
   ```
   - `octomap_server`，`pcl_ros`，`nodelet`，`gazebo_msgs`，`zenity` を apt で，`octomap-python` を pip でインストールします．
   - `~/.bashrc` に octomap の Python バインディング用の `LD_LIBRARY_PATH` を追記します．
5. パッケージをコンパイルします．
   ```sh
   cd ~/catkin_ws/
   catkin_make
   source ~/catkin_ws/devel/setup.bash
   ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## 実行・操作方法
はじめに Int-Ball2 シミュレータを起動し，GSE で Navigation を ON にしてください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 静的OctoMap配信方法
以下で起動します．
```sh
roslaunch sobits_intball2_gnc iss_static_map_server.launch
```
- 主な出力トピック
  - `/occupied_cells_vis_array` [visualization_msgs/MarkerArray]
    - 「障害物がある場所」をボクセル（立方体）の集合として表示します．RViz での描画用です．
  - `/octomap_binary` [octomap_msgs/Octomap]
    - 地図を「占有（障害物あり）」か「自由（空間あり）」の 2 値で表現した軽量なバイナリデータです．通信負荷が低いため，リアルタイムの共有に適しています．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 地点登録方法
1. yamlに登録した地点をTFにしてpublishします．
   ```sh
   rosrun sobits_intball2_gnc location_broadcaster.py
   ```
2. 保存するyamlのパスを選択し（デフォルト: [iss_locations.yaml](maps/iss_locations.yaml)），実行します．
   ```sh
   rosrun sobits_intball2_gnc location_setting.py
   ```
   - GUIが起動します．シミュレータ内でロボットを登録したい地点へ移動させ，姿勢も合わせた後にGUIで登録してください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 自律移動方法
1. 静的OctoMapを配信します．
   ```sh
   roslaunch sobits_intball2_gnc iss_static_map_server.launch
   ```
2. yamlに登録した地点をTFにしてpublishします．
   ```sh
   rosrun sobits_intball2_gnc location_broadcaster.py
   ```
3. 障害物検出のため，深度カメラの点群（`sensor_msgs/PointCloud2`）を `/depth/points` にpublishするノードを起動します．
   - トピック名は[gnc_params.yaml](config/gnc_params.yaml)の `obstacle_topic` で変更できます．
   - 点群が取得できない場合，動的障害物は無視して経路を計画します．
4. 自律移動ノードを起動します．
   ```sh
   rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
   ```
   - コマンドライン引数

     | 引数 | 型 | 説明 |
     |------|-----|------|
     | `--target` | str | 目的地の TF フレーム名（`--goal` と排他・どちらか必須） |
     | `--goal X Y Z` | float×3 | 目的地の iss_body 座標 [m]（`--target` と排他・どちらか必須） |
     | `--offset X Y Z` | float×3 | オフセット [m]（デフォルト: 0 0 0） |

   - 実行例
     ```sh
     # TFフレーム指定
     rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
     # 座標指定
     rosrun sobits_intball2_gnc gnc_manager.py --goal 4.5 -4.0 11.2
     ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### パラメータ
自律移動のパラメータは[gnc_params.yaml](config/gnc_params.yaml)で設定できます．
各パラメータの意味は，同ファイル内のコメントを参照してください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## マイルストーン

現時点のバグや新規機能の依頼を確認するために[Issueページ](https://github.com/TeamSOBITS/sobits_intball2_gnc/issues)をご覧ください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>


## 参考文献
- [OctoMap](https://octomap.github.io/)

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

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
