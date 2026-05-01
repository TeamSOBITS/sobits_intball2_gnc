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
    <li>
      <a href="#概要">概要</a>
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
Int-Ball2 simulaterでロボットを自律移動させるためのパッケージです．

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

### 環境条件
まず，以下の環境を整えてから，次のインストール方法に進んでください．
| System  | Version |
| --- | --- |
| Ubuntu | 20.04 (Focal Fossa) |
| ROS    | Noetic Ninjemys |
| Python | 3.8 |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### インストール方法

1. ROSの`src`フォルダに移動します．
   ```sh
   cd　~/catkin_ws/src/
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
5. パッケージをコンパイルします．
   ```sh
   cd ~/catkin_ws/
   catkin_make
   source ~/catkin_ws/devel/setup.bash
   ```
<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 実行・操作方法
はじめにInt‑Ball2 シミュレータを起動し，GSEでNavigationをONにする．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 静的OctoMap配信方法
以下で起動
```sh
roslaunch sobits_intball2_gnc iss_static_map_server.launch
```
- 主な出力トピック
  - /occupied_cells_vis_array [visualization_msgs/MarkerArray]
    - 「障害物がある場所」をボクセル（立方体）の集合として表示．Rvizでの描画用
  - /octomap_binary [octomap_msgs/Octomap]
    - 地図を「占有（障害物あり）」か「自由（空間あり）」の 2 値で表現した軽量なバイナリデータです．通信負荷が低いため，リアルタイムの共有に適す


<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 地点登録方法
1. yamlに登録した地点をTFにしてpublishする
    ```sh
    rosrun sobits_intball2_gnc location_broadcaster.py
    ```
2. 保存するyamlのパスを選択し(デフォルト: [iss_locations.yaml](maps/iss_locations.yaml)), 実行
    ```sh
    rosrun sobits_intball2_gnc location_setting.py 
    ```
    - GUIが起動します，シム内でロボットを登録したい地点へ移動させ，姿勢も合わせた後にGUIで登録してください

### 自律移動方法
1. 静的octomap配信
    ```sh
    roslaunch sobits_intball2_gnc iss_static_map_server.launch
    ```
2. yamlに登録した地点をTFにしてpublishする
    ```sh
    rosrun sobits_intball2_gnc location_broadcaster.py
    ```
3. 障害物検出のために点群をPublishするコードを起動する

4. 自立移動ノード起動
    ```sh
    rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
    ```
    - コマンドライン引数

      | 引数 | 型 | 説明 |
      |------|-----|------|
      | `--target` | str | 目的地の TF フレーム名（`--goal` と排他・どちらか必須） |
      | `--goal X Y Z` | float×3 | 目的地の iss_body 座標（`--target` と排他・どちらか必須） |
      | `--offset X Y Z` | float×3 | オフセット（デフォルト: 0 0 0） |

    - 実行例
      ```sh
      # TFフレーム指定
      rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
      # 座標指定
      rosrun sobits_intball2_gnc gnc_manager.py --goal 4.5 -4.0 11.2
      ```
    
### パラメータ
自律移動のパラメータは[gnc_params.yaml](config/gnc_params.yaml)で設定できます


<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## マイルストーン

現時点のバグや新規機能の依頼を確認するために[Issueページ](https://github.com/TeamSOBITS/sobits_intball2_gnc/issues)をご覧ください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 参考文献
- [OctoMap](https://octomap.github.io/)

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




