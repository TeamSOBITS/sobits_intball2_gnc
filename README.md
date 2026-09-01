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
      <a href="#gncの枠組みでの位置づけ">GNCの枠組みでの位置づけ</a>
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
    <li><a href="#実行方法">実行方法</a></li>
    <li><a href="#マイルストーン">マイルストーン</a></li>
  </ol>
</details>

## 概要

Int-Ball2 シミュレータでロボットを自律移動させるためのパッケージです．
ROS2 Humble に対応しています．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### GNCの枠組み

| 役割 | 内容 |
|---|---|
| **[Guidance](gnc/sobits_intball2_gnc/guidance/README.md)** | 目標軌道 `p_des(t), v_des(t), a_des(t), q_des(t)` を生成 |
| **[Navigation](gnc/sobits_intball2_gnc/navigation/README.md)** | 自己位置推定，移動先地点配信 | 
| **[Control](gnc/sobits_intball2_gnc/control/README.md)** | 目標軌道を追従する force/torque を計算し，8 duty へ配分 |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### パッケージ構成

```
sobits_intball2_gnc/                 # gitリポジトリルート（colconパッケージを2つ内包）
├── gnc/                             # メインパッケージ（ament_python）
│   ├── config/
│   │   └── gnc_params.yaml          # GNC パラメータ（ROS2 param 形式：ファン配置・推力モデル・制御ゲイン）
│   ├── maps/
│   │   └── iss_location.yaml        # 登録済みロケーション一覧（27地点）
│   ├── sobits_intball2_gnc/
│   │   ├── navigation/              # Navigation（N）: 詳細は navigation/README.md
│   │   ├── control/                 # Control（C）: 詳細は control/README.md
│   │   └── guidance/                # Guidance（G）: min_snap.pyのコアロジックのみ未実装、詳細は guidance/README.md
│   ├── test/                        # 各ロジックの単体テスト（ROS 不要）
│   ├── package.xml
│   ├── setup.py
│   └── setup.cfg
└── minco_native_py/                 # MINCO姿勢/トルク統合軌道生成のpybind11拡張（ament_cmake）
    ├── package.xml
    ├── CMakeLists.txt
    └── src/
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
   colcon build --packages-select minco_native_py sobits_intball2_gnc
   source ~/colcon_ws/install/setup.bash
   ```

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 実行方法

- [gnc_bringup.launch.py](gnc/launch/gnc_bringup.launch.py)を起動し，移動先地点を配信します．
    ```
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py
    ```
- [hover_control.launch.py](gnc/launch/hover_control.launch.py)を起動し，現在位置・姿勢の保持を行います．
    ```
    ros2 launch sobits_intball2_gnc hover_control.launch.py
    ```
- [guidance.py](gnc/sobits_intball2_gnc/guidance/guidance.py)を起動し，目標軌道の生成・追従を行います．
    ```
    ros2 run sobits_intball2_gnc guidance
    ```
    詳細は[guidance/README.md](gnc/sobits_intball2_gnc/guidance/README.md)を参照してください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## マイルストーン

現時点のバグや新規機能の依頼を確認するために[Issueページ](https://github.com/TeamSOBITS/sobits_intball2_gnc/issues)をご覧ください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 参考文献

- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [tf2_ros (ROS2)](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)
- Pham, Hung, and Quang-Cuong Pham. "A new approach to Time-Optimal Path Parameterization based on Reachability Analysis." *IEEE Transactions on Robotics*, vol. 34, no. 3, 2018, pp. 645-659. ([arXiv:1707.07239](https://arxiv.org/abs/1707.07239), [GitHub](https://github.com/hungpham2511/toppra))
- Wang, Zhepei, Xin Zhou, Chao Xu, and Fei Gao. "Geometrically Constrained Trajectory Optimization for Multicopters." *IEEE Transactions on Robotics* (T-RO), vol. 38, no. 5, 2022, pp. 3259-3278. ([arXiv:2103.00190](https://arxiv.org/abs/2103.00190), [GitHub](https://github.com/ZJU-FAST-Lab/GCOPTER))

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
