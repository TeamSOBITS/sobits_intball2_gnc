#!/bin/bash

echo "Updating package list..."
# sudo apt-get update

sudo apt-get install -y \
    zenity \
    pybind11-dev \
    ros-humble-gazebo-msgs  # virtual_camera_node (/gazebo/model_states)

# 力/トルク制約付き軌道の時間割当（TOPP-RA）用
pip3 install toppra

# C++実装パッケージ（sobits_intball2_gnc_cpp、本リポジトリ直下のgnc_cpp/に
# gnc_py/と並ぶ別colconパッケージとして配置）のMINCO軌道最適化用gcopterヘッダ取得
# MITライセンス（Copyright Zhepei Wang, Fei Gao）、colcon buildには含めずここで取得のみ行う
GNC_CPP_DIR="$(dirname "$0")/gnc_cpp"
GCOPTER_DIR="$GNC_CPP_DIR/third_party/gcopter"
GCOPTER_COMMIT="e0444f6d47b84f972ced91746b05feb36ce1fd4f"

if [ -d "$GNC_CPP_DIR" ] && [ ! -d "$GCOPTER_DIR" ]; then
    git clone https://github.com/ZJU-FAST-Lab/GCOPTER.git "$GCOPTER_DIR"
    git -C "$GCOPTER_DIR" checkout "$GCOPTER_COMMIT"
fi

# sudo pip install octomap-python

# octomap Python バインディング用ライブラリパス
# if ! grep -q '.local/lib' ~/.bashrc 2>/dev/null; then
#     echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# fi
# export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

echo "Installation complete."