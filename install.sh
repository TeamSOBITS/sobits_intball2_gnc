#!/bin/bash

echo "Updating package list..."
# sudo apt-get update

sudo apt-get install -y \
    zenity \
    pybind11-dev

# 力/トルク制約付き軌道の時間割当（TOPP-RA）用
pip3 install toppra

# MINCO姿勢/トルク統合軌道生成（minco_native_pyパッケージ、colconワークスペースの
# src/直下に独立git repoとして配置）用のgcopterヘッダ取得
# MITライセンス（Copyright Zhepei Wang, Fei Gao）、colcon buildには含めずここで取得のみ行う
MINCO_NATIVE_PY_DIR="$(dirname "$0")/../minco_native_py"
GCOPTER_DIR="$MINCO_NATIVE_PY_DIR/third_party/gcopter"
GCOPTER_COMMIT="e0444f6d47b84f972ced91746b05feb36ce1fd4f"

if [ -d "$MINCO_NATIVE_PY_DIR" ] && [ ! -d "$GCOPTER_DIR" ]; then
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