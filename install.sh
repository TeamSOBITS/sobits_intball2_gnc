#!/bin/bash

echo "Updating package list..."
# sudo apt-get update

sudo apt-get install -y \
    zenity

# 力/トルク制約付き軌道の時間割当（TOPP-RA）用
pip3 install toppra

# sudo pip install octomap-python

# octomap Python バインディング用ライブラリパス
# if ! grep -q '.local/lib' ~/.bashrc 2>/dev/null; then
#     echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# fi
# export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

echo "Installation complete."