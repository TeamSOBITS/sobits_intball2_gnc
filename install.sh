#!/bin/bash

echo "Updating package list..."
# sudo apt-get update

sudo apt-get install -y \
    ros-${ROS_DISTRO}-octomap-server \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-nodelet \
    ros-${ROS_DISTRO}-gazebo-msgs \
    zenity

sudo pip install octomap-python

# octomap Python バインディング用ライブラリパス
if ! grep -q '.local/lib' ~/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

echo "Installation complete."