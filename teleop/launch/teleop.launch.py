"""Teleop: the backend node (keys -> moving reference) and the RViz view with the teleop panel.

Needs the control side running and TF available, e.g.:

    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_rviz:=false   # control + TF + models
    ros2 launch sobits_intball2_teleop teleop.launch.py

``rviz/teleop.rviz`` uses fixed frame ``body`` and a chase view straight behind and 45 deg above, so
screen directions match the teleop keys whatever the vehicle's attitude in the ISS. ``use_rviz:=false``
starts only the backend.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('sobits_intball2_teleop'))
    gnc_share = Path(get_package_share_directory('sobits_intball2_gnc'))
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(gnc_share / 'config' / 'gnc_params.yaml'),
                              description='Parameter file (teleop limits, TF frames, vehicle model).'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Open RViz with the teleop panel.'),
        Node(package='sobits_intball2_teleop', executable='teleop_node', name='teleop_node', output='screen',
             parameters=[LaunchConfiguration('params_file'), {'use_sim_time': True}]),
        Node(package='rviz2', executable='rviz2', name='teleop_rviz', output='screen',
             arguments=['-d', str(share / 'config' / 'teleop.rviz')],
             parameters=[{'use_sim_time': True, 'teleop_ros_enabled': True}],
             condition=IfCondition(LaunchConfiguration('use_rviz'))),
    ])
