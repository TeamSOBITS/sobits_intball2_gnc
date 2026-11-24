"""Joy-Con teleop (C++): driver -> input mapping -> shared control -> bridge to teleop_node.

Start teleop_node first (it is not started here), without the RViz teleop panel holding the input:
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_rviz:=false
    ros2 launch sobits_intball2_teleop teleop.launch.py use_rviz:=false
    ros2 launch sobits_intball2_teleop joycon.launch.py

``driver:=false`` / ``input:=false`` start only the later nodes (e.g. to feed fake Joy messages).
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory('sobits_intball2_teleop')) / 'config'

    def node(executable, params=None, condition=None):
        return Node(package='sobits_intball2_teleop', executable=executable, name=executable,
                    output='screen', parameters=[LaunchConfiguration(params)] if params else [],
                    condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument('driver', default_value='true', description='Start joycon_driver.'),
        DeclareLaunchArgument('driver_params', default_value=str(config / 'joycon_driver.yaml')),
        DeclareLaunchArgument('input_params', default_value=str(config / 'joycon_input.yaml')),
        DeclareLaunchArgument('bridge_params', default_value=str(config / 'teleop_bridge.yaml')),
        node('joycon_driver', 'driver_params', IfCondition(LaunchConfiguration('driver'))),
        node('joycon_input', 'input_params'),
        node('shared_control'),
        node('teleop_bridge', 'bridge_params'),
    ])
