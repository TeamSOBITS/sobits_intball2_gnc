"""Launch the IntBall2 guidance node with its ROS2 parameter file.

Loads ``config/gnc_params.yaml`` (installed to the package share directory) into
``guidance_node``, the same file ``hover_control.launch.py`` gives
``control_node``, so both nodes read one set of shared physical constants
(``trajectory_controller.mass``, ``thrust_allocator.*``, ...). Started via
plain ``ros2 run`` the node silently fell back to its in-code defaults, which
had drifted from the yaml (e.g. mass 4.5 vs. 3.216).

    ros2 launch sobits_intball2_gnc guidance.launch.py
    ros2 launch sobits_intball2_gnc guidance.launch.py params_file:=/abs/path.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_params = os.path.join(
        get_package_share_directory("sobits_intball2_gnc"),
        "config",
        "gnc_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the ROS2 parameter file for the guidance node.",
    )

    guidance_node = Node(
        package="sobits_intball2_gnc",
        executable="guidance",
        # Must match the node name in code so the params file's `/**` (or a
        # named block) is applied to it.
        name="guidance_node",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([params_file_arg, guidance_node])
