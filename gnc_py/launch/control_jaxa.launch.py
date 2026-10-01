"""Launch ``jaxa_control_node``: IntBall2 driven by the ported JAXA controller.

Comparison baseline (docs/jaxa_controller_port_plan.md). Replaces our control
node, so start the rest without it:

    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_control:=false
    ros2 launch sobits_intball2_gnc control_jaxa.launch.py

``gnc_params.yaml`` supplies the TF frames, timeouts and velocity-filter alpha;
``jaxa_control.yaml`` the JAXA controller values. JAXA's fsm allocates the
wrench, so JAXA ctl_only must stay in STAND_BY (Navigation OFF).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(get_package_share_directory("sobits_intball2_gnc"), "config")
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=os.path.join(config, "gnc_params.yaml"),
        description="Our parameter file (TF frames, timeouts, velocity filter).")
    jaxa_params_file_arg = DeclareLaunchArgument(
        "jaxa_params_file", default_value=os.path.join(config, "jaxa_control.yaml"),
        description="JAXA controller parameters (copied from JAXA ctl_only/ctl.yaml).")
    node = Node(
        package="sobits_intball2_gnc",
        executable="jaxa_control",
        name="jaxa_control_node",
        parameters=[LaunchConfiguration("params_file"), LaunchConfiguration("jaxa_params_file"),
                    {"use_sim_time": True}],
        output="screen",
        emulate_tty=True,
    )
    return LaunchDescription([params_file_arg, jaxa_params_file_arg, node])
