"""Launch ``teleop_node``: keyboard teleoperation (tkinter window + reference on /gnc/trajectory_setpoint).

Needs a display (tkinter) and the control side running, e.g.:

    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_rviz:=false   # control + TF
    ros2 launch sobits_intball2_gnc teleop.launch.py

``use_rviz`` (default true) opens ``rviz/teleop.rviz``: fixed frame ``body``, chase view straight behind and
45 deg above, so screen directions match the teleop keys whatever the vehicle's attitude in the ISS.

Do not send ``/gnc/move_to`` goals while teleoperating: input is ignored while one runs.
``gnc_params.yaml`` (``teleop.*``) supplies the limits.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("sobits_intball2_gnc")
    config = os.path.join(share, "config")
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=os.path.join(config, "gnc_params.yaml"),
        description="Parameter file (teleop limits, TF frames, vehicle model).")
    node = Node(
        package="sobits_intball2_gnc",
        executable="teleop",
        name="teleop_node",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        output="screen",
        emulate_tty=True,
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Open the teleop chase-view RViz (rviz/teleop.rviz).")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="teleop_rviz",
        arguments=["-d", os.path.join(share, "rviz", "teleop.rviz")],
        parameters=[{"use_sim_time": True}],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )
    return LaunchDescription([params_file_arg, use_rviz_arg, node, rviz])
