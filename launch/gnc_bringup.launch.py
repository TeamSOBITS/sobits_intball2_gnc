"""GNC bring-up launch: everything needed to observe/debug the GNC stack
beyond the control node itself.

Currently starts the ISS model (robot_state_publisher, so the ISS TF frames
-- iss_body, dock_body, etc. -- render) and, unless disabled, RViz with a
GNC-specific config (TF tree + the trajectory visualization path,
``/gnc/trajectory_path``). Modeled after nav2's bringup launch: RViz is one
togglable piece of this launch, not its purpose -- as the GNC stack grows
(e.g. a Guidance node), it belongs here too, so the file is named for the
whole stack rather than for RViz alone.

Deliberately a separate launch file from hover_control.launch.py: bring-up
extras like RViz are a debug aid, not something the control node should pull
in by default (e.g. in headless/CI runs). Run both together:

    ros2 launch sobits_intball2_gnc hover_control.launch.py
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_rviz:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to start RViz alongside the ISS model.",
    )

    # Re-declares intball2_programs' robot_state_publisher piece directly
    # (same URDF, same node) rather than including iss_model.launch.py
    # wholesale, since that launch file also starts its own generic rviz2
    # instance (urdf.rviz) -- including it would open a second, redundant
    # RViz window alongside the GNC-specific one started below.
    urdf_path = os.path.join(
        get_package_share_directory("intball2_programs"), "urdf", "iss.urdf"
    )
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    rviz_config_path = os.path.join(
        get_package_share_directory("sobits_intball2_gnc"),
        "rviz",
        "gnc.rviz",
    )

    iss_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="iss_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc, "publish_frequency": 50.0}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="gnc_rviz",
        arguments=["-d", rviz_config_path],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription([use_rviz_arg, iss_state_publisher, rviz_node])
