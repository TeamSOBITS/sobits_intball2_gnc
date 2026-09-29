"""GNC bring-up launch: control_node plus everything needed to observe/debug
the GNC stack.

Starts control_node (control.launch.py, unless ``use_control:=false``), the ISS and ib2 models (robot_state_publisher for each, so
the ISS TF frames -- iss_body, dock_body, etc. -- and the ib2 mesh render)
and, unless disabled, RViz with a
GNC-specific config (TF tree + the trajectory visualization path,
``/gnc/trajectory_path``). Modeled after nav2's bringup launch: RViz is one
togglable piece of this launch, not its purpose -- as the GNC stack grows
(e.g. a Guidance node), it belongs here too, so the file is named for the
whole stack rather than for RViz alone.

control.launch.py stays usable on its own: control_node must be stopped alone
before restarting the sim/bridge, so run it separately with ``use_control:=false``.
guidance_node is not included (guidance.launch.py).

    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_rviz:=false
    ros2 launch sobits_intball2_gnc gnc_bringup.launch.py use_control:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to start RViz alongside the ISS model.",
    )
    use_control_arg = DeclareLaunchArgument(
        "use_control",
        default_value="true",
        description="Whether to start control_node (control.launch.py).",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("sobits_intball2_gnc"), "config", "gnc_params.yaml"
        ),
        description="Path to the ROS2 parameter file for the control node.",
    )

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sobits_intball2_gnc"),
                "launch",
                "control.launch.py",
            )
        ),
        launch_arguments={"params_file": LaunchConfiguration("params_file")}.items(),
        condition=IfCondition(LaunchConfiguration("use_control")),
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

    ib2_urdf_path = os.path.join(
        get_package_share_directory("intball2_programs"), "urdf", "ib2.urdf"
    )
    with open(ib2_urdf_path, "r") as infp:
        ib2_robot_desc = infp.read()

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

    # ib2's own robot_state_publisher, remapped off robot_description so it
    # doesn't clash with iss_state_publisher's -- same pattern as
    # intball2_programs' iss_model.launch.py.
    ib2_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ib2_state_publisher",
        output="screen",
        parameters=[{"robot_description": ib2_robot_desc, "publish_frequency": 50.0}],
        remappings=[("robot_description", "ib2_description")],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="gnc_rviz",
        arguments=["-d", rviz_config_path],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    location_broadcaster_node = Node(
        package="sobits_intball2_gnc",
        executable="location_broadcaster",
        name="location_broadcaster",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    octomap_marker_node = Node(
        package="sobits_intball2_gnc",
        executable="octomap_marker_publisher",
        name="octomap_marker_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            use_rviz_arg,
            use_control_arg,
            params_file_arg,
            control_launch,
            iss_state_publisher,
            ib2_state_publisher,
            rviz_node,
            location_broadcaster_node,
            octomap_marker_node,
        ]
    )
