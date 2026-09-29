"""Launch the virtual (ideal) depth camera for the Gazebo sim.

Starts ``virtual_camera_node`` (sobits_intball2_gnc_cpp) with
``config/virtual_camera.yaml`` and the camera links' optical frames, which the
ib2 URDF does not define (same rotation as intball2_programs'
stereo_pointcloud.launch.py). Sim only, so not part of gnc_bringup.

    ros2 launch sobits_intball2_gnc virtual_camera.launch.py
    ros2 launch sobits_intball2_gnc virtual_camera.launch.py params_file:=/abs/path.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OPTICAL_FRAMES = {
    "cameraL_link": "cameraL_optical_frame",
    "cameraF_link": "cameraF_optical_frame",
}


def optical_frame_publisher(link, optical):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"{optical}_publisher",
        arguments=["--yaw", "-1.5708", "--roll", "-1.5708", "--frame-id", link, "--child-frame-id", optical],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(get_package_share_directory("sobits_intball2_gnc"), "config", "virtual_camera.yaml"),
        description="Path to the virtual camera parameter file.",
    )
    camera_node = Node(
        package="sobits_intball2_gnc_cpp",
        executable="virtual_camera_node",
        name="virtual_camera_node",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        # Passive: idle OpenMP workers otherwise spin between frames (measured 66% -> 34% CPU at 200x200).
        additional_env={"OMP_WAIT_POLICY": "PASSIVE"},
        output="screen",
        emulate_tty=True,
    )
    return LaunchDescription(
        [params_file_arg, camera_node] + [optical_frame_publisher(l, o) for l, o in OPTICAL_FRAMES.items()])
