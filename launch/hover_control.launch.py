"""Launch the IntBall2 hover control node with its ROS2 parameter file.

Loads ``config/gnc_params.yaml`` (installed to the package share directory) into
the ``control`` orchestrator node so TF-corrected IMU hover runs with the tuned
gains. Override the parameter file with ``params_file:=<path>`` to try a
different tuning without editing the installed file.

Runs with Navigation OFF: self-position comes from the TF tree, so nothing here
needs the JAXA navigation stack.

``use_sim_time`` is enabled (2026-08-19): the simulator's ``/clock`` is now
bridged from ROS1 (see ``/root/bridge/bridge_topics.yaml``), so this node's ROS
clock tracks simulation time. Previously it did not, and the control loop used
``time.monotonic()`` (wall-clock) for its own timing while comparing against
TF stamps (sim time) -- under CPU load, Gazebo's real-time factor drops below
1 and the two clocks desync, which was traced to a control-loop tracking
failure (see docs/recording_cpu_load_control_degradation.md). Aligning this
node to sim time removes that mismatch.

    ros2 launch sobits_intball2_gnc hover_control.launch.py
    ros2 launch sobits_intball2_gnc hover_control.launch.py params_file:=/abs/path.yaml
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
        description="Path to the ROS2 parameter file for the control node.",
    )

    control_node = Node(
        package="sobits_intball2_gnc",
        executable="control",
        # Must match the node name in code so the params file's `/**` (or a
        # named block) is applied to it.
        name="control_node",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([params_file_arg, control_node])
