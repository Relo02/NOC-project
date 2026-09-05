"""
lidar.launch.py

Launches the Livox Mid-360 LiDAR adapter for the real G1.

Arguments
---------
  use_sim_time : bool (default: false)
  params_file  : str  (default: package-bundled config/lidar_filter.yaml)
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    share = get_package_share_directory("robot_real_lidar")
    default_params = os.path.join(share, "config", "lidar_filter.yaml")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("params_file",  default_value=default_params),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file  = LaunchConfiguration("params_file")

    lidar_filter = Node(
        package="robot_real_lidar",
        executable="lidar_filter_node",
        name="robot_real_lidar_filter",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(args + [lidar_filter])
