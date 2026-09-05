"""
goal_manager.launch.py

Always launches goal_relay_node (RViz → /global_goal).
Additionally launches mission_runner_node only when `mission_file` is non-empty.

Arguments
---------
  use_sim_time : bool (default: false)
  mission_file : str  (default: "")
      Path to a waypoint mission YAML. Empty disables the mission runner.
  repeat       : bool (default: false)
      If true, the mission loops after the last waypoint.
  start_delay  : float (default: 3.0) seconds to wait before publishing the
                  first waypoint, letting hw_bridge / TF warm up.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    share = get_package_share_directory("robot_real_goal_manager")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("mission_file", default_value="",
                              description="Path to waypoint mission YAML "
                                          "(empty disables the mission runner)"),
        DeclareLaunchArgument("repeat",       default_value="false"),
        DeclareLaunchArgument("start_delay",  default_value="3.0"),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    mission_file = LaunchConfiguration("mission_file")
    repeat       = LaunchConfiguration("repeat")
    start_delay  = LaunchConfiguration("start_delay")

    mission_active = PythonExpression(["'", mission_file, "' != ''"])

    goal_relay = Node(
        package="robot_real_goal_manager",
        executable="goal_relay_node",
        name="robot_real_goal_relay",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    mission_runner = Node(
        package="robot_real_goal_manager",
        executable="mission_runner_node",
        name="robot_real_mission_runner",
        output="screen",
        parameters=[{
            "use_sim_time":     use_sim_time,
            "mission_file":     mission_file,
            "repeat":           repeat,
            "start_delay_sec":  start_delay,
        }],
        condition=IfCondition(mission_active),
    )

    return LaunchDescription(args + [goal_relay, mission_runner])
