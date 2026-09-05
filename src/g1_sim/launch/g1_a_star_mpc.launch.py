"""
g1_a_star_mpc.launch.py — the complete autonomous navigation stack of the G1.

MuJoCo plant: in Gazebo the G1 would have no source of motion without writing one
from scratch (see g1_sim/README.md).

Chain
------
    mujoco_sim  --/odom------------> odom_to_pose_node --/robot_pose--+
                --/livox/lidar--> lidar_filter_node                   |
                                     |                                |
                                     +--/lidar/points_filtered--------+
                                                    |                 |
                                                    v                 v
                                              a_star_node  ------> mpc_node
                                              /a_star/path      /mpc/next_setpoint
                                                                      |
                                                    setpoint_to_cmd_vel_node
                                                                      |
                                                                 /cmd_vel
                                                                      |
                                                                 mujoco_sim

The robot enters the chain only through the parameter file and the name of the
pose topic: no node of the algorithmic stack depends on the platform.

The goal is sent on /global_goal (PoseStamped) or with the "2D Goal Pose" tool
of RViz, which publishes on /goal_pose (see the goal_relay argument).

Arguments
---------
  params_file  : str  planner profile (default: planner_params_g1.yaml)
  lidar_params : str  configuration of the LiDAR adapter
  sim_params   : str  configuration of the simulator
  robot_model  : bool (default true)   publish /robot_description for RViz
  use_rviz     : bool (default true)
  viewer       : bool (default true)   MuJoCo window
  people       : str  (default '')     dynamic obstacles: '' or 'default'
  nav_graph    : bool (default false)  global topological memory (Dijkstra)
  goal_relay   : bool (default true)   /goal_pose -> /global_goal for RViz
  use_mission  : bool (default false)  run a waypoint mission
  mission_file : str  mission YAML (see robot_real_goal_manager)
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    g1_share      = get_package_share_directory("g1_sim")
    planner_share = get_package_share_directory("a_star_mpc_planner")

    default_planner = os.path.join(planner_share, "config", "planner_params_g1.yaml")
    default_overlay = os.path.join(planner_share, "config", "overlay_none.yaml")
    default_lidar   = os.path.join(g1_share, "config", "lidar_filter_g1.yaml")
    default_sim     = os.path.join(g1_share, "config", "g1_sim.yaml")
    default_rviz    = os.path.join(g1_share, "rviz", "g1_nav.rviz")

    args = [
        DeclareLaunchArgument("params_file",  default_value=default_planner),
        # Second parameter file, merged ON TOP of params_file (ROS 2 applies the
        # files in list order and the last one wins). It is there to vary a few
        # parameters without duplicating a 250-line profile: see
        # config/overlay_nonconvex.yaml per i mondi concavi.
        DeclareLaunchArgument("planner_overlay", default_value=default_overlay),
        DeclareLaunchArgument("lidar_params", default_value=default_lidar),
        DeclareLaunchArgument("sim_params",   default_value=default_sim),
        DeclareLaunchArgument("rviz_config",  default_value=default_rviz),
        DeclareLaunchArgument("use_rviz",     default_value="true"),
        DeclareLaunchArgument("robot_model",  default_value="true",
                              description="publish /robot_description to see the G1 in RViz"),
        DeclareLaunchArgument("viewer",       default_value="true"),
        DeclareLaunchArgument("people",       default_value=""),
        # Geometry of the MuJoCo world. "industrial" is the warehouse;
        # long_wall / horseshoe / dead_end are the worlds with non-convex
        # obstacles (see g1_sim/mujoco_world.py, WORLDS). Changing world also
        # changes the spawn pose, taken from the world itself.
        DeclareLaunchArgument("world",        default_value="industrial"),
        DeclareLaunchArgument("nav_graph",    default_value="false"),
        DeclareLaunchArgument("goal_relay",   default_value="true"),
        DeclareLaunchArgument("use_mission",  default_value="false"),
        DeclareLaunchArgument("mission_file", default_value=""),
        # Delay before the mission publishes the FIRST goal. The node default is
        # 3 s, too few: the goal would fire before one can start
        # metrics/record_run.sh, and /global_goal is published ONLY ONCE — a bag
        # that misses it is useless to the analysis tools. 20 s are enough to
        # start the recorder without rushing.
        DeclareLaunchArgument("mission_delay", default_value="20.0"),
    ]

    params_file  = LaunchConfiguration("params_file")
    lidar_params = LaunchConfiguration("lidar_params")
    sim_params   = LaunchConfiguration("sim_params")

    # The simulator IS the source of /clock, so it is the only node that does NOT
    # use use_sim_time; the rest of the chain does.
    sim_time = {"use_sim_time": True}
    planner  = [params_file, LaunchConfiguration("planner_overlay"), sim_time]

    nodes = [
        # The URDF has 'pelvis' as its root, while mujoco_sim publishes the base
        # pose as 'base_link'. Without this bridge the robot tree stays detached
        # from odom: RViz does not know where to put it, stacks the links at the
        # origin and draws them white with the status in error.
        # mujoco_sim imposes the pose of the free joint, which in the MJCF IS the
        # pelvis, so the transform is the identity.
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_link_to_pelvis", output="log",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "pelvis"],
            parameters=[sim_time],
            condition=IfCondition(LaunchConfiguration("robot_model")),
        ),

        # ── robot model: /joint_states -> TF of the 29 joints -> RViz ──
        # mujoco_sim already publishes /joint_states; robot_state_publisher turns
        # them into the full TF chain and into /robot_description, which is what
        # the RobotModel display of RViz consumes. Navigation does not need it:
        # with robot_model:=false the stack works identically, without the model.
        Node(
            package="robot_state_publisher", executable="robot_state_publisher",
            name="robot_state_publisher", output="log",
            parameters=[sim_time, {"robot_description": ParameterValue(
                Command(["cat ", os.path.join(g1_share, "description", "g1_29dof.urdf")]),
                value_type=str)}],
            condition=IfCondition(LaunchConfiguration("robot_model")),
        ),

        # ── impianto ────────────────────────────────────────────────────
        Node(
            package="g1_sim", executable="mujoco_sim", name="mujoco_sim",
            output="screen",
            parameters=[
                sim_params,
                {"viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                 "people": LaunchConfiguration("people"),
                 "world": LaunchConfiguration("world"),
                 "use_sim_time": False},
            ],
        ),

        # ── perception: cloud from the sensor frame to the planning frame ──
        # The adapter is fully parametric (topic, frame, range, heights, voxel):
        # the robot enters through the YAML only.

        Node(
            package="robot_real_lidar", executable="lidar_filter_node",
            name="g1_lidar_filter", output="screen",
            parameters=[lidar_params, sim_time],
        ),

        # ── pose: /odom -> PoseStamped in the planning frame ───────────
        Node(
            package="a_star_mpc_planner", executable="odom_to_pose_node",
            name="odom_to_pose_node", output="screen",
            parameters=planner,
            remappings=[("/odom/raw", "/odom")],
        ),

        # ── pianificazione e controllo ──────────────────────────────────
        Node(
            package="a_star_mpc_planner", executable="a_star_node",
            name="a_star_node", output="screen", parameters=planner,
        ),
        Node(
            package="a_star_mpc_planner", executable="mpc_node",
            name="mpc_node", output="screen", parameters=planner,
        ),
        Node(
            package="a_star_mpc_planner", executable="setpoint_to_cmd_vel_node",
            name="setpoint_to_cmd_vel_node", output="screen", parameters=planner,
        ),

        # ── memoria globale topologica (opzionale) ──────────────────────
        Node(
            package="a_star_mpc_planner", executable="nav_graph_node",
            name="nav_graph_node", output="screen", parameters=planner,
            condition=IfCondition(LaunchConfiguration("nav_graph")),
        ),

        # ── goal: the "2D Goal Pose" tool of RViz publishes on /goal_pose,
        #    the stack listens on /global_goal. The relay is parametric and forces
        #    the frame, which avoids a dependency on topic_tools ──────────
        Node(
            package="robot_real_goal_manager", executable="goal_relay_node",
            name="goal_relay_node", output="screen",
            parameters=[sim_time, {"input_topic": "/goal_pose",
                                   "output_topic": "/global_goal",
                                   "override_frame": "odom",
                                   "force_frame": True}],
            condition=IfCondition(LaunchConfiguration("goal_relay")),
        ),

        # ── missione a waypoint per prove ripetibili (opzionale) ────────
        # Needed by the experiments that require identical repeated scenarios:
        # vedi metrics/pareto_front.py (fronte di Pareto).
        Node(
            package="robot_real_goal_manager", executable="mission_runner_node",
            name="mission_runner_node", output="screen",
            parameters=[sim_time, {"mission_file": LaunchConfiguration("mission_file"),
                                   "global_goal_topic": "/global_goal",
                                   "odom_topic": "/odom",
                                   "start_delay_sec": ParameterValue(
                                       LaunchConfiguration("mission_delay"),
                                       value_type=float)}],
            condition=IfCondition(LaunchConfiguration("use_mission")),
        ),

        # ── visualizzazione ─────────────────────────────────────────────
        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[sim_time],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]

    return LaunchDescription(args + nodes)
