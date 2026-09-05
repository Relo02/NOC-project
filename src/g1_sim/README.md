# g1_sim — MuJoCo plant of the Unitree G1

MuJoCo instead of Gazebo: in Gazebo the G1 would have no **source of motion**
without writing one from scratch — either an RL policy (which speaks the Unitree
DDS contract, not Gazebo) or a plugin that teleports the base, that is, the same
kinematic plant MuJoCo already provides.

Derived from the material of the CIHR laboratory (ROS package `policypilot`),
reduced to simulation only: Nav2, slam_toolbox and maps are not needed, because
this stack plans **without a prior map**, with rolling-horizon A\* on the
Gaussian grid.

---

## Why the kinematic mode

In kinematic mode (default) `mujoco_sim` integrates the base pose from
`/cmd_vel`:

```
x_{k+1}   = x_k   + (vx·cos(yaw_k) − vy·sin(yaw_k))·dt
y_{k+1}   = y_k   + (vx·sin(yaw_k) + vy·cos(yaw_k))·dt
yaw_{k+1} = yaw_k + wz·dt
```

which is **exactly** the holonomic SE(2) model the MPC optimises over, with zero
actuator time constants. The model/plant mismatch is therefore nil by
construction.

It is a deliberate choice, not a fallback: it makes the optimization experiments
(IPOPT iterations, conditioning, warm start, exact penalty, active set vs
interior point, see [`metrics/`](../../metrics/)) measurements of the
**solver**, not of gait noise.

The `physics:=true` mode (walking under physics with the AMO RL policy on the
Unitree DDS bus) is present in the code but **not used**: it requires torch,
`unitree_sdk2py` and `cyclonedds` in a separate environment, and on the source
material the walking is not verified.

---

## ROS interface

| direction | topic | type |
|---|---|---|
| IN | `/cmd_vel` | `geometry_msgs/Twist` (body vx, vy, wz) |
| OUT | `/odom` | `nav_msgs/Odometry` |
| OUT | `/livox/lidar` | `sensor_msgs/PointCloud2`, frame `mid360_sim` |
| OUT | `/clock` | `rosgraph_msgs/Clock` |
| OUT | `/joint_states` | `sensor_msgs/JointState` |
| TF | `odom → base_link` | pose of the base |
| TF | `odom → mid360_sim` | taken from the MuJoCo site that casts the rays |

The TF of the sensor is published by this node and not by
`robot_state_publisher`: the cloud and the transform are then consistent by
construction, and neither the 29-DoF URDF nor its meshes are needed.

The Mid-360 is simulated with `mj_multiRay` from the `mid360` site, with the
ray-cast restricted to the geometry group of the environment: **the robot does
not map itself**, so in simulation self-filtering is unnecessary. Measured: 8640
rays in ~6 ms, i.e. 6 % of a core at the 10 Hz rate.

---

## Usage

### Plant only, driven by hand

```bash
ros2 launch g1_sim g1_sim.launch.py
ros2 run g1_sim key_teleop          # in a second terminal
```

### Full autonomous navigation stack

```bash
ros2 launch g1_sim g1_a_star_mpc.launch.py
```

Then a goal is sent with the **2D Goal Pose** tool of RViz, or:

```bash
ros2 topic pub --once /global_goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "odom"}, pose: {position: {x: 5.0, y: 3.0}}}'
```

Useful arguments:

```bash
# dynamic obstacles (3 test people)
ros2 launch g1_sim g1_a_star_mpc.launch.py people:=default

# no MuJoCo window and no RViz (headless, for measurement campaigns)
ros2 launch g1_sim g1_a_star_mpc.launch.py viewer:=false use_rviz:=false

# global topological memory
ros2 launch g1_sim g1_a_star_mpc.launch.py nav_graph:=true

# repeatable waypoint mission
ros2 launch g1_sim g1_a_star_mpc.launch.py use_mission:=true \
    mission_file:=/path/to/mission.yaml
```

---

## Contents

```
g1_sim/
├── g1_sim/
│   ├── mujoco_sim.py         ROS node: plant + simulated Mid-360 + TF
│   ├── mujoco_world.py       model construction: G1 + warehouse + people
│   ├── lowlevel_bridge.py    Unitree DDS bridge (physics:=true only, unused)
│   ├── key_teleop.py         keyboard driving
│   └── cloud_self_filter.py  removal of the support rig (needed on the real robot)
├── assets/
│   ├── g1/g1_29dof_rev_1_0.xml + meshes/   MJCF of the G1 (needed by MuJoCo)
│   └── industrial.sdf        warehouse geometry, replicated in mujoco_world.py
├── config/
│   ├── g1_sim.yaml           simulator parameters
│   └── lidar_filter_g1.yaml  LiDAR adapter parameters
├── launch/
│   ├── g1_sim.launch.py      plant only
│   └── g1_a_star_mpc.launch.py   full stack
└── rviz/g1_nav.rviz
```

## Requirements

```bash
pip install mujoco        # verified with 3.9.0
```
