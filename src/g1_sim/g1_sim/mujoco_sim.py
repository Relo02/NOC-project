"""
mujoco_sim — MuJoCo plant of the Unitree G1 for the A*+MPC stack.

MuJoCo instead of Gazebo: in Gazebo the G1 would have no source of motion
without writing one from scratch, and one would end up hand-writing the same
kinematic plant MuJoCo already provides.

Motion modes
------------
  * physics=False (DEFAULT, the only one used here) — the base is moved
    KINEMATICALLY by /cmd_vel, imposing the pose of the free joint:

        x_{k+1}   = x_k   + (vx cos(yaw_k) - vy sin(yaw_k)) dt
        y_{k+1}   = y_k   + (vx sin(yaw_k) + vy cos(yaw_k)) dt
        yaw_{k+1} = yaw_k + wz dt

    which is EXACTLY the holonomic SE(2) model the MPC optimises over (with zero
    actuator time constants). The model/plant mismatch is therefore nil by
    construction: a deliberate choice, because it makes the optimization
    experiments (IPOPT iterations, conditioning, warm start, exact penalty,
    active set vs interior point) measurements of the SOLVER and not of gait
    noise.

  * physics=True — the G1 walks under physics, driven by the AMO RL policy over
    the Unitree DDS bus (lowlevel_bridge.py). NOT used here: it requires torch,
    unitree_sdk2py and cyclonedds in a separate environment, and on the source
    material the walking is not verified. The code stays because the path is the
    same as on the real robot.

ROS interface
-------------
  IN   /cmd_vel                geometry_msgs/Twist  (body vx, vy, wz)
  OUT  /odom                   nav_msgs/Odometry
  OUT  /livox/lidar            sensor_msgs/PointCloud2, frame `lidar_frame`
  OUT  /clock                  rosgraph_msgs/Clock
  OUT  /joint_states           sensor_msgs/JointState (standing pose)
  TF   odom -> base_link       pose of the base
  TF   odom -> lidar_frame     taken from the MuJoCo site that casts the rays

Mid-360 simulated with mj_multiRay from the `mid360` site, with the ray-cast
restricted to the geometry group of the environment: the robot does not map
itself, so in simulation self-filtering is unnecessary (it is needed on the real
robot, where the LiDAR sees the support rig — see cloud_self_filter.py).

Unlike the original, the TF of the sensor is published by this node and not by
robot_state_publisher: this way the cloud and the transform are consistent by
construction, and neither the 29-DoF URDF nor its meshes are needed.

Derived from the material of the CIHR laboratory (policypilot package).
"""

import math
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor

import mujoco
import mujoco.viewer  # at module scope so 'mujoco' is not shadowed as a local in __init__

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import Header, Bool
from tf2_ros import TransformBroadcaster

from g1_sim.mujoco_world import build_model


def default_g1_mjcf() -> str:
    """
    MJCF of the G1, looked up first in the share of the installed package and
    then in the source tree (--symlink-install layout). No absolute paths.
    """
    import os
    try:
        from ament_index_python.packages import get_package_share_directory
        cand = os.path.join(get_package_share_directory('g1_sim'),
                            'assets', 'g1', 'g1_29dof_rev_1_0.xml')
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'assets', 'g1',
                                        'g1_29dof_rev_1_0.xml'))


def _mat_to_quat(R):
    """3x3 rotation matrix -> (x, y, z, w). Shepperd, most stable branch."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _stamp(t):
    s = int(t)
    return TimeMsg(sec=s, nanosec=int((t - s) * 1e9))


def _quat_to_yaw(q):
    """Yaw of a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# AMO/RoboJuDo G1 nominal standing pose (G1AmoPolicyCfg default_pos), in Unitree
# G1 motor order. The robot is frozen in THIS pose while waiting for AMO so the
# policy takes over from its own stance (knees bent) — a straight-leg handover
# topples before the policy, running below real-time on CPU, can react.
AMO_G1_DEFAULT_POS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # right leg
    0.0, 0.0, 0.0,                         # waist
    -0.3, 0.15, -0.4, 1.57, 0.0, 0.0, 0.0,  # left arm
    -0.3, -0.15, 0.4, 1.57, 0.0, 0.0, 0.0,  # right arm
]


class MujocoSim(Node):
    def __init__(self):
        super().__init__('mujoco_sim')
        self.declare_parameter('g1_xml', '')   # default resolved below
        # World: 'industrial' (the warehouse, default) or one of the worlds with
        # non-convex obstacles — long_wall, horseshoe, dead_end. See
        # mujoco_world.WORLDS.
        self.declare_parameter('world', 'industrial')
        # With true the spawn pose is decided by the WORLD (every world has a
        # starting point from which its trap makes sense); with false the
        # spawn_x/spawn_y/spawn_yaw below apply. The spawn_* values are those of
        # the warehouse, so changing world without this flag would spawn the robot
        # inside a wall.
        self.declare_parameter('use_world_spawn', True)
        self.declare_parameter('spawn_x', -12.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_yaw', 0.0)
        self.declare_parameter('base_height', 0.793)   # pelvis standing height (MJCF)
        self.declare_parameter('sim_rate_hz', 100.0)
        self.declare_parameter('lidar_rate_hz', 10.0)
        self.declare_parameter('js_rate_hz', 30.0)
        # LiDAR pattern (Mid-360 approximation, widened downward like the Gazebo SDF)
        self.declare_parameter('lidar_h_samples', 360)
        self.declare_parameter('lidar_v_samples', 24)
        self.declare_parameter('lidar_v_min', -0.60)    # rad (≈ -34°)
        self.declare_parameter('lidar_v_max', 0.9076)   # rad (≈ +52°)
        self.declare_parameter('lidar_range_min', 0.20)
        self.declare_parameter('lidar_range_max', 40.0)
        self.declare_parameter('lidar_noise_std', 0.01)
        # frames / limits
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lidar_frame', 'mid360_link')
        self.declare_parameter('vx_limit', 1.0)
        self.declare_parameter('vy_limit', 1.0)
        self.declare_parameter('wz_limit', 1.0)
        self.declare_parameter('cmd_timeout', 1.0)
        self.declare_parameter('viewer', True)
        # RENDERING rate of the passive viewer, DECOUPLED from sim_rate_hz.
        # viewer.sync() costs ~20 ms under WSLg (measured, industrial world):
        # calling it at every _sim_tick (100 Hz) would take 200% of a core, so the
        # tick slips, sim_t advances more slowly than wall time and EVERY node with
        # use_sim_time slows down in proportion (measured RTF 0.305). Decimating to
        # 30 Hz brings rendering down to ~60% of a core and the simulation clock
        # back to real time. 0 disables the decimator.
        self.declare_parameter('viewer_fps', 30.0)
        # ---- physics / AMO mode ----
        # physics=False (default): kinematic teleport from /cmd_vel (fast, for
        #   developing the nav logic).
        # physics=True: the G1 walks under physics, driven by the AMO policy over
        #   DDS (rt/lowcmd -> joint PD torque, rt/lowstate <- joint+IMU state),
        #   exactly like the real robot and like unitree_sim_isaaclab. /cmd_vel is
        #   NOT consumed here in this mode; it goes to AMO (see mujoco_sim docstring).
        self.declare_parameter('physics', False)
        self.declare_parameter('net_if', '')          # DDS network interface for the bridge
        # MuJoCo timestep in physics mode. Match RoboJuDo's MujocoEnv sim_dt
        # (env_cfgs.MujocoEnvCfg.sim_dt = 0.001): the AMO policy is designed and
        # validated at a 1 ms physics step. A coarser step (e.g. 0.005) under-
        # resolves the foot contact — the feet are 4 small (5 mm) collision
        # spheres — so the friction cone is not held and the feet SKATE outward
        # (legs splay, robot slips and falls). Keep this at 0.001 unless you also
        # retune the contact/solver in the MJCF.
        self.declare_parameter('physics_dt', 0.001)
        # Hold the standing pose until AMO's first rt/lowcmd (AMO is a balance
        # policy, not a get-up: it must take over from a standing robot).
        self.declare_parameter('prestand', True)
        # Virtual harness: keep the robot held upright (feet just touching) through
        # AMO's PASSIVE `prepare` ramp — which on the real G1 runs while the robot
        # hangs on a harness — and release it only on /mujoco_sim/start_balance
        # (publish once AMO logs 'prepare_done'). hold_release_s > 0 also auto-
        # releases that many seconds after AMO's first command, as a fallback.
        self.declare_parameter('hold_release_s', 0.0)
        # Freeze in AMO's nominal stance (knees bent) while waiting for AMO.
        self.declare_parameter('amo_stance', True)
        self.declare_parameter('stance_height', 0.78)   # pelvis height in that stance
        # Real-time factor for the physics clock: <1 runs the sim SLOWER than
        # wall time so a policy that can only achieve ~50 Hz on CPU still gets
        # its design control rate in SIM time (AMO needs ~120 Hz). 0.5 is a good
        # start in a CPU container; 1.0 = real-time.
        self.declare_parameter('realtime_factor', 1.0)
        # AMO velocity-command ranges (mirror G1AmoPolicyCfg.commands_map); used
        # to map /cmd_vel m/s -> gamepad stick in the bridge.
        self.declare_parameter('cmd_vx_max', 0.5)
        self.declare_parameter('cmd_vy_max', 0.4)
        self.declare_parameter('cmd_wz_max', 0.4)
        # Dynamic people (test obstacles). Accept EITHER a list of strings or a
        # single string (the launch passes `people:=default` as a scalar string,
        # but `people:="['line:...']"` as a string array) → dynamic typing.
        #   "line:x1,y1,x2,y2,speed"    ping-pong along a segment
        #   "circle:cx,cy,radius,speed" loop on a circle
        #   "default"                   3 standard test people
        self.declare_parameter('people', '',
                               ParameterDescriptor(dynamic_typing=True))

        g1 = self.get_parameter('g1_xml').value or default_g1_mjcf()

        self._people = self._parse_people(self.get_parameter('people').value)
        world = str(self.get_parameter('world').value)
        self.model, self.info = build_model(g1, n_people=len(self._people),
                                            world=world)
        self.data = mujoco.MjData(self.model)

        self.base_z = float(self.get_parameter('base_height').value)
        if bool(self.get_parameter('use_world_spawn').value):
            self.x, self.y, self.yaw = (float(v) for v in self.info['world_spawn'])
        else:
            self.x = float(self.get_parameter('spawn_x').value)
            self.y = float(self.get_parameter('spawn_y').value)
            self.yaw = float(self.get_parameter('spawn_yaw').value)
        gx, gy = self.info['world_goal']
        self.get_logger().info(
            f"world '{self.info['world']}': {self.info['world_desc']} | "
            f"spawn ({self.x:.1f}, {self.y:.1f}, yaw {self.yaw:.2f}) | "
            f"suggested goal ({gx:.1f}, {gy:.1f}), to be sent with 2D Goal Pose")
        self.vx = self.vy = self.wz = 0.0
        self._last_cmd_t = 0.0
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.vx_lim = float(self.get_parameter('vx_limit').value)
        self.vy_lim = float(self.get_parameter('vy_limit').value)
        self.wz_lim = float(self.get_parameter('wz_limit').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.range_min = float(self.get_parameter('lidar_range_min').value)
        self.range_max = float(self.get_parameter('lidar_range_max').value)
        self.noise_std = float(self.get_parameter('lidar_noise_std').value)

        self._apply_pose()
        self._init_people()
        mujoco.mj_forward(self.model, self.data)

        # precompute LiDAR ray directions in the sensor frame (constant)
        nh = int(self.get_parameter('lidar_h_samples').value)
        nv = int(self.get_parameter('lidar_v_samples').value)
        az = np.repeat(np.linspace(-np.pi, np.pi, nh, endpoint=False), nv)
        el = np.tile(np.linspace(float(self.get_parameter('lidar_v_min').value),
                                 float(self.get_parameter('lidar_v_max').value), nv), nh)
        self._local_dirs = np.column_stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]).astype(np.float64)
        self._nray = self._local_dirs.shape[0]
        self._gg = np.zeros(6, dtype=np.uint8)
        self._gg[self.info['lidar_group']] = 1
        self._geomid = np.full(self._nray, -1, dtype=np.int32)
        self._dist = np.zeros(self._nray, dtype=np.float64)

        # sim clock
        self.sim_t = 0.0
        self.sim_dt = 1.0 / float(self.get_parameter('sim_rate_hz').value)

        # ---- physics / AMO low-level bridge ----
        self.physics = bool(self.get_parameter('physics').value)
        self._bridge = None
        self._substeps = 1
        if self.physics:
            self.physics_dt = float(self.get_parameter('physics_dt').value)
            self.model.opt.timestep = self.physics_dt
            self._rtf = float(self.get_parameter('realtime_factor').value)
            self._sim_advance_accum = 0.0   # fractional sim-time carry (see _physics_tick)
            # Start from the MJCF nominal standing keyframe (if any) so the policy
            # has a sane initial pose, then re-apply the spawn base pose.
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            # Put the robot in AMO's nominal stance (knees bent) so the policy
            # takes over from its own default pose (see AMO_G1_DEFAULT_POS).
            if bool(self.get_parameter('amo_stance').value):
                from g1_sim.lowlevel_bridge import G1_MOTOR_JOINT_NAMES
                qadr = {}
                for j in range(self.model.njnt):
                    nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if nm:
                        qadr[nm] = int(self.model.jnt_qposadr[j])
                for jn, v in zip(G1_MOTOR_JOINT_NAMES, AMO_G1_DEFAULT_POS):
                    if jn in qadr:
                        self.data.qpos[qadr[jn]] = v
                self.base_z = float(self.get_parameter('stance_height').value)
            self._apply_pose()
            mujoco.mj_forward(self.model, self.data)
            # Correct the spawn height so the feet rest exactly on the floor. The
            # AMO bent-knee stance at the nominal pelvis height can leave the feet
            # a couple cm BELOW z=0: a static penetration that, the instant AMO
            # takes over and mj_step runs, is resolved with a strong normal impulse
            # that kicks the robot. Auto-ground removes that handover transient.
            self._ground_to_floor()
            # Standing pose held while waiting for AMO (see _physics_tick).
            self._home_qpos = self.data.qpos.copy()
            self._prestand = bool(self.get_parameter('prestand').value)
            self._amo_ready = False
            # Virtual-harness state (see _physics_tick / _release_hold_cb).
            self._hold_active = self._prestand
            self._first_cmd_wall = None
            self._hold_release_s = float(self.get_parameter('hold_release_s').value)
            self.get_logger().info(
                f"physics mode: realtime_factor={self._rtf}, "
                f"amo_stance={bool(self.get_parameter('amo_stance').value)}")
            from g1_sim.lowlevel_bridge import LowLevelBridge
            self._bridge = LowLevelBridge(
                self.model, mujoco, self.info['free_qpos_adr'],
                net_if=self.get_parameter('net_if').value,
                cmd_vx_max=float(self.get_parameter('cmd_vx_max').value),
                cmd_vy_max=float(self.get_parameter('cmd_vy_max').value),
                cmd_wz_max=float(self.get_parameter('cmd_wz_max').value),
                logger=self.get_logger())

        # ROS I/O
        self.sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        if self.physics:
            # Operator trigger to drop the startup harness once AMO is past `prepare`
            # and actively balancing (watch AMO's console for 'prepare_done').
            self.create_subscription(Bool, '/mujoco_sim/start_balance', self._release_hold_cb, 1)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        latched = QoSProfile(depth=1)
        self.lidar_pub = self.create_publisher(PointCloud2, '/livox/lidar', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(self.sim_dt, self._sim_tick)
        self.create_timer(1.0 / float(self.get_parameter('lidar_rate_hz').value), self._lidar_tick)
        self.create_timer(1.0 / float(self.get_parameter('js_rate_hz').value), self._js_tick)

        self.viewer = None
        self._viewer_next_t = 0.0
        self._viewer_fps = float(self.get_parameter('viewer_fps').value)
        if bool(self.get_parameter('viewer').value):
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # The warehouse geoms live in the LiDAR geom group, which MuJoCo
                # hides by default (default geomgroup = [1,1,1,0,0,0]); enable it
                # so the world is actually visible in the viewer.
                with self.viewer.lock():
                    self.viewer.opt.geomgroup[self.info['lidar_group']] = 1
                self.viewer.sync()
            except Exception as e:
                self.get_logger().warn(f"viewer unavailable: {e}")

        self.get_logger().info(
            f"mujoco_sim ready: G1 + warehouse, LiDAR {self._nray} rays → /livox/lidar; "
            f"spawn=({self.x:.1f},{self.y:.1f}); people={len(self._people)}")

    # ---- dynamic people ----
    @staticmethod
    def _parse_people(raw):
        """Parse the `people` param (list of strings) into motion specs.

        Each entry: "line:x1,y1,x2,y2,speed" or "circle:cx,cy,radius,speed".
        The single keyword "default" expands to the 3 standard test people
        (matching the Gazebo dynamic_people.launch.py). Empty / malformed
        entries are skipped."""
        # ROS may deliver this param as a real list, a bare string ("default"),
        # or a string that is a list literal ("[]" / "['line:...','circle:...']")
        # — the latter because the launch passes it as a plain string so that an
        # empty default does not become a (rejected) empty ROS list.
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith('['):
                import ast
                try:
                    raw = ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    raw = [s]
            else:
                raw = [raw]
        if raw and len(raw) == 1 and (raw[0] or '').strip().lower() == 'default':
            raw = ['line:-9,0,11,0,0.9', 'line:-8.5,-8,-8.5,8,0.8',
                   'circle:-5,2,1.3,0.5']
        people = []
        for s in (raw or []):
            s = (s or '').strip()
            if not s or ':' not in s:
                continue
            kind, _, rest = s.partition(':')
            kind = kind.strip().lower()
            try:
                vals = [float(v) for v in rest.split(',')]
            except ValueError:
                continue
            if kind == 'line' and len(vals) == 5:
                x1, y1, x2, y2, speed = vals
                people.append(dict(pattern='line', p1=(x1, y1), p2=(x2, y2),
                                   speed=speed, s=0.0, dir=1.0))
            elif kind == 'circle' and len(vals) == 4:
                cx, cy, radius, speed = vals
                people.append(dict(pattern='circle', cx=cx, cy=cy, radius=radius,
                                   speed=speed, theta=0.0))
        return people

    def _init_people(self):
        """Place each person at its starting pose (precompute line geometry)."""
        ids = self.info.get('person_mocap_ids', [])
        for p, mid in zip(self._people, ids):
            p['mocap'] = mid
            if p['pattern'] == 'line':
                dx, dy = p['p2'][0] - p['p1'][0], p['p2'][1] - p['p1'][1]
                p['L'] = math.hypot(dx, dy)
                p['yaw'] = math.atan2(dy, dx) if p['L'] > 1e-6 else 0.0
                x, y, yaw = p['p1'][0], p['p1'][1], p['yaw']
            else:
                x, y = p['cx'] + p['radius'], p['cy']
                yaw = math.pi / 2.0
            self._set_person(mid, x, y, yaw)

    def _set_person(self, mid, x, y, yaw):
        self.data.mocap_pos[mid] = [x, y, 0.0]
        self.data.mocap_quat[mid] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    def _step_people(self):
        dt = self.sim_dt
        for p in self._people:
            if p['pattern'] == 'circle' and p['radius'] > 1e-3:
                omega = p['speed'] / p['radius']
                p['theta'] = (p['theta'] + omega * dt) % (2.0 * math.pi)
                x = p['cx'] + p['radius'] * math.cos(p['theta'])
                y = p['cy'] + p['radius'] * math.sin(p['theta'])
                yaw = p['theta'] + math.pi / 2.0
            else:  # line ping-pong
                L = p['L']
                p['s'] += p['dir'] * p['speed'] * dt
                if p['s'] >= L:
                    p['s'], p['dir'] = L, -1.0
                elif p['s'] <= 0.0:
                    p['s'], p['dir'] = 0.0, 1.0
                t = p['s'] / L if L > 1e-6 else 0.0
                x = p['p1'][0] + t * (p['p2'][0] - p['p1'][0])
                y = p['p1'][1] + t * (p['p2'][1] - p['p1'][1])
                yaw = p['yaw'] if p['dir'] > 0 else p['yaw'] + math.pi
            self._set_person(p['mocap'], x, y, yaw)

    # ---- kinematic pose ----
    def _apply_pose(self):
        a = self.info['free_qpos_adr']
        self.data.qpos[a:a + 3] = [self.x, self.y, self.base_z]
        self.data.qpos[a + 3:a + 7] = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]

    def _ground_to_floor(self, clearance=0.001, probe_drop=0.12):
        """Adjust the floating-base height so the lowest foot collision geom rests
        on the floor (z=0), leaving a 1 mm clearance (no initial penetration).

        Method: drop the base `probe_drop` below the current stance so the feet are
        GUARANTEED to penetrate the floor (the depth is then measurable from the
        contact set), read the deepest foot/floor penetration, and set base_z so
        that point sits exactly on z=0. This is independent of the probe depth as
        long as the feet remain the lowest geoms (true for any standing stance),
        so it corrects both an initial penetration and a small initial gap. Only
        the base z is touched — the joint angles (the stance) are left untouched."""
        floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if floor_gid < 0:
            return
        a = self.info['free_qpos_adr']
        probe_z = self.base_z - probe_drop
        self.data.qpos[a + 2] = probe_z
        mujoco.mj_forward(self.model, self.data)
        pen, found = 0.0, False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if floor_gid in (c.geom1, c.geom2):
                pen = max(pen, -float(c.dist))   # dist < 0 ⇒ penetration depth
                found = True
        if found:
            self.base_z = probe_z + pen + clearance
            self.get_logger().info(
                f"auto-ground: feet on floor, base_z -> {self.base_z:.4f} m "
                f"(stance_height was {float(self.get_parameter('stance_height').value):.4f})")
        else:
            self.get_logger().warn(
                "auto-ground: no foot/floor contact at probe height; base_z unchanged.")
        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)

    def _cmd_cb(self, msg: Twist):
        self.vx = max(-self.vx_lim, min(self.vx_lim, msg.linear.x))
        self.vy = max(-self.vy_lim, min(self.vy_lim, msg.linear.y))
        self.wz = max(-self.wz_lim, min(self.wz_lim, msg.angular.z))
        self._last_cmd_t = self.sim_t
        # Physics mode: /cmd_vel is AMO's command, not a kinematic teleport.
        # Forward it to the bridge, which encodes it as AMO's gamepad stick.
        if self.physics and self._bridge is not None:
            self._bridge.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def _sim_tick(self):
        self.sim_t += self.sim_dt
        self.clock_pub.publish(Clock(clock=_stamp(self.sim_t)))

        if self.physics:
            self._physics_tick()
            return

        # ---- kinematic mode ----
        if self.sim_t - self._last_cmd_t > self.cmd_timeout:
            self.vx = self.vy = self.wz = 0.0

        moving = bool(self.vx or self.vy or self.wz)
        if moving:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (c * self.vx - s * self.vy) * self.sim_dt
            self.y += (s * self.vx + c * self.vy) * self.sim_dt
            self.yaw = math.atan2(math.sin(self.yaw + self.wz * self.sim_dt),
                                  math.cos(self.yaw + self.wz * self.sim_dt))
            self._apply_pose()
        if self._people:
            self._step_people()
        # Refresh kinematics if the robot OR any person moved (so the LiDAR
        # ray-casts against the people's current pose and the viewer updates).
        if moving or self._people:
            mujoco.mj_forward(self.model, self.data)

        self._publish_odom_tf()
        self._viewer_sync()

    def _viewer_sync(self):
        """Redraw the viewer at most at viewer_fps, in SIMULATION time.

        Called at every tick, but the actual redraw only happens when the period
        has elapsed: this way the rendering cost does not enter the integration
        step and does not drag the /clock with it."""
        if self.viewer is None or not self.viewer.is_running():
            return
        fps = self._viewer_fps
        if fps > 0.0:
            if self.sim_t < self._viewer_next_t:
                return
            self._viewer_next_t = self.sim_t + 1.0 / fps
        self.viewer.sync()

    def _publish_odom_tf(self):
        t = TransformStamped()
        t.header.stamp = _stamp(self.sim_t)
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.base_z
        t.transform.rotation.z = math.sin(self.yaw / 2)
        t.transform.rotation.w = math.cos(self.yaw / 2)
        self.tf_broadcaster.sendTransform(t)
        self._publish_odom_msg(t.transform.translation, t.transform.rotation,
                               self.vx, self.vy, self.wz)

    def _publish_odom_msg(self, translation, rotation, vx, vy, wz):
        """nav_msgs/Odometry on /odom, alongside the TF.

        The TF alone is enough to place the robot, but Nav2's controller_server
        reads the CURRENT velocity from this topic: DWB samples its trajectories
        around the speed the robot actually has, and its acceleration limits are
        applied relative to it. Without /odom the controller assumes a robot that
        is permanently at rest and plans accordingly."""
        o = Odometry()
        o.header.stamp = _stamp(self.sim_t)
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = translation.x
        o.pose.pose.position.y = translation.y
        o.pose.pose.position.z = translation.z
        o.pose.pose.orientation = rotation
        # Body-frame twist, which is what Odometry.twist is defined to carry.
        o.twist.twist.linear.x = float(vx)
        o.twist.twist.linear.y = float(vy)
        o.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(o)

    def _release_hold_cb(self, msg: Bool):
        """Operator drops the virtual harness (start AMO balancing) — see _physics_tick."""
        if msg.data and self._hold_active:
            self._hold_active = False
            self.get_logger().info(
                "harness released by /mujoco_sim/start_balance — robot is now FREE; "
                "AMO must balance it from here.")

    # ---- physics (AMO) mode ----
    def _physics_tick(self):
        # People are kinematic (mocap) — set them before stepping physics.
        if self._people:
            self._step_people()

        has_cmd = self._bridge.has_command()
        if has_cmd and self._first_cmd_wall is None:
            self._first_cmd_wall = time.monotonic()
            self.get_logger().info(
                "AMO connected (first rt/lowcmd). Holding the robot on a VIRTUAL "
                "HARNESS through AMO's `prepare` ramp. When AMO logs 'prepare_done' "
                "and is balancing, release with:  ros2 topic pub --once "
                "/mujoco_sim/start_balance std_msgs/msg/Bool '{data: true}'")
        # Optional auto-release fallback, hold_release_s after the first command.
        if (self._hold_active and self._hold_release_s > 0.0 and self._first_cmd_wall is not None
                and (time.monotonic() - self._first_cmd_wall) > self._hold_release_s):
            self._hold_active = False
            self.get_logger().info(
                f"harness auto-released after {self._hold_release_s:.0f}s — robot is now FREE.")

        if self._hold_active or not has_cmd:
            # Virtual harness. AMO is a balance/walk policy, not a get-up controller,
            # and its startup `prepare` ramp does NOT balance — on the real robot it
            # runs hanging on a harness. So freeze the standing pose (no base
            # dynamics, feet just touching) until the operator releases the harness;
            # otherwise the passively-held stance topples before AMO ever balances.
            # We also re-freeze if AMO stops commanding (crash/stop) so a restart
            # takes over from a standing robot. Still publish lowstate so AMO runs.
            self.data.qpos[:] = self._home_qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            if not self._amo_ready:
                self._amo_ready = True
                self.get_logger().info(
                    "harness off and AMO commanding — robot under policy control.")
            # Apply the latest AMO joint command (held across the steps), then advance
            # physics by realtime_factor * wall-tick of SIM time, via a fractional
            # accumulator so realtime_factor works for ANY value (e.g. 0.2 to let a
            # CPU-limited AMO keep up: the sim runs in slow motion, but AMO's achieved
            # rate becomes its design control rate in sim time).
            self._bridge.apply_lowcmd(self.data)
            self._sim_advance_accum += self._rtf * self.sim_dt
            nsteps = int(self._sim_advance_accum / self.physics_dt)
            self._sim_advance_accum -= nsteps * self.physics_dt
            for _ in range(nsteps):
                mujoco.mj_step(self.model, self.data)
        self._bridge.publish_lowstate(self.data)
        self._publish_odom_tf_full()
        self._viewer_sync()

    def _publish_odom_tf_full(self):
        """odom -> base_link from the physics floating-base pose (full 6-DOF)."""
        a = self.info['free_qpos_adr']
        t = TransformStamped()
        t.header.stamp = _stamp(self.sim_t)
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(self.data.qpos[a])
        t.transform.translation.y = float(self.data.qpos[a + 1])
        t.transform.translation.z = float(self.data.qpos[a + 2])
        t.transform.rotation.w = float(self.data.qpos[a + 3])
        t.transform.rotation.x = float(self.data.qpos[a + 4])
        t.transform.rotation.y = float(self.data.qpos[a + 5])
        t.transform.rotation.z = float(self.data.qpos[a + 6])
        self.tf_broadcaster.sendTransform(t)
        # Free-joint qvel is [linear in WORLD, angular in BODY]; Odometry.twist
        # wants the body frame, so rotate the linear part back by the yaw.
        v = self.data.qvel
        yaw = _quat_to_yaw(t.transform.rotation)
        c, s = math.cos(-yaw), math.sin(-yaw)
        vx = c * float(v[a]) - s * float(v[a + 1])
        vy = s * float(v[a]) + c * float(v[a + 1])
        self._publish_odom_msg(t.transform.translation, t.transform.rotation,
                               vx, vy, float(v[a + 5]))

    def _js_tick(self):
        js = JointState()
        js.header.stamp = _stamp(self.sim_t)
        js.name = self.info['joint_names']
        js.position = [float(self.data.qpos[a]) for a in self.info['joint_qpos_adr']]
        self.js_pub.publish(js)

    # ---- LiDAR ----
    def _lidar_tick(self):
        sid = self.info['site_id']
        pnt = self.data.site_xpos[sid].copy()
        R = self.data.site_xmat[sid].reshape(3, 3)
        # The TF of the sensor is published HERE, from the same pose as the site
        # that casts the rays: this way the cloud and the transform are consistent
        # by construction and robot_state_publisher (nor the 29-DoF URDF) is needed
        # to have the odom -> mid360_link chain. The consumer downstream
        # (lidar_filter_node) brings the cloud back into the planning frame.
        self._publish_lidar_tf(pnt, R)
        world_dirs = (R @ self._local_dirs.T).T
        mujoco.mj_multiRay(self.model, self.data, pnt, world_dirs.flatten(),
                           self._gg, 1, -1, self._geomid, self._dist, None,
                           self._nray, self.range_max)
        valid = (self._dist > self.range_min) & (self._dist < self.range_max)
        if not np.any(valid):
            return
        d = self._dist[valid].copy()
        if self.noise_std > 0.0:
            d += np.random.normal(0.0, self.noise_std, d.shape)
        # point in the sensor (mid360) frame = dist * local_dir
        pts = (self._local_dirs[valid] * d[:, None]).astype(np.float32)
        self.lidar_pub.publish(self._cloud(pts))

    def _publish_lidar_tf(self, pnt, R):
        """odom -> lidar_frame, presa direttamente dal site MuJoCo."""
        qx, qy, qz, qw = _mat_to_quat(R)
        t = TransformStamped()
        t.header.stamp = _stamp(self.sim_t)
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.lidar_frame
        t.transform.translation.x = float(pnt[0])
        t.transform.translation.y = float(pnt[1])
        t.transform.translation.z = float(pnt[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _cloud(self, pts: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header = Header(stamp=_stamp(self.sim_t), frame_id=self.lidar_frame)
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                      for n, o in (('x', 0), ('y', 4), ('z', 8))]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pts.shape[0]
        msg.is_dense = True
        msg.data = pts.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, '_bridge', None) is not None:
            try:
                node._bridge.shutdown()
            except Exception:
                pass
        if node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        # The passive MuJoCo viewer keeps a non-daemon thread alive: without this
        # the process does not terminate on SIGINT and the launch has to escalate
        # to SIGTERM/SIGKILL. In the meantime the old process keeps publishing
        # /odom and /clock, and the next session finds TWO of them in conflict
        # (ghost pose and "jump back in time" in RViz).
        # At this point the ROS shutdown is already complete.
        os._exit(0)


if __name__ == '__main__':
    main()
