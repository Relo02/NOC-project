"""
lowlevel_bridge — Unitree DDS low-level bridge for the MuJoCo physics sim.

Turns the navigation MuJoCo sim into a robot that the AMO policy (RoboJuDo) can
drive, EXACTLY like the real G1 and like unitree_sim_isaaclab:

    AMO/RoboJuDo  ──rt/lowcmd──▶  this bridge ──τ──▶  MuJoCo physics
    AMO/RoboJuDo  ◀──rt/lowstate── this bridge ◀──q,dq,imu── MuJoCo

So the same `UnitreeEnv` (DDS backend) that talks to hardware also talks to
MuJoCo — no policy change. This mirrors:
  - the joint-level PD law of RoboJuDo's MujocoEnv.step()
    (policy_runtime/robojudo/environment/mujoco_env.py),
  - the rt/lowcmd subscribe / rt/lowstate publish of unitree_sim_isaaclab's
    G1RobotDDS,
  - the field/ordering conventions of UnitreeEnv
    (policy_runtime/robojudo/environment/unitree_env.py) and robot_state.py.

NOTE — validation pending: AMO must actually run against this to confirm the
mode_machine handshake, the qd/dq field names of the unitree_hg IDL, and that
the policy walks in this MuJoCo world (sim2sim). See sim/isaac/README.md and the
mujoco_sim docstring for the checklist.
"""

import struct
import threading
import time

import numpy as np

# unitree_hg DDS (same packages used by robot_state.py and RoboJuDo)
from unitree_sdk2py.core.channel import (  # type: ignore
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # type: ignore
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_  # type: ignore


# G1 29-DOF motor order = Unitree G1JointIndex (0..28). MUST match
# policypilot/state/robot_state.py (_joint_index_to_ros_name) and the joint
# names in the MJCF / 29dof.urdf. Validated against the model at startup.
G1_MOTOR_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NUM_MOTORS = len(G1_MOTOR_JOINT_NAMES)


def _quat_to_rpy(w, x, y, z):
    """MuJoCo (w,x,y,z) quaternion -> roll, pitch, yaw."""
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


class LowLevelBridge:
    """Bridges rt/lowcmd <-> MuJoCo joint torques <-> rt/lowstate for the G1.

    Build the joint<->motor map from the compiled model, then each control tick:
      apply_lowcmd(data)   read latest rt/lowcmd, set data.ctrl (PD torque)
      publish_lowstate(data) build & send rt/lowstate from data (q, dq, IMU)
    """

    def __init__(self, model, mj, free_qpos_adr, net_if="",
                 lowcmd_topic="rt/lowcmd", lowstate_topic="rt/lowstate",
                 mode_machine=1, cmd_vx_max=0.5, cmd_vy_max=0.4, cmd_wz_max=0.4,
                 cmd_active_timeout=0.5, logger=None):
        self._mj = mj
        self._model = model
        self._free_qadr = int(free_qpos_adr)
        self._mode_machine = int(mode_machine)
        self._log = logger
        self._tick = 0

        # ---- joint <-> actuator map (Unitree motor order) ----
        # name -> (ctrl index, qpos address, dof/qvel address)
        name_to_addr = {}
        for a in range(model.nu):
            jnt = int(model.actuator_trnid[a, 0])
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt)
            if jname is None:
                continue
            name_to_addr[jname] = (a, int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt]))

        self._ctrl_idx = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._qpos_adr = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._qvel_adr = np.zeros(NUM_MOTORS, dtype=np.int32)
        missing = []
        for i, jn in enumerate(G1_MOTOR_JOINT_NAMES):
            if jn not in name_to_addr:
                missing.append(jn)
                continue
            c, qp, qv = name_to_addr[jn]
            self._ctrl_idx[i] = c
            self._qpos_adr[i] = qp
            self._qvel_adr[i] = qv
        if missing:
            raise RuntimeError(
                "[lowlevel_bridge] MJCF is missing actuated joints required by the "
                f"G1 motor map: {missing}\nAvailable actuated joints: "
                f"{sorted(name_to_addr.keys())}")

        # actuator force range for torque clamping (MuJoCo also clamps if the
        # actuator is forcelimited; we clip defensively to be safe).
        fr = model.actuator_forcerange.copy()
        self._tau_max = np.where(fr[:, 1] > fr[:, 0], fr[:, 1], np.inf)[self._ctrl_idx]
        self._tau_min = np.where(fr[:, 1] > fr[:, 0], fr[:, 0], -np.inf)[self._ctrl_idx]

        # latest command buffers (filled by the DDS callback)
        self._lock = threading.Lock()
        self._q_des = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._qd_des = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._kp = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._kd = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._tau_ff = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._got_cmd = False
        self._cmd_stamp = 0.0           # time.monotonic() of the last rt/lowcmd
        self._cmd_active_timeout = float(cmd_active_timeout)

        # latest base velocity command (ROS /cmd_vel). In sim AMO has no gamepad,
        # so we encode this into rt/lowstate.wireless_remote (its only velocity
        # input). Maxes mirror G1AmoPolicyCfg.commands_map so m/s -> stick is exact.
        self._vlock = threading.Lock()
        self._cmd_vx = self._cmd_vy = self._cmd_wz = 0.0
        self._cmd_vx_max = float(cmd_vx_max)
        self._cmd_vy_max = float(cmd_vy_max)
        self._cmd_wz_max = float(cmd_wz_max)

        # ---- DDS setup ----
        try:
            ChannelFactoryInitialize(0, net_if) if net_if else ChannelFactoryInitialize(0)
        except Exception as e:  # already initialized elsewhere in the process
            if self._log:
                self._log.warn(f"[lowlevel_bridge] ChannelFactoryInitialize: {e}")
        self._state_msg = unitree_hg_msg_dds__LowState_()
        self._state_pub = ChannelPublisher(lowstate_topic, LowState_)
        self._state_pub.Init()
        self._cmd_sub = ChannelSubscriber(lowcmd_topic, LowCmd_)
        self._cmd_sub.Init(self._on_lowcmd, 10)
        if self._log:
            self._log.info(
                f"[lowlevel_bridge] DDS up: sub '{lowcmd_topic}', pub '{lowstate_topic}', "
                f"{NUM_MOTORS} motors mapped.")

    # ---- rt/lowcmd (from AMO) ----
    def _on_lowcmd(self, msg: LowCmd_):
        q = np.empty(NUM_MOTORS); qd = np.empty(NUM_MOTORS)
        kp = np.empty(NUM_MOTORS); kd = np.empty(NUM_MOTORS); tau = np.empty(NUM_MOTORS)
        mc = msg.motor_cmd
        for i in range(NUM_MOTORS):
            m = mc[i]
            q[i] = m.q
            # cmd velocity field is `qd` in the hg IDL (set_cmd_i); fall back to dq.
            qd[i] = getattr(m, "qd", None) if hasattr(m, "qd") else getattr(m, "dq", 0.0)
            kp[i] = m.kp
            kd[i] = m.kd
            tau[i] = m.tau
        with self._lock:
            self._q_des, self._qd_des = q, qd
            self._kp, self._kd, self._tau_ff = kp, kd, tau
            self._got_cmd = True
            self._cmd_stamp = time.monotonic()

    def has_command(self):
        """True while AMO is ACTIVELY sending rt/lowcmd (within the active
        timeout). Goes back to False when AMO stops/crashes, so the sim
        re-freezes the robot to a standing pose — a restarted AMO then takes
        over from a STANDING robot instead of one already collapsed."""
        with self._lock:
            return (time.monotonic() - self._cmd_stamp) < self._cmd_active_timeout

    # ---- velocity command (ROS /cmd_vel -> AMO gamepad) ----
    def set_cmd_vel(self, vx, vy, wz):
        """Store the latest base velocity command (body frame). Encoded into
        rt/lowstate.wireless_remote in publish_lowstate so AMO reads it as its
        gamepad stick (AMO has no other velocity input in simulation)."""
        with self._vlock:
            self._cmd_vx, self._cmd_vy, self._cmd_wz = float(vx), float(vy), float(wz)

    def _encode_wireless_remote(self):
        """Pack the latest /cmd_vel into a 40-byte Unitree wireless-remote frame.

        Inverts AMO's (linear) command_remap so the commanded m/s reach the
        policy:  LeftY = vx/vx_max (forward), LeftX = -wz/wz_max (yaw),
        RightX = -vy/vy_max (lateral); each saturated to the stick range
        [-1, 1]. Offsets match unitreeRemoteController.parse (struct '<f')."""
        with self._vlock:
            vx, vy, wz = self._cmd_vx, self._cmd_vy, self._cmd_wz
        c = lambda v: -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)
        ly = c(vx / self._cmd_vx_max) if self._cmd_vx_max else 0.0
        lx = c(-wz / self._cmd_wz_max) if self._cmd_wz_max else 0.0
        rx = c(-vy / self._cmd_vy_max) if self._cmd_vy_max else 0.0
        buf = bytearray(40)
        struct.pack_into("<f", buf, 4, lx)    # LeftX
        struct.pack_into("<f", buf, 8, rx)    # RightX
        struct.pack_into("<f", buf, 12, 0.0)  # RightY (unused by AMO)
        struct.pack_into("<f", buf, 20, ly)   # LeftY
        return list(buf)

    def apply_lowcmd(self, data):
        """Compute the per-joint PD torque and write data.ctrl. No-op until the
        first rt/lowcmd arrives (so the robot is not driven before AMO is up)."""
        with self._lock:
            if not self._got_cmd:
                return
            q_des, qd_des = self._q_des, self._qd_des
            kp, kd, tau_ff = self._kp, self._kd, self._tau_ff
        q = data.qpos[self._qpos_adr]
        dq = data.qvel[self._qvel_adr]
        # Same PD law as RoboJuDo MujocoEnv.step / the real motor controller,
        # but with kp/kd taken from the lowcmd message (as unitree_sim_isaaclab does).
        tau = (q_des - q) * kp + (qd_des - dq) * kd + tau_ff
        tau = np.clip(tau, self._tau_min, self._tau_max)
        data.ctrl[self._ctrl_idx] = tau

    # ---- rt/lowstate (to AMO) ----
    def publish_lowstate(self, data):
        st = self._state_msg
        self._tick = (self._tick + 1) & 0xFFFFFFFF
        # AMO blocks until tick != 0 (UnitreeEnv.wait_for_low_state).
        st.tick = self._tick if self._tick != 0 else 1
        st.mode_machine = self._mode_machine

        q = data.qpos[self._qpos_adr]
        dq = data.qvel[self._qvel_adr]
        ms = st.motor_state
        for i in range(NUM_MOTORS):
            ms[i].q = float(q[i])
            ms[i].dq = float(dq[i])
            try:
                ms[i].mode = 1
            except Exception:
                pass

        # IMU from the floating base. MuJoCo qpos quat is (w,x,y,z); the hg IDL
        # quaternion is also w-first (UnitreeEnv reorders [1,2,3,0] -> xyzw).
        a = self._free_qadr
        w, x, y, z = (float(data.qpos[a + 3]), float(data.qpos[a + 4]),
                      float(data.qpos[a + 5]), float(data.qpos[a + 6]))
        # free-joint angular velocity is in the body frame = gyroscope.
        gx, gy, gz = (float(data.qvel[3]), float(data.qvel[4]), float(data.qvel[5]))
        roll, pitch, yaw = _quat_to_rpy(w, x, y, z)
        imu = st.imu_state
        imu.quaternion = [w, x, y, z]
        imu.gyroscope = [gx, gy, gz]
        imu.rpy = [roll, pitch, yaw]
        try:
            imu.accelerometer = [0.0, 0.0, 9.81]   # AMO g1 path does not use it
        except Exception:
            pass

        # /cmd_vel -> gamepad stick so AMO walks (its only velocity input in sim)
        try:
            st.wireless_remote = self._encode_wireless_remote()
        except Exception:
            pass

        self._state_pub.Write(st)

    def shutdown(self):
        try:
            self._cmd_sub.Close()
        except Exception:
            pass
