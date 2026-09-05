#!/usr/bin/env python3
"""
Bayesian MPC Tuner for Go2 — with full recording pipeline.

Per-trial artifacts saved to tuning_results/trial_NNN/:
  planner_params.yaml         — exact YAML used (parameter history)
  scenario_<name>/rosbag/     — ros2 bag record for every scenario
  gp_surrogate.json           — ARD-GP kernel params fit on accumulated data
  tpe_state.json              — hyperopt TPE Trials state snapshot
  metadata.json               — params, per-scenario scores, timing

Root-level artifacts:
  best_planner_params.yaml    — YAML for the best trial so far
  results.json                — all trial summaries + best
  gp_history.json             — GP kernel param evolution (length scales, noise)
  convergence.png             — score vs trial
  param_importance.png        — GP-derived parameter sensitivity
  length_scales.png           — GP length scale evolution

Note on the GP:
  hyperopt uses TPE (Tree-structured Parzen Estimators), not a GP.
  A separate ARD Matern-5/2 GP is fit on accumulated (params, score) data
  after each trial purely for analysis — short length scales = sensitive params.
"""

import copy
import json
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent.resolve()
BASE_PARAMS  = REPO_ROOT / "src/a_star_mpc_planner/config/planner_params.yaml"
RESULTS_DIR  = Path(os.environ.get("TUNING_RESULTS_DIR",
                    REPO_ROOT / "tuning_results"))
ROS_SETUP    = "/opt/ros/humble/setup.bash"
PKG_SETUP    = REPO_ROOT / "install/setup.bash"

# ─── Search space ─────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    # Position tracking
    "mpc_Q_x":              hp.uniform("mpc_Q_x",              50.0,  500.0),
    "mpc_Q_y":              hp.uniform("mpc_Q_y",              50.0,  500.0),
    "mpc_Q_yaw":            hp.uniform("mpc_Q_yaw",             0.1,   15.0),
    "mpc_Q_terminal":       hp.uniform("mpc_Q_terminal",       20.0,  300.0),
    # Obstacle avoidance
    "mpc_W_obs_sigmoid":    hp.uniform("mpc_W_obs_sigmoid",    50.0,  400.0),
    # "mpc_obs_r":            hp.uniform("mpc_obs_r",             0.35,   0.85),
    # "mpc_obs_alpha":        hp.uniform("mpc_obs_alpha",         1.0,    8.0),
    # Path following
    # "mpc_lookahead_dist":   hp.uniform("mpc_lookahead_dist",    0.5,    2.5),
    # A* soft obstacle cost
    "grid_std":            hp.uniform("grid_std",            0.1,    0.25),
}

PARAM_NAMES  = list(SEARCH_SPACE.keys())

# ─── Trial settings ───────────────────────────────────────────────────────────

MAX_EVALS         = 30
N_RANDOM_INIT     = 8
SCENARIO_TIMEOUT  = 120     # seconds to wait for all goals reached (multi-goal scenarios)
PLANNER_DELAY_SEC = 30      # seconds for Gazebo to stabilise
CLEANUP_WAIT_SEC  = 5
NAV_LOG_INTERVAL  = 5.0     # seconds between navigation status lines

# Topics recorded in every rosbag
BAG_TOPICS = [
    "/odom",
    "/go2/pose",
    "/cmd_vel",
    "/a_star/path",
    "/mpc/next_setpoint",
    "/mpc/predicted_path",
    "/mpc/diagnostics",
    "/goal_pose",
    "/tf",
    "/tf_static",
    "/lidar/points_filtered",
]

# Test scenarios across three environments.
#
# world_pkg: ROS2 package that owns the world file.
#   "go2_sim"   → share/go2_sim/worlds/<world>      (Ignition SDF)
#   "sim_worlds" → share/sim_worlds/worlds/<world>   (SDF 1.6, Gazebo-compatible)
#
# robot_x/y/heading: spawn pose (world_init_* launch args).
#
# goals: ordered list of [x, y] waypoints. The robot must reach each in sequence.
#        The first goal is also sent to the launch system via goal_x/goal_y args.
#        Use a per-scenario "timeout" key (seconds) to override SCENARIO_TIMEOUT.
#
# Weights must sum to 1.0.
SCENARIOS = [
    # ── default.sdf: open flat world — probes fundamental motion quality ──
    # Obstacles are placed ON each leg's midpoint, forcing active avoidance.
    # L-shape (0→6,0 → 6,6 → 0,6): one cylinder per leg.
    {
        "name": "open_square",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        "goals": [[6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        "obstacles": [
            {"x": 2.5, "y":  0.0},   # mid-leg 1: (0,0)→(6,0)
            {"x": 6.0, "y":  2.5},   # mid-leg 2: (6,0)→(6,6)
            {"x": 3.5, "y":  6.0},   # mid-leg 3: (6,6)→(0,6)
        ],
        "weight": 0.10,
    },
    # Zigzag: (0,0)→(-4,5)→(4,5)→(0,10). One obstacle per leg.
    {
        "name": "open_zigzag",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        "goals": [[-4.0, 5.0], [4.0, 5.0], [0.0, 10.0]],
        "obstacles": [
            {"x": -2.0, "y": 2.5},   # mid-leg 1
            {"x":  0.0, "y": 5.0},   # mid-leg 2: the reversal point
            {"x":  2.0, "y": 7.5},   # mid-leg 3
        ],
        "weight": 0.10,
    },
    # ── warehouse.world: 30×20 m (x: -15..15, y: -10..10) ──
    # Shelves at y=±6 (depth ±0.5 m). Aisles: central (|y|<5.5), north/south (|y|>6.5).
    # Cross-aisles at x≈-4 (gap between row-A and row-B) and x≈3.5 (row-B to row-C).
    #
    # warehouse_loop: south corridor → east turn → central aisle → north corridor.
    # Obstacles placed along each leg among the existing shelf structure.
    {
        "name": "warehouse_loop",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -10.0, "robot_y": -8.0, "robot_heading": 0.0,
        "goals": [[10.0, -8.0], [10.0, 0.0], [-10.0, 0.0], [-10.0, 8.0]],
        "obstacles": [
            {"x":  0.0, "y": -8.0},   # south corridor (leg 1 mid)
            {"x": 10.0, "y": -4.0},   # east cross-aisle (leg 2 mid)
            {"x":  2.0, "y":  0.0},   # central aisle east half (leg 3)
            {"x": -6.0, "y":  0.0},   # central aisle west half (leg 3)
        ],
        "weight": 0.25,
        "timeout": 180,
    },
    # warehouse_cross_aisle: north-south at x=-4 through the shelf cross-aisle,
    # then east along central aisle, then south. Obstacles block each leg midpoint.
    {
        "name": "warehouse_cross_aisle",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": 8.0, "robot_heading": -1.5708,
        "goals": [[-4.0, 0.0], [4.0, 0.0], [4.0, -8.0]],
        "obstacles": [
            {"x": -4.0, "y":  4.0},   # mid of southward leg 1
            {"x":  0.0, "y":  0.5},   # mid of eastward leg 2 (offset from centreline)
            {"x":  4.0, "y": -4.0},   # mid of southward leg 3
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    # ── indoor_office.world: 20×15 m (x: -10..10, y: -7.5..7.5) ──
    # Reception desk (0,-5.5). Cubicle dividers at y=±1.5.
    # Meeting room NW, server room NE (north corridor x: -3.15..3.1).
    #
    # office_traverse: east of reception, north through open east corridor,
    # then into the meeting/server north corridor. Obstacles mid each leg.
    {
        "name": "office_traverse",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": 2.0, "robot_y": -6.0, "robot_heading": 1.5708,
        "goals": [[4.0, 0.0], [4.0, 3.5], [1.0, 6.0]],
        "obstacles": [
            {"x": 3.0, "y": -3.0},   # mid of leg 1 (start → 4,0)
            {"x": 4.2, "y":  2.0},   # mid of leg 2 (north, east of all dividers)
            {"x": 2.5, "y":  5.0},   # mid of leg 3 (into north corridor)
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    # office_corridor: west side → tight 0.7 m divider gap at x=-4 → east crossing.
    # Obstacles on legs between and after the structural dividers.
    {
        "name": "office_corridor",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": -6.0, "robot_heading": 1.5708,
        "goals": [[-4.0, 0.0], [4.0, 0.0], [4.0, 3.5]],
        "obstacles": [
            {"x": -3.5, "y": -3.0},  # mid of northward leg 1 (before tight gap)
            {"x":  0.0, "y": -0.5},  # mid of eastward leg 2 (between divider rows)
            {"x":  3.8, "y":  2.0},  # mid of northward leg 3
        ],
        "weight": 0.15,
        "timeout": 150,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _source_cmd(cmd: str) -> str:
    return f"source {ROS_SETUP} && source {PKG_SETUP} && {cmd}"


def _log_wait(seconds: float, prefix: str = "") -> None:
    """Sleep for `seconds`, printing a progress line every 10 s."""
    step = 10
    elapsed = 0.0
    while elapsed < seconds:
        chunk = min(step, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining = seconds - elapsed
        if remaining > 0:
            print(f"{prefix} …{remaining:.0f}s remaining", flush=True)


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ─── YAML snapshot builder ────────────────────────────────────────────────────

def build_trial_yaml(base: dict, params: dict, trial_num: int) -> dict:
    """
    Deep-copy the base YAML and patch in the tuned parameter values.
    Adds a _tuning_meta block for traceability without touching real params.
    """
    trial = copy.deepcopy(base)
    ros_params = trial["/**"]["ros__parameters"]

    for key, val in params.items():
        if key in ros_params:
            ros_params[key] = float(val)
        else:
            # Parameter not in base — add it (shouldn't happen with current space)
            ros_params[key] = float(val)

    # Traceability block (prefixed so it's clearly not a real ROS param)
    ros_params["_tuning_trial"]     = trial_num
    ros_params["_tuning_timestamp"] = datetime.astimezone(datetime.now()).isoformat() + "Z"
    return trial


# ─── Rosbag recorder ─────────────────────────────────────────────────────────

class RosbagRecorder:
    """Thin wrapper around `ros2 bag record` subprocess."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._proc = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        topics = " ".join(BAG_TOPICS)
        cmd = _source_cmd(
            f"ros2 bag record {topics} --output {self.output_dir}/bag"
        )
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=5)
        except Exception:
            pass
        self._proc = None


# ─── Simulation manager ───────────────────────────────────────────────────────

class SimulationManager:
    """Launch / kill the Gazebo + planner stack for one scenario."""

    def __init__(self, gui: bool = False):
        self._proc = None
        self._gui  = gui

    def launch(self, params_yaml: Path, scenario: dict) -> None:
        # Resolve world path — supports both go2_sim and sim_worlds packages
        world_rel = scenario["world"]
        world_pkg = scenario.get("world_pkg", "go2_sim")
        pkg_prefix = subprocess.check_output(
            ["bash", "-c", _source_cmd(f"ros2 pkg prefix {world_pkg} 2>/dev/null")],
            text=True,
        ).strip()
        world_path = f"{pkg_prefix}/share/{world_pkg}/worlds/{world_rel}"

        robot_x       = scenario.get("robot_x", 0.0)
        robot_y       = scenario.get("robot_y", 0.0)
        robot_heading = scenario.get("robot_heading", 0.0)

        # Support both legacy goal_x/goal_y and new multi-goal "goals" list.
        goals = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
        first_gx, first_gy = goals[0]

        cmd = _source_cmd(
            "ros2 launch robot_sim sim_a_star_mpc.launch.py"
            f" gui:={str(self._gui).lower()}"
            f" use_rviz:={str(self._gui).lower()}"
            f" planner_params:={params_yaml}"
            f" wait_for_goal:=false"
            f" goal_x:={first_gx}"
            f" goal_y:={first_gy}"
            f" goal_z:=0.0"
            f" planner_delay_sec:={PLANNER_DELAY_SEC}"
            f" world:={world_path}"
            f" world_init_x:={robot_x}"
            f" world_init_y:={robot_y}"
            f" world_init_heading:={robot_heading}"
        )
        sim_log = RESULTS_DIR / "sim_launch.log"
        sim_log.parent.mkdir(parents=True, exist_ok=True)
        self._sim_log_fh = open(sim_log, "w")
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setsid,
            stdout=self._sim_log_fh,
            stderr=self._sim_log_fh,
        )
        time.sleep(3)
        if self._proc.poll() is not None:
            raise RuntimeError(f"Simulation failed to start — see {sim_log}")

    def spawn_obstacles(self, scenario: dict) -> None:
        """Spawn path-blocking obstacles via the sim_scenarios CLI (uses /spawn_entity)."""
        obstacles = scenario.get("obstacles", [])
        for i, obs in enumerate(obstacles):
            model = obs.get("model", "obstacle_cylinder")
            name  = f"tuner_obs_{i}"
            cmd = _source_cmd(
                f"ros2 run sim_scenarios spawn_obstacle"
                f" --name {name}"
                f" --model {model}"
                f" --x {obs['x']}"
                f" --y {obs['y']}"
                f" --z 0.5"
            )
            try:
                subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=12,
                )
            except subprocess.TimeoutExpired:
                pass  # non-fatal — scenario continues without this obstacle

    def kill(self) -> None:
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=8)
            except Exception:
                pass
            self._proc = None
        if hasattr(self, "_sim_log_fh") and self._sim_log_fh:
            self._sim_log_fh.close()
            self._sim_log_fh = None
        # Belt-and-suspenders: nuke any lingering processes
        for pattern in [
            "ign gazebo", "gzserver", "gzclient",
            "a_star_node", "mpc_node", "setpoint_to_cmd_vel",
            "odom_to_pose", "cloud_self_filter",
        ]:
            subprocess.call(
                ["pkill", "-9", "-f", pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        time.sleep(CLEANUP_WAIT_SEC)


# ─── ROS 2 performance monitor ────────────────────────────────────────────────

class PerformanceMonitor(Node):
    """
    Records trajectory, commands, LiDAR closest-obstacle distance, and MPC
    diagnostics during a single scenario run.

    MPC diagnostics topic layout (/mpc/diagnostics Float64MultiArray):
      [0] success   — IPOPT solve success (1.0 / 0.0)
      [1] cost      — total MPC objective value
      [2] solve_ms  — wall-clock solve time [ms]
      [3] avg_ms    — rolling average solve time [ms]
      [4] fails     — consecutive IPOPT failure count
      [5] security  — security-protocol active (1.0 / 0.0)
      [6] vx_eff    — effective vx limit after adaptive clamping
    """

    # Index aliases for /mpc/diagnostics fields
    _DIAG_SUCCESS  = 0
    _DIAG_COST     = 1
    _DIAG_SOLVE_MS = 2
    _DIAG_AVG_MS   = 3
    _DIAG_FAILS    = 4
    _DIAG_SECURITY = 5
    _DIAG_VX_EFF   = 6

    def __init__(self):
        super().__init__("performance_monitor")
        self.trajectory: list    = []   # [(t, x, y, yaw)]
        self.cmd_history: list   = []   # [(t, vx, vy, wz)]
        self.mpc_diag: list      = []   # [(t, success, cost, solve_ms, avg_ms, fails, security, vx_eff)]
        self.predicted_paths: list = [] # [(t, n_waypoints, last_wx, last_wy)] — lightweight summary
        self.obs_dist_history: list = [] # [(t, min_dist_this_scan)] — per-scan closest obstacle
        self.n_cloud_msgs: int   = 0    # number of LiDAR scans received (0 = detection failure)
        self.min_obs_dist: float = float("inf")
        self.recording   = False
        self.start_time  = None
        self.goal_pos    = None

        # Sensor topics use BEST_EFFORT — match publisher QoS to avoid dropping msgs
        _sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PoseStamped,      "/go2/pose",              self._on_pose,  10)
        self.create_subscription(Twist,            "/cmd_vel",               self._on_cmd,   10)
        self.create_subscription(PointCloud2,      "/lidar/points_filtered", self._on_cloud,  _sensor_qos)
        self.create_subscription(Float64MultiArray,"/mpc/diagnostics",       self._on_diag,  10)
        self.create_subscription(NavPath,          "/mpc/predicted_path",    self._on_path,   5)

        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def publish_goal(self, goal_x: float, goal_y: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal_x)
        msg.pose.position.y = float(goal_y)
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self.goal_pos = (goal_x, goal_y)

    def start(self, goal_x: float, goal_y: float) -> None:
        self.trajectory.clear()
        self.cmd_history.clear()
        self.mpc_diag.clear()
        self.predicted_paths.clear()
        self.obs_dist_history.clear()
        self.n_cloud_msgs = 0
        self.min_obs_dist = float("inf")
        self.start_time   = time.time()
        self.recording    = True
        self.publish_goal(goal_x, goal_y)

    def stop(self) -> None:
        self.recording = False

    def _on_pose(self, msg: PoseStamped) -> None:
        if not self.recording:
            return
        q   = msg.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
        self.trajectory.append((
            time.time() - self.start_time,
            msg.pose.position.x,
            msg.pose.position.y,
            float(yaw),
        ))

    def _on_cmd(self, msg: Twist) -> None:
        if not self.recording:
            return
        self.cmd_history.append((
            time.time() - self.start_time,
            msg.linear.x, msg.linear.y, msg.angular.z,
        ))

    def _on_diag(self, msg: Float64MultiArray) -> None:
        if not self.recording:
            return
        d = msg.data
        if len(d) < 7:
            return
        self.mpc_diag.append((
            time.time() - self.start_time,
            float(d[self._DIAG_SUCCESS]),
            float(d[self._DIAG_COST]),
            float(d[self._DIAG_SOLVE_MS]),
            float(d[self._DIAG_AVG_MS]),
            float(d[self._DIAG_FAILS]),
            float(d[self._DIAG_SECURITY]),
            float(d[self._DIAG_VX_EFF]),
        ))

    def _on_path(self, msg: NavPath) -> None:
        if not self.recording or not msg.poses:
            return
        last = msg.poses[-1].pose.position
        self.predicted_paths.append((
            time.time() - self.start_time,
            len(msg.poses),
            float(last.x),
            float(last.y),
        ))

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.recording or msg.width == 0:
            return
        try:
            import struct
            point_step = msg.point_step
            n = min(msg.width * msg.height, 200)
            dists = []
            for i in range(n):
                off = i * point_step
                x, y, z = struct.unpack_from("fff", bytes(msg.data[off:off+12]))
                dists.append(np.hypot(x, y))
            if dists:
                scan_min = min(dists)
                self.min_obs_dist = min(self.min_obs_dist, scan_min)
                self.obs_dist_history.append((
                    time.time() - self.start_time,
                    scan_min,
                ))
                self.n_cloud_msgs += 1
        except Exception:
            pass


# ─── Score computation ────────────────────────────────────────────────────────

def compute_score(monitor: PerformanceMonitor, goal: tuple, goals_reached_frac: float = 1.0) -> tuple:
    """
    Compute a composite score in [0, 1] from the recorded scenario data.
    goal: final waypoint (x, y) used for distance/progress metrics.
    goals_reached_frac: fraction of ordered waypoints reached (0.0 – 1.0).
    Returns (score, metrics_dict).
    """
    if not monitor.trajectory:
        return 0.0, {"error": "no trajectory"}

    traj  = np.array([(x, y) for _, x, y, _ in monitor.trajectory])
    start = traj[0]
    final = traj[-1]
    goal_arr = np.array(goal)

    dist_to_goal  = float(np.linalg.norm(final - goal_arr))
    initial_dist  = float(np.linalg.norm(start - goal_arr))
    goal_reached  = dist_to_goal < 0.5
    progress_frac = max(0.0, (initial_dist - dist_to_goal) / max(initial_dist, 0.01))

    # Path efficiency
    if len(traj) > 1:
        path_len  = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
        efficiency = min(1.0, initial_dist / max(path_len, initial_dist))
    else:
        path_len, efficiency = 0.0, 0.0

    # Control smoothness (jerk proxy)
    if len(monitor.cmd_history) > 3:
        cmds       = np.array([(vx, vy, wz) for _, vx, vy, wz in monitor.cmd_history])
        mean_jerk  = float(np.mean(np.abs(np.diff(cmds, n=2, axis=0))))
        smoothness = float(np.exp(-mean_jerk / 2.0))
    else:
        mean_jerk, smoothness = 0.0, 0.0

    # Obstacle avoidance — continuous score derived from per-scan distances
    # Requires at least 5 LiDAR scans; penalises if detection failed entirely.
    _DANGER_THRESH  = 0.3   # m — critical proximity
    _WARNING_THRESH = 0.6   # m — caution zone
    obstacle_detected = monitor.n_cloud_msgs >= 5
    if not obstacle_detected:
        # No LiDAR data received — treat as sensor failure, score 0
        obs_avoidance_score = 0.0
        danger_frac  = float("nan")
        warning_frac = float("nan")
        mean_clearance = float("nan")
    else:
        scan_dists = np.array([d for _, d in monitor.obs_dist_history])
        danger_frac   = float(np.mean(scan_dists < _DANGER_THRESH))
        warning_frac  = float(np.mean(
            (scan_dists >= _DANGER_THRESH) & (scan_dists < _WARNING_THRESH)
        ))
        mean_clearance = float(np.mean(np.minimum(scan_dists, 2.0)))
        # Weights: stay out of danger (50%), minimise warning-zone time (30%),
        #          reward higher mean clearance up to 2 m (20%)
        obs_avoidance_score = (
            (1.0 - danger_frac)  * 0.50
            + (1.0 - warning_frac) * 0.30
            + min(mean_clearance / 2.0, 1.0) * 0.20
        )

    # Time efficiency (only if goal reached)
    if goal_reached and monitor.trajectory:
        elapsed      = monitor.trajectory[-1][0]
        expected_sec = initial_dist / 0.5
        time_score   = min(1.0, expected_sec / max(elapsed, 0.1))
    else:
        time_score = 0.0

    if goal_reached:
        # goals_reached_frac == 1.0 when all waypoints reached; gives bonus for multi-goal completion.
        score = (0.25 * 1.0
                 + 0.15 * goals_reached_frac
                 + 0.18 * efficiency
                 + 0.12 * smoothness
                 + 0.20 * obs_avoidance_score
                 + 0.10 * time_score)
    else:
        # Max achievable ~0.60 — implicit goal-reaching penalty.
        score = (0.20 * goals_reached_frac
                 + 0.15 * progress_frac
                 + 0.08 * efficiency
                 + 0.08 * smoothness
                 + 0.09 * obs_avoidance_score)

    metrics = {
        "goal_reached":        goal_reached,
        "goals_reached_frac":  float(goals_reached_frac),
        "dist_to_goal":        dist_to_goal,
        "progress_frac":       progress_frac,
        "path_length":         path_len,
        "efficiency":          efficiency,
        "mean_jerk":           mean_jerk,
        "smoothness":          smoothness,
        "min_obs_dist":        float(monitor.min_obs_dist),
        "obstacle_detected":   obstacle_detected,
        "n_cloud_msgs":        monitor.n_cloud_msgs,
        "obs_danger_frac":     danger_frac,
        "obs_warning_frac":    warning_frac,
        "obs_mean_clearance":  mean_clearance,
        "obs_avoidance_score": float(obs_avoidance_score),
        "time_score":          time_score,
        "elapsed_sec":         float(monitor.trajectory[-1][0]) if monitor.trajectory else 0.0,
        "n_traj_points":       len(monitor.trajectory),
        "n_cmd_points":        len(monitor.cmd_history),
        "score":               float(score),
    }

    # ── MPC diagnostics summary (from /mpc/diagnostics) ──────────────────
    if monitor.mpc_diag:
        diag = np.array(monitor.mpc_diag)   # shape (N, 8): t,success,cost,solve_ms,avg_ms,fails,security,vx_eff
        n_solves         = len(diag)
        success_rate     = float(diag[:, 1].mean())          # fraction of solves that succeeded
        mean_cost        = float(diag[:, 2].mean())
        mean_solve_ms    = float(diag[:, 3].mean())
        max_solve_ms     = float(diag[:, 3].max())
        mean_avg_ms      = float(diag[:, 4].mean())
        total_ipopt_fails= float(diag[:, 5].max())           # peak consecutive failures
        security_frac    = float(diag[:, 6].mean())          # fraction of time in security mode
        mean_vx_eff      = float(diag[:, 7].mean())          # mean effective vx limit
        metrics.update({
            "mpc_n_solves":        n_solves,
            "mpc_success_rate":    success_rate,
            "mpc_mean_cost":       mean_cost,
            "mpc_mean_solve_ms":   mean_solve_ms,
            "mpc_max_solve_ms":    max_solve_ms,
            "mpc_mean_avg_ms":     mean_avg_ms,
            "mpc_peak_fails":      total_ipopt_fails,
            "mpc_security_frac":   security_frac,
            "mpc_mean_vx_eff":     mean_vx_eff,
        })
    else:
        metrics["mpc_n_solves"] = 0

    # ── Predicted-path summary (from /mpc/predicted_path) ────────────────
    if monitor.predicted_paths:
        metrics["mpc_n_predicted_paths"] = len(monitor.predicted_paths)
        metrics["mpc_mean_horizon_pts"]  = float(
            np.mean([p[1] for p in monitor.predicted_paths])
        )
    else:
        metrics["mpc_n_predicted_paths"] = 0

    return float(score), metrics


# ─── GP surrogate analysis ────────────────────────────────────────────────────

def fit_gp_surrogate(history: list) -> dict:
    """
    Fit an ARD Matern-5/2 GP on all (params, score) data collected so far.

    This GP is separate from hyperopt's TPE — it's used purely to extract:
      - Length scales per parameter (short scale = sensitive parameter)
      - Signal variance and noise level
      - Normalised parameter sensitivity / importance

    Returns a dict ready to be saved as gp_surrogate.json.
    """
    if len(history) < 3:
        return {"skipped": "need at least 3 observations", "n": len(history)}

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel, Matern, WhiteKernel,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"skipped": "scikit-learn not installed"}

    # Build (X, y) matrix
    X = np.array([[t["params"][p] for p in PARAM_NAMES] for t in history])
    y = np.array([t["score"] for t in history])

    # Normalise inputs so length scales are comparable
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    n_dims = X_s.shape[1]
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(1e-2, 1e2)] * n_dims,
            nu=2.5,
        )
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr.fit(X_s, y)

    fitted = gpr.kernel_
    theta  = fitted.theta.tolist()   # log-scale hyperparameters

    # Extract component kernels:
    #   ConstantKernel * Matern  → k1 (product)
    #   k1.k2  → Matern
    #   + WhiteKernel → k2 (sum)
    try:
        matern       = fitted.k1.k2          # ConstantKernel * Matern → .k2 is Matern
        constant_val = float(fitted.k1.k1.constant_value)
        length_scales= matern.length_scale.tolist()
        noise_level  = float(fitted.k2.noise_level)
    except Exception as e:
        return {"error": str(e), "theta": theta, "n": len(history)}

    # Sensitivity = inverse length scale (normalised so they sum to 1)
    inv_ls     = [1.0 / ls for ls in length_scales]
    total_inv  = sum(inv_ls) or 1.0
    sensitivity = {name: float(v / total_inv)
                   for name, v in zip(PARAM_NAMES, inv_ls)}

    # GP posterior mean at the best observed point
    best_idx   = int(np.argmax(y))
    best_x     = X_s[best_idx:best_idx+1]
    gp_mean_at_best, gp_std_at_best = gpr.predict(best_x, return_std=True)

    return {
        "n_observations":    len(history),
        "kernel_theta_log":  theta,
        "constant_value":    constant_val,
        "length_scales":     {n: float(ls)
                              for n, ls in zip(PARAM_NAMES, length_scales)},
        "noise_level":       noise_level,
        "param_sensitivity": sensitivity,
        "gp_mean_at_best":   float(gp_mean_at_best[0]),
        "gp_std_at_best":    float(gp_std_at_best[0]),
        "scaler_mean":       scaler.mean_.tolist(),
        "scaler_scale":      scaler.scale_.tolist(),
    }


def serialize_tpe_state(trials: Trials, trial_num: int) -> dict:
    """
    Snapshot of hyperopt's TPE Trials object after `trial_num` observations.
    Captures the raw trial data that drives the TPE model.
    """
    losses = [t["result"].get("loss") for t in trials.trials]
    # best_trial raises AllTrialsFailed when called inside the objective
    # (current trial not yet committed) — guard with try/except
    try:
        best_t = trials.best_trial if trials.trials else {}
    except Exception:
        best_t = {}

    return {
        "trial":       trial_num,
        "n_trials":    len(trials.trials),
        "losses":      [float(l) if l is not None else None for l in losses],
        "best_loss":   float(best_t.get("result", {}).get("loss", float("inf"))),
        "best_tid":    best_t.get("tid"),
        # Raw parameter values for all completed trials
        "Xi": [
            {k: float(v[0]) if v else None
             for k, v in t["misc"]["vals"].items()}
            for t in trials.trials
            if t["result"].get("status") == STATUS_OK
        ],
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def _get_plt():
    """Lazy matplotlib import — keeps the tuner runnable when system mpl is broken."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}) — skipping plot", flush=True)
        return None


def _plot_convergence(results: list, out: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    scores      = [r["score"] for r in results]
    best_so_far = [max(scores[:i+1]) for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(1, len(scores)+1), scores, alpha=0.6, s=30, label="Trial score")
    ax.plot(range(1, len(scores)+1), best_so_far, "r-", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Score"); ax.set_title("MPC Tuning — Convergence")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _plot_param_importance(gp_history: list, out: Path) -> None:
    valid = [s for s in gp_history if "param_sensitivity" in s]
    if not valid:
        return
    plt = _get_plt()
    if plt is None:
        return
    trials = [s["trial"] for s in valid]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in PARAM_NAMES:
        vals = [s["param_sensitivity"][name] for s in valid]
        ax.plot(trials, vals, marker="o", ms=4, label=name)
    ax.set_xlabel("Trial"); ax.set_ylabel("Normalised sensitivity (inv length-scale)")
    ax.set_title("Parameter Importance from GP Surrogate")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_length_scales(gp_history: list, out: Path) -> None:
    valid = [s for s in gp_history if "length_scales" in s]
    if not valid:
        return
    plt = _get_plt()
    if plt is None:
        return
    trials = [s["trial"] for s in valid]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in PARAM_NAMES:
        vals = [s["length_scales"][name] for s in valid]
        ax.semilogy(trials, vals, marker="o", ms=4, label=name)
    ax.set_xlabel("Trial"); ax.set_ylabel("GP length scale (log)")
    ax.set_title("GP Length Scales — Landscape Understanding Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


# ─── Main tuner ───────────────────────────────────────────────────────────────

class BayesianMPCTuner:

    def __init__(self, gui: bool = False):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._base_params  = _load_yaml(BASE_PARAMS)
        self._sim          = SimulationManager(gui=gui)
        self._trial_num    = 0
        self._max_evals    = MAX_EVALS
        self._history: list[dict] = []
        self._gp_history:  list[dict] = []
        self._best_score   = -np.inf
        self._best_params  = None
        self._best_trial   = -1
        self._hp_trials    = Trials()
        self._run_start_t  = None

    # ── Per-scenario runner ───────────────────────────────────────────────────

    def _run_scenario(
        self,
        scenario: dict,
        params_yaml: Path,
        trial_dir: Path,
    ) -> dict:
        scenario_dir = trial_dir / f"scenario_{scenario['name']}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        bag = RosbagRecorder(scenario_dir / "rosbag")

        try:
            self._sim.launch(params_yaml, scenario)
            bag.start()

            # Let Gazebo and the ROS bridge come up, then inject path obstacles
            print(f"    [setup] waiting 10 s for Gazebo bridge…", flush=True)
            time.sleep(10)
            print(f"    [setup] spawning obstacles…", flush=True)
            self._sim.spawn_obstacles(scenario)
            remaining_delay = max(PLANNER_DELAY_SEC - 10, 5)
            print(f"    [setup] waiting {remaining_delay} s for planner to connect…", flush=True)
            _log_wait(remaining_delay, prefix="    [setup]")

            # Start ROS 2 monitor
            if rclpy.ok():
                rclpy.shutdown()
            rclpy.init()
            monitor = PerformanceMonitor()

            # Multi-goal: extract ordered waypoints
            goals = scenario.get(
                "goals",
                [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]],
            )
            timeout = scenario.get("timeout", SCENARIO_TIMEOUT)

            monitor.start(goals[0][0], goals[0][1])
            print(f"    [nav] starting — {len(goals)} goal(s), timeout {timeout}s", flush=True)

            # Spin until all goals reached or timeout
            current_idx   = 0
            goals_reached = [False] * len(goals)
            end_t         = time.time() + timeout
            last_log_t    = 0.0
            nav_start_t   = time.time()
            while time.time() < end_t:
                rclpy.spin_once(monitor, timeout_sec=0.1)
                now = time.time()
                if monitor.trajectory:
                    _, x, y, _ = monitor.trajectory[-1]
                    gx, gy = goals[current_idx]
                    dist = np.hypot(x - gx, y - gy)
                    if dist < 0.5:
                        goals_reached[current_idx] = True
                        print(
                            f"    [nav] goal {current_idx+1}/{len(goals)} reached  "
                            f"pos=({x:.2f},{y:.2f})  "
                            f"elapsed={now-nav_start_t:.0f}s",
                            flush=True,
                        )
                        current_idx += 1
                        if current_idx >= len(goals):
                            break
                        monitor.publish_goal(goals[current_idx][0], goals[current_idx][1])
                        print(
                            f"    [nav] → next goal {current_idx+1}/{len(goals)}: "
                            f"({goals[current_idx][0]:.1f},{goals[current_idx][1]:.1f})",
                            flush=True,
                        )

                    # Periodic status line
                    if now - last_log_t >= NAV_LOG_INTERVAL:
                        elapsed   = now - nav_start_t
                        remaining = end_t - now
                        vx = vy = 0.0
                        if monitor.cmd_history:
                            _, vx, vy, _ = monitor.cmd_history[-1]
                        mpc_ok_pct = solve_ms = float("nan")
                        if monitor.mpc_diag:
                            recent = monitor.mpc_diag[-1]
                            mpc_ok_pct = sum(1 for d in monitor.mpc_diag[-20:] if d[1] > 0.5) / min(len(monitor.mpc_diag), 20) * 100
                            solve_ms   = recent[3]
                        print(
                            f"    [nav {elapsed:5.1f}s / {timeout}s remaining {remaining:.0f}s]  "
                            f"pos=({x:.2f},{y:.2f})  "
                            f"goal[{current_idx+1}/{len(goals)}]=({gx:.1f},{gy:.1f})  "
                            f"dist={dist:.2f}m  "
                            f"cmd=({vx:.2f},{vy:.2f})  "
                            f"mpc_ok={mpc_ok_pct:.0f}%  solve={solve_ms:.1f}ms",
                            flush=True,
                        )
                        last_log_t = now
                elif now - last_log_t >= NAV_LOG_INTERVAL:
                    elapsed   = now - nav_start_t
                    remaining = end_t - now
                    print(
                        f"    [nav {elapsed:5.1f}s / {timeout}s remaining {remaining:.0f}s]  "
                        f"waiting for pose…",
                        flush=True,
                    )
                    last_log_t = now

            goals_reached_frac = sum(goals_reached) / len(goals)
            final_goal = tuple(goals[-1])
            print(
                f"    [nav] done — goals {sum(goals_reached)}/{len(goals)}  "
                f"frac={goals_reached_frac:.0%}  "
                f"elapsed={time.time()-nav_start_t:.0f}s",
                flush=True,
            )

            monitor.stop()
            score, metrics = compute_score(monitor, final_goal, goals_reached_frac=goals_reached_frac)
            metrics["n_goals"]           = len(goals)
            metrics["goals_reached_frac"] = goals_reached_frac
            monitor.destroy_node()
            rclpy.shutdown()

        except Exception as e:
            print(f"    [ERROR] {scenario['name']}: {e}", flush=True)
            score   = 0.0
            metrics = {"error": str(e), "score": 0.0}
        finally:
            bag.stop()
            self._sim.kill()

        metrics["scenario"] = scenario["name"]
        metrics["weight"]   = scenario["weight"]
        return metrics

    # ── Objective function for hyperopt ──────────────────────────────────────

    def _objective(self, params: dict) -> dict:
        self._trial_num += 1
        trial_num  = self._trial_num
        trial_dir  = RESULTS_DIR / f"trial_{trial_num:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # ETA: average seconds per completed trial × remaining trials
        now = time.time()
        if self._run_start_t is None:
            self._run_start_t = now
        completed = trial_num - 1
        if completed > 0:
            avg_sec = (now - self._run_start_t) / completed
            eta_min = avg_sec * (self._max_evals - completed) / 60
            eta_str = f"  ETA ~{eta_min:.0f}min"
        else:
            eta_str = ""

        n_random  = getattr(self, "_n_random", N_RANDOM_INIT)
        mode_tag  = "RANDOM INIT" if trial_num <= n_random else "TPE-GUIDED"
        print(f"\n{'='*60}", flush=True)
        print(
            f"  Trial {trial_num:03d}/{self._max_evals}"
            f"  [{mode_tag}]"
            f"  {datetime.now().strftime('%H:%M:%S')}"
            f"{eta_str}",
            flush=True,
        )
        print(f"  { {k: f'{v:.3f}' for k, v in params.items()} }", flush=True)
        print(f"{'='*60}", flush=True)

        # 1. Write per-trial YAML snapshot
        trial_yaml_data  = build_trial_yaml(self._base_params, params, trial_num)
        trial_params_path = trial_dir / "planner_params.yaml"
        _save_yaml(trial_yaml_data, trial_params_path)

        # 2. Run each scenario
        t0                = time.time()
        scenario_results  = []
        aggregate_score   = 0.0

        n_scenarios = len(SCENARIOS)
        for sc_idx, scenario in enumerate(SCENARIOS, start=1):
            sc_goals = scenario.get(
                "goals",
                [[scenario.get("goal_x"), scenario.get("goal_y")]],
            )
            print(
                f"\n  ── Scenario {sc_idx}/{n_scenarios}: {scenario['name']}"
                f"  ({len(sc_goals)} goals, final {sc_goals[-1][0]},{sc_goals[-1][1]})"
                f"  weight={scenario['weight']:.2f} ──",
                flush=True,
            )
            result          = self._run_scenario(scenario, trial_params_path, trial_dir)
            weighted        = result.get("score", 0.0) * scenario["weight"]
            aggregate_score += weighted
            scenario_results.append(result)
            obs_det = result.get("obstacle_detected")
            obs_sc  = result.get("obs_avoidance_score", float("nan"))
            grf     = result.get("goals_reached_frac", float("nan"))
            n_goals = result.get("n_goals", len(sc_goals))
            print(
                f"    goals={grf:.0%}/{n_goals}  "
                f"final_reached={result.get('goal_reached')}  "
                f"score={result.get('score', 0):.3f}  "
                f"weighted={weighted:.4f}  "
                f"running_total={aggregate_score:.4f}  "
                f"dist={result.get('dist_to_goal', float('nan')):.2f}  "
                f"obs_avoidance={obs_sc:.3f}  "
                f"n_scans={result.get('n_cloud_msgs', 0)}",
                flush=True,
            )

        elapsed = time.time() - t0

        # 3. Fit GP surrogate on all data so far and extract kernel params
        self._history.append({
            "trial":  trial_num,
            "params": {k: float(v) for k, v in params.items()},
            "score":  float(aggregate_score),
        })
        print(f"\n  {'─'*56}", flush=True)
        print(f"  [GP] fitting surrogate on {len(self._history)} observations…", flush=True)
        gp_state = fit_gp_surrogate(self._history)
        if "param_sensitivity" in gp_state:
            top = sorted(gp_state["param_sensitivity"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = "  ".join(f"{k}={v:.3f}" for k, v in top)
            print(f"  [GP] ✓ active  —  noise={gp_state.get('noise_level', float('nan')):.4f}"
                  f"  gp_mean_at_best={gp_state.get('gp_mean_at_best', float('nan')):.4f}"
                  f"  ±{gp_state.get('gp_std_at_best', float('nan')):.4f}", flush=True)
            print(f"  [GP] top-5 sensitivity: {top_str}", flush=True)
        elif "skipped" in gp_state:
            reason = gp_state["skipped"]
            if "scikit-learn" in reason:
                print(f"  [GP] ✗ DISABLED — scikit-learn not installed  →  pip install scikit-learn", flush=True)
            else:
                n_have = gp_state.get("n", len(self._history))
                n_need = 3
                print(f"  [GP] ⏳ warming up — {n_have}/{n_need} observations (needs {n_need} to start)", flush=True)
        elif "error" in gp_state:
            print(f"  [GP] ✗ fit error — {gp_state['error']}", flush=True)
        print(f"  {'─'*56}", flush=True)
        gp_state["trial"] = trial_num
        gp_state["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self._gp_history.append(gp_state)
        _save_json(gp_state, trial_dir / "gp_surrogate.json")

        # 4. Snapshot TPE Trials state
        tpe_state = serialize_tpe_state(self._hp_trials, trial_num)
        _save_json(tpe_state, trial_dir / "tpe_state.json")

        # 5. Full trial metadata
        metadata = {
            "trial":           trial_num,
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "params":          {k: float(v) for k, v in params.items()},
            "aggregate_score": float(aggregate_score),
            "elapsed_sec":     float(elapsed),
            "scenarios":       scenario_results,
            "gp_summary": {
                "n_observations":    gp_state.get("n_observations"),
                "noise_level":       gp_state.get("noise_level"),
                "param_sensitivity": gp_state.get("param_sensitivity"),
            },
        }
        _save_json(metadata, trial_dir / "metadata.json")

        # 6. Track best and copy YAML
        prev_best = self._best_score
        is_new_best = aggregate_score > self._best_score
        if is_new_best:
            self._best_score  = aggregate_score
            self._best_params = {k: float(v) for k, v in params.items()}
            self._best_trial  = trial_num
            shutil.copy(trial_params_path, RESULTS_DIR / "best_planner_params.yaml")

        # 7. Persist cumulative results + plots
        self._persist(trial_num)

        new_best_tag = f"  *** NEW BEST (+{aggregate_score - prev_best:.4f}) ***" if is_new_best else ""
        gp_status = (
            f"GP=active(n={gp_state.get('n_observations')})"
            if "param_sensitivity" in gp_state
            else f"GP=disabled({gp_state.get('skipped', 'error')})"
            if "skipped" in gp_state
            else "GP=error"
        )
        print(
            f"\n  [trial {trial_num:03d}/{self._max_evals}]"
            f"  aggregate={aggregate_score:.4f}"
            f"  best={self._best_score:.4f} (trial {self._best_trial:03d})"
            f"  elapsed={elapsed/60:.1f}min"
            f"  {gp_status}"
            f"{new_best_tag}",
            flush=True,
        )

        # hyperopt minimises loss
        return {"loss": -aggregate_score, "status": STATUS_OK, "score": aggregate_score}

    def _persist(self, trial_num: int) -> None:
        summary = {
            "best_score":  float(self._best_score),
            "best_trial":  self._best_trial,
            "best_params": self._best_params,
            "trials":      [
                {
                    "trial":  t["trial"],
                    "params": t["params"],
                    "score":  t["score"],
                    "is_best": t["trial"] == self._best_trial,
                }
                for t in self._history
            ],
        }
        _save_json(summary,          RESULTS_DIR / "results.json")
        _save_json(self._gp_history, RESULTS_DIR / "gp_history.json")

        _plot_convergence(
            [{"score": t["score"]} for t in self._history],
            RESULTS_DIR / "convergence.png",
        )
        _plot_param_importance(self._gp_history, RESULTS_DIR / "param_importance.png")
        _plot_length_scales(self._gp_history,    RESULTS_DIR / "length_scales.png")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, max_evals: int = MAX_EVALS, n_random: int = N_RANDOM_INIT) -> None:
        self._n_random = n_random          # store so _objective can read it

        # ── dependency check ────────────────────────────────────────────────
        try:
            import sklearn  # noqa: F401
            sklearn_ok = True
            sklearn_ver = sklearn.__version__
        except ImportError:
            sklearn_ok = False
            sklearn_ver = "NOT INSTALLED"

        print(f"\n{'#'*60}")
        print(f"# Bayesian MPC Tuner — Go2 Gazebo")
        print(f"# Trials: {max_evals}  |  Random init: {n_random}  |  TPE-guided: {max_evals - n_random}")
        print(f"# Scenarios per trial: {len(SCENARIOS)}")
        print(f"# Parameters: {len(PARAM_NAMES)}")
        secs_per_trial = sum(PLANNER_DELAY_SEC + s.get("timeout", SCENARIO_TIMEOUT) + 10 for s in SCENARIOS)
        print(f"# Est. time: ~{max_evals * secs_per_trial / 3600:.1f}h")
        print(f"# Results: {RESULTS_DIR}")
        print(f"#")
        if sklearn_ok:
            print(f"# [OK]  scikit-learn {sklearn_ver} — GP surrogate active from trial {n_random + 1}")
        else:
            print(f"# [!!]  scikit-learn NOT INSTALLED — GP surrogate DISABLED")
            print(f"#       Run: pip install scikit-learn")
            print(f"#       All trials will use TPE only (no parameter sensitivity)")
        print(f"{'#'*60}\n")

        self._max_evals = max_evals
        try:
            fmin(
                fn=self._objective,
                space=SEARCH_SPACE,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=self._hp_trials,
                rstate=np.random.default_rng(42),
                show_progressbar=False,
            )
        except KeyboardInterrupt:
            print("\n[Interrupted] Saving partial results…")
        finally:
            self._persist(self._trial_num)

        print(f"\n{'='*60}")
        print(f"  COMPLETE — best score: {self._best_score:.4f} (trial {self._best_trial})")
        if self._best_params:
            for k, v in self._best_params.items():
                print(f"    {k}: {v:.4f}")
        print(f"  Deploy:  cp {RESULTS_DIR}/best_planner_params.yaml \\")
        print(f"              {BASE_PARAMS}")
        print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Bayesian MPC tuner for Go2")
    ap.add_argument("--trials",   type=int,  default=MAX_EVALS,    help="Total trials")
    ap.add_argument("--random",   type=int,  default=N_RANDOM_INIT, help="Random init trials")
    ap.add_argument("--gui",      action="store_true",             help="Launch Gazebo and RViz with GUI (default: headless)")
    args = ap.parse_args()

    BayesianMPCTuner(gui=args.gui).run(max_evals=args.trials, n_random=args.random)
