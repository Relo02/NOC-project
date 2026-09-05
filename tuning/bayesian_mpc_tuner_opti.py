#!/usr/bin/env python3
"""
Bayesian MPC Tuner for Go2 — hardened for long overnight runs.

Per-trial artifacts saved to tuning_results/trial_NNN/:
  planner_params.yaml         — exact YAML used (parameter history)
  scenario_<name>/rosbag/     — ros2 bag record for every scenario
  gp_surrogate.json           — ARD-GP kernel params fit on accumulated data
  gp_data/training_data.json  — full reloadable GP data (X, y, predictions, marginals)
  gp_data/gp_model.joblib     — fitted GaussianProcessRegressor (reload with joblib.load)
  tpe_state.json              — hyperopt TPE Trials state snapshot
  metadata.json               — params, per-scenario scores, timing

Root-level artifacts:
  best_planner_params.yaml    — YAML for the best trial so far
  results.json                — all trial summaries + best
  gp_history.json             — GP kernel param evolution (length scales, noise)
  convergence.png             — score vs trial
  param_importance.png        — GP-derived parameter sensitivity
  length_scales.png           — GP length scale evolution
  gp_marginals.png            — GP posterior mean ±1σ per parameter (1-D slices)
  gp_fitted_vs_actual.png     — GP posterior mean at training points vs observed score

Reload snippet (run outside tuner after experiment):
    import joblib, json
    import numpy as np
    import matplotlib.pyplot as plt

    gpr  = joblib.load("trial_NNN/gp_data/gp_model.joblib")
    data = json.load(open("trial_NNN/gp_data/training_data.json"))

    # Marginal for any parameter
    m = data["marginals"]["mpc_Q_x"]
    plt.plot(m["x_raw"], m["mean"], label="GP mean")
    plt.fill_between(m["x_raw"], m["std_lo"], m["std_hi"], alpha=0.3, label="±1σ")
    plt.axvline(m["best_x"], color="r", linestyle="--", label="best observed")
    plt.show()

    # Fitted vs actual
    y_pred = np.array(data["y_pred_mean"])
    y_true = np.array(data["y"])
    plt.scatter(y_true, y_pred); plt.plot([0,1],[0,1],"r--"); plt.show()

Hardening changes applied (see FIX-N labels throughout):
  FIX-1  ROS 2 lifecycle: rclpy.init() once at program start, nodes destroyed per scenario
  FIX-2  Process management: PID tracking, SIGTERM→wait→SIGKILL, no pkill -9
  FIX-3  Simulation reuse: Gazebo kept alive across scenarios, reset between them
  FIX-4  Early termination: abort scenario on stall / no-progress / MPC failure rate
  FIX-5  PointCloud2 parsing: sensor_msgs_py with safe fallback
  FIX-6  Score function: MPC penalty, NaN guard, clamping, better obs weight
  FIX-7  Disk usage: topic filtering, bag compression, keep-last-N cleanup policy
  FIX-8  GP surrogate: min 10 samples, defensive error handling, smoothed importance
  FIX-9  Logging: structured with timestamps/IDs, failure reasons, heartbeat
  FIX-10 Runtime: condition-based waits, configurable timeouts, reduced idle sleep
  FIX-11 Safety guards: global trial timeout, catch-all, clean recovery
  FIX-12 Code quality: type hints, modularity, removed duplication
"""

import copy
import json
import logging
import os
import signal
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

# FIX-1: rclpy imported at top level; init/shutdown managed by the tuner lifecycle
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# ─── Structured logging setup (FIX-9) ────────────────────────────────────────

def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mpc_tuner")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-7s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / "tuner.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Module-level logger (reconfigured in BayesianMPCTuner.__init__)
log = logging.getLogger("mpc_tuner")


# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent.resolve()
BASE_PARAMS  = REPO_ROOT / "src/a_star_mpc_planner/config/planner_params.yaml"
RESULTS_DIR  = Path(os.environ.get("TUNING_RESULTS_DIR",
                    REPO_ROOT / "tuning_results"))
ROS_SETUP    = "/opt/ros/humble/setup.bash"
PKG_SETUP    = REPO_ROOT / "install/setup.bash"

# ─── Search space ─────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "mpc_Q_x":              hp.uniform("mpc_Q_x",              50.0,  500.0),
    "mpc_Q_y":              hp.uniform("mpc_Q_y",              50.0,  500.0),
    "mpc_Q_yaw":            hp.uniform("mpc_Q_yaw",             0.1,   15.0),
    "mpc_Q_terminal":       hp.uniform("mpc_Q_terminal",       20.0,  300.0),
    "mpc_W_obs_sigmoid":    hp.uniform("mpc_W_obs_sigmoid",    50.0,  400.0),
    "grid_std":             hp.uniform("grid_std",              0.1,   0.25),
}

PARAM_NAMES = list(SEARCH_SPACE.keys())

# Search bounds extracted once at module level for use in marginal grids
SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "mpc_Q_x":           (50.0,  500.0),
    "mpc_Q_y":           (50.0,  500.0),
    "mpc_Q_yaw":         (0.1,   15.0),
    "mpc_Q_terminal":    (20.0,  300.0),
    "mpc_W_obs_sigmoid": (50.0,  400.0),
    "grid_std":          (0.1,   0.25),
}

# ─── Trial settings ───────────────────────────────────────────────────────────

MAX_EVALS             = 30
N_RANDOM_INIT         = 8
SCENARIO_TIMEOUT      = 300     # default per-scenario timeout (s)
PLANNER_DELAY_SEC     = 30      # Gazebo stabilisation delay (s)
CLEANUP_WAIT_SEC      = 5
NAV_LOG_INTERVAL      = 5.0     # s between navigation status lines

# FIX-11: per-trial hard ceiling (s)
GLOBAL_TRIAL_TIMEOUT  = 1200    # 20 min absolute max per trial

# FIX-4: early-termination thresholds
EARLY_TERM_STALL_SEC        = 100.0
EARLY_TERM_PROGRESS_SEC     = 30.0
EARLY_TERM_PROGRESS_THRESH  = 0.02
EARLY_TERM_MPC_FAIL_RATE    = 0.7

# FIX-7: disk/bag policy
BAG_KEEP_LAST_N_TRIALS = 5
BAG_RECORD_ONLY_BEST   = False
BAG_COMPRESS           = True

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
]

SCENARIOS = [
    {
        "name": "open_square",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        # Goals in clear space past static boxes/cylinders; dynamic obstacles on the path
        "goals": [[7.0, 0.0], [7.0, 6.0], [0.0, 7.0]],
        "obstacles": [
            {"x": 3.5, "y":  0.0},
            {"x": 7.0, "y":  3.0},
            {"x": 3.5, "y":  6.5},
        ],
        "weight": 0.10,
    },
    {
        "name": "open_zigzag",
        "world": "default.sdf", "world_pkg": "go2_sim",
        "robot_x": 0.0, "robot_y": 0.0, "robot_heading": 0.0,
        # Goals in clear space; obstacles staggered left-right to force a zigzag path
        "goals": [[-6.0, 3.0], [6.0, 6.0], [-6.0, 9.0]],
        "obstacles": [
            {"x":  2.0, "y": 1.5},
            {"x": -2.0, "y": 4.5},
            {"x":  2.0, "y": 7.5},
        ],
        "weight": 0.10,
    },
    {
        "name": "warehouse_loop",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -10.0, "robot_y": -8.0, "robot_heading": 0.0,
        # Goals in aisle corridors (y=-8 south corridor / y=0 central / y=+8 north corridor)
        "goals": [[12.0, -8.0], [12.0, 0.0], [-12.0, 0.0], [-12.0, 8.0]],
        "obstacles": [
            {"x":  0.0, "y": -8.0},
            {"x": 12.0, "y": -4.0},
            {"x":  0.0, "y":  0.0},
            {"x": -6.0, "y":  4.0},
        ],
        "weight": 0.25,
        "timeout": 180,
    },
    {
        "name": "warehouse_cross_aisle",
        "world": "warehouse.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": 8.0, "robot_heading": -1.5708,
        # Goals in the central inter-tier corridor and south corridor (clear of all shelves)
        "goals": [[-4.0, 2.0], [4.0, 2.0], [4.0, -8.0]],
        "obstacles": [
            {"x": -4.0, "y":  5.5},
            {"x":  0.0, "y":  2.0},
            {"x":  4.0, "y": -2.0},
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    {
        "name": "office_traverse",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": 2.0, "robot_y": -6.0, "robot_heading": 1.5708,
        # Goals in open floor areas past the cubicle dividers (y=-1.5/+1.5) and server wall (y≈4)
        "goals": [[4.0, 0.0], [7.0, 3.0], [8.5, 6.0]],
        "obstacles": [
            {"x": 3.0, "y": -3.0},
            {"x": 5.5, "y":  1.5},
            {"x": 7.5, "y":  4.5},
        ],
        "weight": 0.20,
        "timeout": 150,
    },
    {
        "name": "office_corridor",
        "world": "indoor_office.world", "world_pkg": "sim_worlds",
        "robot_x": -4.0, "robot_y": -6.0, "robot_heading": 1.5708,
        # Goals in open floor areas past the cubicle dividers and north of the meeting-room wall (y≈4.25)
        "goals": [[-4.0, 0.0], [4.0, 0.0], [0.0, 6.5]],
        "obstacles": [
            {"x": -3.5, "y": -3.0},
            {"x":  0.0, "y":  0.0},
            {"x": -1.0, "y":  3.5},
        ],
        "weight": 0.15,
        "timeout": 150,
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _source_cmd(cmd: str) -> str:
    return f"source {ROS_SETUP} && source {PKG_SETUP} && {cmd}"


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
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


def _ts() -> str:  # FIX-12: type hints throughout
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ─── YAML snapshot builder ────────────────────────────────────────────────────

def build_trial_yaml(base: dict, params: dict, trial_num: int) -> dict:
    trial = copy.deepcopy(base)
    ros_params = trial["/**"]["ros__parameters"]
    for key, val in params.items():
        ros_params[key] = float(val)
    ros_params["_tuning_trial"]     = trial_num
    ros_params["_tuning_timestamp"] = datetime.astimezone(datetime.now()).isoformat() + "Z"
    return trial


# ─── Rosbag recorder (FIX-2, FIX-7) ─────────────────────────────────────────

class RosbagRecorder:
    def __init__(self, output_dir: Path, compress: bool = BAG_COMPRESS):
        self.output_dir = output_dir
        self._compress  = compress
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        topics        = " ".join(BAG_TOPICS)
        compress_flag = "--compression-mode file --compression-format zstd" if self._compress else ""
        cmd = _source_cmd(
            f"ros2 bag record {topics} {compress_flag} --output {self.output_dir}/bag"
        )
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.debug("RosbagRecorder started  pid=%d  dir=%s", self._proc.pid, self.output_dir)

    def stop(self) -> None:
        if self._proc is None:
            return
        pid = self._proc.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            self._proc.wait(timeout=8)
            log.debug("RosbagRecorder stopped  pid=%d", pid)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            log.warning("RosbagRecorder SIGTERM timed out — sending SIGKILL  pid=%d", pid)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
        except Exception as exc:
            log.warning("RosbagRecorder stop error  pid=%d  err=%s", pid, exc)
        finally:
            self._proc = None


# ─── Process kill helper (FIX-2) ─────────────────────────────────────────────

def _kill_proc(proc: subprocess.Popen, label: str, sigterm_timeout: float = 8.0) -> None:
    if proc is None:
        return
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
        log.debug("Sending SIGTERM to %s  pgid=%d", label, pgid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=sigterm_timeout)
        log.debug("%s exited cleanly", label)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        log.warning("%s SIGTERM timed out — sending SIGKILL  pid=%d", label, pid)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            proc.wait(timeout=3)
        except Exception as exc:
            log.warning("SIGKILL failed for %s  pid=%d  err=%s", label, pid, exc)
    except Exception as exc:
        log.warning("kill_proc error for %s  pid=%d  err=%s", label, pid, exc)


# ─── Simulation manager (FIX-2, FIX-3) ───────────────────────────────────────

class SimulationManager:
    """
    FIX-3: One Gazebo instance per trial. Between scenarios we reset robot pose
    and respawn obstacles instead of doing a full restart.
    """

    def __init__(self, gui: bool = False):
        self._proc: Optional[subprocess.Popen] = None
        self._gui  = gui
        self._sim_log_fh = None
        self._current_world: Optional[str] = None

    def launch(self, params_yaml: Path, scenario: dict) -> None:
        world_rel = scenario["world"]
        world_pkg = scenario.get("world_pkg", "go2_sim")

        if self._proc is not None and self._current_world == world_rel:
            log.info("  [sim] reusing Gazebo for world=%s  — resetting robot pose", world_rel)
            self._reset_robot(scenario)
            self._reset_nav_state(scenario)
            return

        if self._proc is not None:
            self.kill()

        pkg_prefix = subprocess.check_output(
            ["bash", "-c", _source_cmd(f"ros2 pkg prefix {world_pkg} 2>/dev/null")],
            text=True,
        ).strip()
        world_path = f"{pkg_prefix}/share/{world_pkg}/worlds/{world_rel}"

        robot_x       = scenario.get("robot_x", 0.0)
        robot_y       = scenario.get("robot_y", 0.0)
        robot_heading = scenario.get("robot_heading", 0.0)
        goals         = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
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
        self._current_world = world_rel

        deadline = time.time() + 10
        while time.time() < deadline:
            time.sleep(0.5)
            if self._proc.poll() is not None:
                raise RuntimeError(f"Simulation failed to start — see {sim_log}")
        log.info("  [sim] launched  pid=%d  world=%s", self._proc.pid, world_rel)

    def _reset_robot(self, scenario: dict) -> None:
        """
        Try strategies in order, each with its own short timeout.

        Strategy 1 — gz service /world/<name>/set_pose   (Gazebo Harmonic, gz-sim 8+)
        Strategy 2 — ign service /world/<name>/set_pose  (Ignition Gazebo, gz-sim 6/7)
        Strategy 3 — ros2 /world/<name>/set_entity_state bridge (ros_gz)
        Strategy 4 — Custom /reset_robot_pose service (may not exist).
        Strategy 5 — Publish to /initialpose (Nav2 AMCL only — last resort).
        """
        import math
        rx = scenario.get("robot_x", 0.0)
        ry = scenario.get("robot_y", 0.0)
        rh = scenario.get("robot_heading", 0.0)
        qz = math.sin(rh / 2.0)
        qw = math.cos(rh / 2.0)

        # World name without extension: "default.sdf" → "default"
        world_name = Path(scenario.get("world", "default.sdf")).stem

        # gz/ign transport Pose proto (text format)
        _pose_proto = (
            f"name: 'go2' "
            f"position: {{x: {rx} y: {ry} z: 0.05}} "
            f"orientation: {{x: 0.0 y: 0.0 z: {qz:.6f} w: {qw:.6f}}}"
        )
        _gz_svc = f"/world/{world_name}/set_pose"

        # ros_gz bridge SetEntityState payload (ROS 2 service)
        _bridge_payload = (
            "{state: {name: go2, pose: {position:"
            f" {{x: {rx}, y: {ry}, z: 0.05}},"
            f" orientation: {{x: 0.0, y: 0.0, z: {qz:.6f}, w: {qw:.6f}}}"
            "}}}"
        )

        _ip_payload = (
            "{header: {frame_id: map},"
            f" pose: {{pose: {{position: {{x: {rx}, y: {ry}, z: 0.0}},"
            f" orientation: {{x: 0.0, y: 0.0, z: {qz:.6f}, w: {qw:.6f}}}"
            "}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0,"
            " 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.07]}}"
        )

        strategies: list[tuple[str, str]] = [
            (
                # Gazebo Harmonic (gz-sim 8+): direct physics teleport via gz transport
                f"gz service {_gz_svc}",
                (
                    f"gz service -s {_gz_svc}"
                    f" --reqtype gz.msgs.Pose"
                    f" --reptype gz.msgs.Boolean"
                    f" --timeout 2000"
                    f" --req \"{_pose_proto}\" 2>/dev/null"
                ),
            ),
            (
                # Ignition Gazebo (gz-sim 6/7, ROS 2 Humble default)
                f"ign service {_gz_svc}",
                (
                    f"ign service -s {_gz_svc}"
                    f" --reqtype ignition.msgs.Pose"
                    f" --reptype ignition.msgs.Boolean"
                    f" --timeout 2000"
                    f" --req \"{_pose_proto}\" 2>/dev/null"
                ),
            ),
            (
                # ros_gz bridge: /world/<name>/set_entity_state (ROS 2 service)
                f"/world/{world_name}/set_entity_state",
                _source_cmd(
                    f"ros2 service call /world/{world_name}/set_entity_state"
                    f" gazebo_msgs/srv/SetEntityState '{_bridge_payload}'  2>/dev/null"
                ),
            ),
            (
                "/reset_robot_pose service",
                _source_cmd("ros2 service call /reset_robot_pose std_srvs/srv/Empty 2>/dev/null"),
            ),
            (
                "/initialpose topic",
                _source_cmd(
                    f"ros2 topic pub --once /initialpose"
                    f" geometry_msgs/msg/PoseWithCovarianceStamped '{_ip_payload}'  2>/dev/null"
                ),
            ),
        ]

        for label, cmd in strategies:
            try:
                ret = subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                if ret == 0:
                    log.info("  [sim] robot reset via '%s'  pos=(%.2f, %.2f)  heading=%.3f",
                             label, rx, ry, rh)
                    time.sleep(1.5)
                    return
                else:
                    log.debug("  [sim] reset strategy '%s' returned %d — trying next", label, ret)
            except subprocess.TimeoutExpired:
                log.debug("  [sim] reset strategy '%s' timed out (15s) — trying next", label)
            except Exception as exc:
                log.debug("  [sim] reset strategy '%s' error: %s — trying next", label, exc)

        log.warning(
            "  [sim] all robot-reset strategies failed for scenario '%s'"
            "  pos=(%.2f, %.2f) — continuing without reset",
            scenario.get("name", "?"), rx, ry,
        )

    def _reset_nav_state(self, scenario: dict) -> None:
        """
        Redirect the planner to the new scenario's first goal immediately after
        the robot is teleported, so it stops pursuing the previous scenario's goal.

        Also clears path visualisations in RViz and Nav2 costmaps (best-effort).
        """
        goals   = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
        gx, gy  = goals[0]
        _empty_path = "{header: {frame_id: 'map'}, poses: []}"
        _goal_pose  = (
            "{header: {frame_id: 'map'},"
            f" pose: {{position: {{x: {gx}, y: {gy}, z: 0.0}},"
            " orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
        )
        cmds: list[tuple[str, str]] = [
            # Redirect the planner to the new scenario goal — stops it chasing the old one
            ("redirect goal",
             _source_cmd(
                 f"ros2 topic pub --once"
                 f" --qos-reliability reliable --qos-durability transient_local"
                 f" /goal_pose geometry_msgs/msg/PoseStamped '{_goal_pose}' 2>/dev/null"
             )),
            # Clear stale path visualisations in RViz
            ("clear /a_star/path",
             _source_cmd(
                 f"ros2 topic pub --once /a_star/path"
                 f" nav_msgs/msg/Path '{_empty_path}' 2>/dev/null"
             )),
            ("clear /mpc/predicted_path",
             _source_cmd(
                 f"ros2 topic pub --once /mpc/predicted_path"
                 f" nav_msgs/msg/Path '{_empty_path}' 2>/dev/null"
             )),
            # Clear Nav2 costmaps (best-effort — no-op if not running Nav2)
            ("clear global costmap",
             _source_cmd(
                 "ros2 service call /global_costmap/clear_entirely_global_costmap"
                 " nav2_msgs/srv/ClearEntireCostmap '{}' 2>/dev/null"
             )),
            ("clear local costmap",
             _source_cmd(
                 "ros2 service call /local_costmap/clear_entirely_local_costmap"
                 " nav2_msgs/srv/ClearEntireCostmap '{}' 2>/dev/null"
             )),
        ]
        procs = []
        for label, cmd in cmds:
            try:
                p = subprocess.Popen(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                procs.append((label, p))
            except Exception as exc:
                log.debug("  [sim] nav-reset '%s' launch error: %s", label, exc)

        deadline = time.time() + 5.0
        for label, p in procs:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(timeout=remaining)
                if p.returncode == 0:
                    log.debug("  [sim] nav-reset '%s' ok", label)
                else:
                    log.debug("  [sim] nav-reset '%s' returned %d", label, p.returncode)
            except subprocess.TimeoutExpired:
                p.kill()
                log.debug("  [sim] nav-reset '%s' timed out — killed", label)
        log.info("  [sim] nav state flushed — planner redirected to (%.2f, %.2f)", gx, gy)

    def spawn_obstacles(self, scenario: dict) -> None:
        self._delete_obstacles()
        for i, obs in enumerate(scenario.get("obstacles", [])):
            model = obs.get("model", "obstacle_cylinder")
            name  = f"tuner_obs_{i}"
            cmd   = _source_cmd(
                f"ros2 run sim_scenarios spawn_obstacle"
                f" --name {name} --model {model}"
                f" --x {obs['x']} --y {obs['y']} --z 0.5"
            )
            try:
                subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=12,
                )
            except subprocess.TimeoutExpired:
                log.warning("  [sim] obstacle spawn timed out  name=%s", name)

    def _delete_obstacles(self) -> None:
        for i in range(20):
            name = f"tuner_obs_{i}"
            cmd  = _source_cmd(
                f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
                f"'{{name: {name}}}' 2>/dev/null"
            )
            try:
                subprocess.call(
                    ["bash", "-c", cmd],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=6,
                )
            except Exception:
                break

    def kill(self) -> None:
        if self._proc is not None:
            _kill_proc(self._proc, "sim_launch")
            self._proc = None
        if self._sim_log_fh:
            try:
                self._sim_log_fh.close()
            except Exception:
                pass
            self._sim_log_fh = None
        self._current_world = None

        _KNOWN_PROCS = [
            "ign gazebo", "gzserver", "gzclient",
            "a_star_node", "mpc_node", "setpoint_to_cmd_vel",
            "odom_to_pose", "cloud_self_filter",
        ]
        for pattern in _KNOWN_PROCS:
            subprocess.call(["pkill", "-TERM", "-f", pattern],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        for pattern in _KNOWN_PROCS:
            subprocess.call(["pkill", "-KILL", "-f", pattern],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(CLEANUP_WAIT_SEC)
        log.debug("  [sim] killed")


# ─── ROS 2 performance monitor (FIX-1, FIX-5) ────────────────────────────────

class PerformanceMonitor(Node):
    _DIAG_SUCCESS  = 0
    _DIAG_COST     = 1
    _DIAG_SOLVE_MS = 2
    _DIAG_AVG_MS   = 3
    _DIAG_FAILS    = 4
    _DIAG_SECURITY = 5
    _DIAG_VX_EFF   = 6

    def __init__(self):
        super().__init__("performance_monitor")
        self.trajectory: list      = []
        self.cmd_history: list     = []
        self.mpc_diag: list        = []
        self.predicted_paths: list = []
        self.obs_dist_history: list = []
        self.n_cloud_msgs: int     = 0
        self.min_obs_dist: float   = float("inf")
        self.recording             = False
        self.start_time: Optional[float] = None
        self.goal_pos: Optional[tuple]   = None
        self._last_goal: Optional[tuple] = None

        _sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # TRANSIENT_LOCAL so the planner receives the goal even if it subscribes late
        _goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped,       "/go2/pose",              self._on_pose,  10)
        self.create_subscription(Twist,             "/cmd_vel",               self._on_cmd,   10)
        self.create_subscription(PointCloud2,       "/lidar/points_filtered", self._on_cloud, _sensor_qos)
        self.create_subscription(Float64MultiArray, "/mpc/diagnostics",       self._on_diag,  10)
        self.create_subscription(NavPath,           "/mpc/predicted_path",    self._on_path,   5)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", _goal_qos)

    def publish_goal(self, goal_x: float, goal_y: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id    = "map"
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x    = float(goal_x)
        msg.pose.position.y    = float(goal_y)
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self.goal_pos   = (goal_x, goal_y)
        self._last_goal = (goal_x, goal_y)

    def republish_last_goal(self) -> None:
        if self._last_goal is not None:
            self.publish_goal(*self._last_goal)

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

    def _elapsed(self) -> float:
        return time.time() - (self.start_time or time.time())

    def _on_pose(self, msg: PoseStamped) -> None:
        if not self.recording:
            return
        q   = msg.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
        self.trajectory.append((
            self._elapsed(),
            msg.pose.position.x,
            msg.pose.position.y,
            float(yaw),
        ))

    def _on_cmd(self, msg: Twist) -> None:
        if not self.recording:
            return
        self.cmd_history.append((
            self._elapsed(),
            msg.linear.x, msg.linear.y, msg.angular.z,
        ))

    def _on_diag(self, msg: Float64MultiArray) -> None:
        if not self.recording or len(msg.data) < 7:
            return
        d = msg.data
        self.mpc_diag.append((
            self._elapsed(),
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
            self._elapsed(), len(msg.poses), float(last.x), float(last.y),
        ))

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.recording or msg.width == 0:
            return
        try:
            dists = _parse_pointcloud_dists(msg, max_points=200)
            if dists:
                scan_min = float(min(dists))
                self.min_obs_dist = min(self.min_obs_dist, scan_min)
                self.obs_dist_history.append((self._elapsed(), scan_min))
                self.n_cloud_msgs += 1
        except Exception as exc:
            log.debug("PointCloud2 parse error: %s", exc)


def _parse_pointcloud_dists(msg: PointCloud2, max_points: int = 200) -> list:  # FIX-5
    try:
        from sensor_msgs_py import point_cloud2 as pc2_utils
        points = list(pc2_utils.read_points(msg, field_names=("x", "y"), skip_nans=True))
        return [float(np.hypot(p[0], p[1])) for p in points[:max_points]]
    except ImportError:
        pass
    import struct
    point_step = msg.point_step
    n = min(msg.width * msg.height, max_points)
    dists = []
    for i in range(n):
        off = i * point_step
        x, y, _ = struct.unpack_from("fff", bytes(msg.data[off:off + 12]))
        dists.append(float(np.hypot(x, y)))
    return dists


# ─── Early termination (FIX-4) ───────────────────────────────────────────────

def _check_early_termination(
    monitor: "PerformanceMonitor",
    current_goal: tuple,
    initial_dist: float,
    now: float,
    nav_start_t: float,
) -> Optional[str]:
    # elapsed: seconds since this goal started
    elapsed = now - nav_start_t

    # Never fire in the first EARLY_TERM_PROGRESS_SEC — planner may still be replanning.
    if elapsed < EARLY_TERM_PROGRESS_SEC:
        return None

    # traj_now / goal_traj_start: trajectory-clock equivalents (p[0] units = time since
    # monitor.start_time, not since nav_start_t).  Using these ensures stall and progress
    # checks are on the same timescale as the stored trajectory points.
    traj_now       = now - (monitor.start_time or now)
    goal_traj_start = nav_start_t - (monitor.start_time or nav_start_t)

    if monitor.trajectory:
        last_pose_t = monitor.trajectory[-1][0]
        if (traj_now - last_pose_t) > EARLY_TERM_STALL_SEC:
            return f"pose stall >{EARLY_TERM_STALL_SEC:.0f}s"
    elif traj_now > EARLY_TERM_STALL_SEC:
        return f"no pose data after {EARLY_TERM_STALL_SEC:.0f}s"

    if monitor.trajectory:
        # Only consider points from the current goal onward and within the progress window
        window = [
            p for p in monitor.trajectory
            if p[0] >= goal_traj_start and (traj_now - p[0]) <= EARLY_TERM_PROGRESS_SEC
        ]
        if len(window) >= 5:
            gx, gy = current_goal
            d_old  = float(np.hypot(window[0][1] - gx, window[0][2] - gy))
            d_new  = float(np.hypot(window[-1][1] - gx, window[-1][2] - gy))
            improvement = (d_old - d_new) / max(initial_dist, 0.01)
            # Only terminate for genuine stalling (0 ≤ Δ < threshold).
            # Negative Δ means the robot is moving — possibly navigating around
            # an obstacle — so let the timeout and MPC checks handle that case.
            if 0.0 <= improvement < EARLY_TERM_PROGRESS_THRESH:
                return (
                    f"insufficient progress over {EARLY_TERM_PROGRESS_SEC:.0f}s "
                    f"(Δ={improvement:.3f} < {EARLY_TERM_PROGRESS_THRESH})"
                )

    if len(monitor.mpc_diag) >= 20:
        success_rate = sum(1 for d in monitor.mpc_diag[-20:] if d[1] > 0.5) / 20.0
        if success_rate < EARLY_TERM_MPC_FAIL_RATE:
            return f"MPC success rate {success_rate:.0%} < {EARLY_TERM_MPC_FAIL_RATE:.0%}"

    return None


# ─── Score computation (FIX-6) ────────────────────────────────────────────────

def _safe(val: float, default: float = 0.0) -> float:
    return default if not np.isfinite(val) else float(val)


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def compute_score(
    monitor: "PerformanceMonitor",
    goal: tuple,
    goals_reached_frac: float = 1.0,
) -> tuple:
    if not monitor.trajectory:
        return 0.0, {"error": "no trajectory"}

    traj     = np.array([(x, y) for _, x, y, _ in monitor.trajectory])
    start    = traj[0]
    final    = traj[-1]
    goal_arr = np.array(goal)

    dist_to_goal  = _safe(float(np.linalg.norm(final - goal_arr)))
    initial_dist  = _safe(float(np.linalg.norm(start - goal_arr)), default=1.0)
    goal_reached  = dist_to_goal < 0.5
    progress_frac = _clamp(_safe(max(0.0, (initial_dist - dist_to_goal) / max(initial_dist, 0.01))))

    if len(traj) > 1:
        path_len   = _safe(float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))))
        efficiency = _clamp(_safe(initial_dist / max(path_len, initial_dist)))
    else:
        path_len, efficiency = 0.0, 0.0

    if len(monitor.cmd_history) > 3:
        cmds       = np.array([(vx, vy, wz) for _, vx, vy, wz in monitor.cmd_history])
        mean_jerk  = _safe(float(np.mean(np.abs(np.diff(cmds, n=2, axis=0)))), default=999.0)
        smoothness = _clamp(_safe(float(np.exp(-mean_jerk / 2.0))))
    else:
        mean_jerk, smoothness = 0.0, 0.0

    _DANGER_THRESH  = 0.3
    _WARNING_THRESH = 0.6
    obstacle_detected = monitor.n_cloud_msgs >= 5
    if not obstacle_detected:
        obs_avoidance_score = 0.0
        danger_frac = warning_frac = mean_clearance = float("nan")
    else:
        scan_dists     = np.array([d for _, d in monitor.obs_dist_history])
        danger_frac    = _safe(float(np.mean(scan_dists < _DANGER_THRESH)))
        warning_frac   = _safe(float(np.mean(
            (scan_dists >= _DANGER_THRESH) & (scan_dists < _WARNING_THRESH)
        )))
        mean_clearance = _safe(float(np.mean(np.minimum(scan_dists, 2.0))))
        obs_avoidance_score = _clamp(
            (1.0 - danger_frac) * 0.50
            + (1.0 - warning_frac) * 0.30
            + min(mean_clearance / 2.0, 1.0) * 0.20
        )

    if goal_reached and monitor.trajectory:
        elapsed    = monitor.trajectory[-1][0]
        time_score = _clamp(_safe((initial_dist / 0.5) / max(elapsed, 0.1)))
    else:
        time_score = 0.0

    mpc_success_rate = 1.0
    if monitor.mpc_diag:
        diag             = np.array(monitor.mpc_diag)
        mpc_success_rate = _safe(float(diag[:, 1].mean()), default=0.0)
    mpc_penalty = _clamp(mpc_success_rate)

    if goal_reached:
        score = _clamp(
            0.25 * 1.0
            + 0.15 * _safe(goals_reached_frac)
            + 0.15 * efficiency
            + 0.10 * smoothness
            + 0.20 * obs_avoidance_score
            + 0.10 * time_score
            + 0.05 * mpc_penalty
        )
    else:
        score = _clamp(
            0.18 * _safe(goals_reached_frac)
            + 0.14 * progress_frac
            + 0.07 * efficiency
            + 0.07 * smoothness
            + 0.12 * obs_avoidance_score
            + 0.07 * mpc_penalty
        )

    metrics: dict[str, Any] = {
        "goal_reached":        goal_reached,
        "goals_reached_frac":  float(_safe(goals_reached_frac)),
        "dist_to_goal":        dist_to_goal,
        "progress_frac":       progress_frac,
        "path_length":         path_len,
        "efficiency":          efficiency,
        "mean_jerk":           mean_jerk,
        "smoothness":          smoothness,
        "min_obs_dist":        float(_safe(monitor.min_obs_dist, default=-1.0)),
        "obstacle_detected":   obstacle_detected,
        "n_cloud_msgs":        monitor.n_cloud_msgs,
        "obs_danger_frac":     danger_frac,
        "obs_warning_frac":    warning_frac,
        "obs_mean_clearance":  mean_clearance,
        "obs_avoidance_score": float(obs_avoidance_score),
        "mpc_success_rate":    mpc_success_rate,
        "mpc_penalty":         mpc_penalty,
        "time_score":          time_score,
        "elapsed_sec":         float(monitor.trajectory[-1][0]) if monitor.trajectory else 0.0,
        "n_traj_points":       len(monitor.trajectory),
        "n_cmd_points":        len(monitor.cmd_history),
        "score":               float(score),
    }

    if monitor.mpc_diag:
        diag = np.array(monitor.mpc_diag)
        metrics.update({
            "mpc_n_solves":      len(diag),
            "mpc_mean_cost":     _safe(float(diag[:, 2].mean())),
            "mpc_mean_solve_ms": _safe(float(diag[:, 3].mean())),
            "mpc_max_solve_ms":  _safe(float(diag[:, 3].max())),
            "mpc_mean_avg_ms":   _safe(float(diag[:, 4].mean())),
            "mpc_peak_fails":    _safe(float(diag[:, 5].max())),
            "mpc_security_frac": _safe(float(diag[:, 6].mean())),
            "mpc_mean_vx_eff":   _safe(float(diag[:, 7].mean())),
        })
    else:
        metrics["mpc_n_solves"] = 0

    if monitor.predicted_paths:
        metrics["mpc_n_predicted_paths"] = len(monitor.predicted_paths)
        metrics["mpc_mean_horizon_pts"]  = float(np.mean([p[1] for p in monitor.predicted_paths]))
    else:
        metrics["mpc_n_predicted_paths"] = 0

    return float(score), metrics


# ─── GP surrogate analysis (FIX-8) ───────────────────────────────────────────

_GP_IMPORTANCE_HISTORY: deque = deque(maxlen=5)


def fit_gp_surrogate(history: list) -> dict:
    """
    Fit ARD Matern-5/2 GP and return both summary scalars and full reloadable data.

    The returned dict contains:
      • All the usual summary keys (length_scales, param_sensitivity, noise_level, …)
      • "_reloadable" sub-dict with:
            X_raw          (N, D) — original parameter matrix
            y              (N,)   — observed scores
            X_scaled       (N, D) — standardised inputs
            y_pred_mean    (N,)   — GP posterior mean at every training point
            y_pred_std     (N,)   — GP posterior std  at every training point
            marginals      per-parameter 1-D slices (50 points, others fixed at best)
            search_bounds, param_names, scaler_mean, scaler_scale
      • "_gpr_object" — the live fitted GPR (not JSON-safe; used by save_gp_checkpoint)
    """
    MIN_SAMPLES = 10
    if len(history) < MIN_SAMPLES:
        return {"skipped": f"need at least {MIN_SAMPLES} observations", "n": len(history)}

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"skipped": "scikit-learn not installed"}

    try:
        X = np.array([[t["params"][p] for p in PARAM_NAMES] for t in history])
        y = np.array([t["score"] for t in history])

        if np.std(y) < 1e-8:
            return {"skipped": "y variance too low for GP fitting", "n": len(history)}

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
            kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-6,
        )
        gpr.fit(X_s, y)

        fitted        = gpr.kernel_
        matern        = fitted.k1.k2
        constant_val  = float(fitted.k1.k1.constant_value)
        length_scales = matern.length_scale.tolist()
        noise_level   = float(fitted.k2.noise_level)

        inv_ls          = [1.0 / max(ls, 1e-9) for ls in length_scales]
        total_inv       = sum(inv_ls) or 1.0
        raw_sensitivity = {n: float(v / total_inv) for n, v in zip(PARAM_NAMES, inv_ls)}

        _GP_IMPORTANCE_HISTORY.append(raw_sensitivity)
        smoothed_sensitivity = {
            name: float(np.mean([h.get(name, 0.0) for h in _GP_IMPORTANCE_HISTORY]))
            for name in PARAM_NAMES
        }

        best_idx      = int(np.argmax(y))
        best_x_s      = X_s[best_idx:best_idx + 1]
        gp_mean_best, gp_std_best = gpr.predict(best_x_s, return_std=True)

        # ── GP predictions at every observed point ────────────────────────────
        y_pred_mean, y_pred_std = gpr.predict(X_s, return_std=True)

        # ── 1-D marginal slices ───────────────────────────────────────────────
        # For each param: sweep 50 values across its search bounds,
        # all other params held at the best-observed value.
        N_GRID      = 50
        best_x_raw  = X[best_idx]
        marginals: dict = {}

        for dim_i, pname in enumerate(PARAM_NAMES):
            lo, hi   = SEARCH_BOUNDS.get(pname, (float(X[:, dim_i].min()), float(X[:, dim_i].max())))
            grid_raw = np.linspace(lo, hi, N_GRID)

            X_query_raw           = np.tile(best_x_raw, (N_GRID, 1))
            X_query_raw[:, dim_i] = grid_raw
            X_query_s             = scaler.transform(X_query_raw)

            m, s = gpr.predict(X_query_s, return_std=True)
            marginals[pname] = {
                "x_raw":  grid_raw.tolist(),
                "mean":   m.tolist(),
                "std_lo": (m - s).tolist(),   # mean − 1σ
                "std_hi": (m + s).tolist(),   # mean + 1σ
                "best_x": float(best_x_raw[dim_i]),
                "best_y": float(y[best_idx]),
                "bounds": [lo, hi],
                # Scatter: actual observed (x_i, score_i) for this parameter
                "obs_x":  X[:, dim_i].tolist(),
                "obs_y":  y.tolist(),
            }

        # ── Assemble result ───────────────────────────────────────────────────
        result = {
            # Summary scalars — go into gp_history.json
            "n_observations":        len(history),
            "kernel_theta_log":      fitted.theta.tolist(),
            "constant_value":        constant_val,
            "length_scales":         {n: float(ls) for n, ls in zip(PARAM_NAMES, length_scales)},
            "noise_level":           noise_level,
            "param_sensitivity":     smoothed_sensitivity,
            "param_sensitivity_raw": raw_sensitivity,
            "gp_mean_at_best":       float(gp_mean_best[0]),
            "gp_std_at_best":        float(gp_std_best[0]),
            "scaler_mean":           scaler.mean_.tolist(),
            "scaler_scale":          scaler.scale_.tolist(),
            # Full reloadable data — saved separately by save_gp_checkpoint()
            "_reloadable": {
                "param_names":   PARAM_NAMES,
                "search_bounds": {n: list(b) for n, b in SEARCH_BOUNDS.items()},
                "X_raw":         X.tolist(),
                "y":             y.tolist(),
                "X_scaled":      X_s.tolist(),
                "y_pred_mean":   y_pred_mean.tolist(),
                "y_pred_std":    y_pred_std.tolist(),
                "marginals":     marginals,
                "scaler_mean":   scaler.mean_.tolist(),
                "scaler_scale":  scaler.scale_.tolist(),
                "best_idx":      int(best_idx),
            },
            # Live GPR object — used by save_gp_checkpoint(); not JSON-serialisable
            "_gpr_object": gpr,
        }
        return result

    except Exception as exc:
        log.warning("[GP] fit error (non-fatal): %s", exc)
        return {"error": str(exc), "n": len(history)}


def save_gp_checkpoint(gp_state: dict, out_dir: Path) -> None:
    """
    Persist the full reloadable GP data to <out_dir>/gp_data/.

    Files written:
      training_data.json  — X_raw, y, predictions, marginals (reload without sklearn)
      gp_model.joblib     — fitted GaussianProcessRegressor  (reload with joblib)
      gp_summary.json     — scalar kernel params / sensitivity

    All three are self-contained — you can regenerate any plot offline from them.
    """
    reloadable = gp_state.get("_reloadable")
    gpr_object = gp_state.get("_gpr_object")
    if reloadable is None:
        return  # GP was skipped or errored — nothing to save

    gp_dir = out_dir / "gp_data"
    gp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full training + prediction data (JSON)
    _save_json(reloadable, gp_dir / "training_data.json")

    # 2. Fitted GPR (joblib pickle)
    if gpr_object is not None:
        try:
            import joblib
            joblib.dump(gpr_object, gp_dir / "gp_model.joblib")
            log.debug("[GP] model checkpoint saved  %s", gp_dir / "gp_model.joblib")
        except Exception as exc:
            log.warning("[GP] joblib save failed (non-fatal): %s", exc)

    # 3. Scalar summary (JSON) — same keys as gp_surrogate.json but without private keys
    summary = {k: v for k, v in gp_state.items() if not k.startswith("_")}
    _save_json(summary, gp_dir / "gp_summary.json")
    log.debug("[GP] checkpoint written  %s", gp_dir)


def _gp_state_for_json(gp_state: dict) -> dict:
    """Strip non-JSON-serialisable private keys before saving to gp_history / gp_surrogate."""
    return {k: v for k, v in gp_state.items() if not k.startswith("_")}


def serialize_tpe_state(trials: Trials, trial_num: int) -> dict:
    losses = [t["result"].get("loss") for t in trials.trials]
    try:
        best_t = trials.best_trial if trials.trials else {}
    except Exception:
        best_t = {}
    return {
        "trial":     trial_num,
        "n_trials":  len(trials.trials),
        "losses":    [float(l) if l is not None else None for l in losses],
        "best_loss": float(best_t.get("result", {}).get("loss", float("inf"))),
        "best_tid":  best_t.get("tid"),
        "Xi": [
            {k: float(v[0]) if v else None for k, v in t["misc"]["vals"].items()}
            for t in trials.trials
            if t["result"].get("status") == STATUS_OK
        ],
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        log.warning("[plot] matplotlib unavailable: %s", exc)
        return None


def _plot_convergence(results: list, out: Path) -> None:
    plt = _get_plt()
    if plt is None:
        return
    scores      = [r["score"] for r in results]
    best_so_far = [max(scores[:i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(1, len(scores) + 1), scores, alpha=0.6, s=30, label="Trial score")
    ax.plot(range(1, len(scores) + 1), best_so_far, "r-", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("Score")
    ax.set_title("MPC Tuning — Convergence")
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
    ax.set_xlabel("Trial"); ax.set_ylabel("Normalised sensitivity")
    ax.set_title("Parameter Importance from GP Surrogate (smoothed)")
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
    ax.set_title("GP Length Scales Evolution")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_gp_marginals(gp_state: dict, out: Path) -> None:
    """
    One subplot per parameter: GP posterior mean (line) ± 1σ band (fill),
    observed (x_i, score_i) scatter, and a vertical line at the best-observed value.
    Requires the "_reloadable" key produced by fit_gp_surrogate().
    """
    reloadable = gp_state.get("_reloadable")
    if not reloadable or "marginals" not in reloadable:
        return
    plt = _get_plt()
    if plt is None:
        return

    marginals  = reloadable["marginals"]
    n_params   = len(PARAM_NAMES)
    n_cols     = 3
    n_rows     = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if n_params > 1 else [axes]

    for ax, pname in zip(axes_flat, PARAM_NAMES):
        m       = marginals[pname]
        x_raw   = m["x_raw"]
        mean    = m["mean"]
        std_lo  = m["std_lo"]
        std_hi  = m["std_hi"]
        obs_x   = m["obs_x"]
        obs_y   = m["obs_y"]
        best_x  = m["best_x"]
        best_y  = m["best_y"]

        ax.fill_between(x_raw, std_lo, std_hi, alpha=0.25, color="steelblue", label="±1σ")
        ax.plot(x_raw, mean, color="steelblue", lw=2, label="GP mean")
        ax.scatter(obs_x, obs_y, s=25, color="dimgray", alpha=0.7, zorder=3, label="Observed")
        ax.axvline(best_x, color="crimson", lw=1.5, linestyle="--", label=f"Best ({best_x:.2f})")
        ax.scatter([best_x], [best_y], color="crimson", s=60, zorder=4)

        ax.set_xlabel(pname, fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(pname, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    n_obs = len(reloadable.get("y", []))
    fig.suptitle(
        f"GP Posterior Marginals — {n_obs} observations\n"
        "(each panel: one param swept, all others fixed at best-observed value)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.debug("[plot] gp_marginals saved  %s", out)


def _plot_gp_fitted_vs_actual(gp_state: dict, out: Path) -> None:
    """
    Scatter: GP posterior mean at every training point (x-axis = observed score,
    y-axis = GP prediction). Points coloured by trial order. Perfect fit = diagonal.
    Includes R² and RMSE in the title.
    """
    reloadable = gp_state.get("_reloadable")
    if not reloadable or "y" not in reloadable or "y_pred_mean" not in reloadable:
        return
    plt = _get_plt()
    if plt is None:
        return

    y_true = np.array(reloadable["y"])
    y_pred = np.array(reloadable["y_pred_mean"])
    y_std  = np.array(reloadable["y_pred_std"])
    n      = len(y_true)

    # R² and RMSE
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        y_true, y_pred,
        c=np.arange(n), cmap="viridis",
        s=60, zorder=3, alpha=0.85,
    )
    ax.errorbar(
        y_true, y_pred, yerr=y_std,
        fmt="none", ecolor="gray", alpha=0.4, lw=1, zorder=2,
    )
    # Perfect-fit diagonal
    lo = min(y_true.min(), y_pred.min()) - 0.02
    hi = max(y_true.max(), y_pred.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

    plt.colorbar(sc, ax=ax, label="Trial index")
    ax.set_xlabel("Observed score")
    ax.set_ylabel("GP predicted mean")
    ax.set_title(
        f"GP Fitted vs Actual  (n={n})\n"
        f"R²={r2:.3f}   RMSE={rmse:.4f}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.debug("[plot] gp_fitted_vs_actual saved  %s", out)


# ─── Disk cleanup (FIX-7) ────────────────────────────────────────────────────

def _cleanup_old_bags(results_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    trial_dirs = sorted(results_dir.glob("trial_???"))
    to_prune   = trial_dirs[:-keep_last_n] if len(trial_dirs) > keep_last_n else []
    for td in to_prune:
        for bag_dir in td.rglob("rosbag"):
            if bag_dir.is_dir():
                shutil.rmtree(bag_dir, ignore_errors=True)
                log.debug("[disk] removed bag  %s", bag_dir)


# ─── Main tuner ───────────────────────────────────────────────────────────────

class BayesianMPCTuner:

    def __init__(self, gui: bool = False):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        global log
        log = _setup_logger(RESULTS_DIR)

        self._base_params  = _load_yaml(BASE_PARAMS)
        self._sim          = SimulationManager(gui=gui)
        self._trial_num    = 0
        self._max_evals    = MAX_EVALS
        self._history: list[dict]    = []
        self._gp_history: list[dict] = []
        self._best_score   = -np.inf
        self._best_params: Optional[dict] = None
        self._best_trial   = -1
        self._hp_trials    = Trials()
        self._run_start_t: Optional[float] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        # Keep last GP state for _persist() to use in plots
        self._last_gp_state: Optional[dict] = None

    def _load_checkpoint(self) -> int:
        """
        Restore state from a previous interrupted run.
        Returns the number of trials already completed (0 if nothing to restore).
        """
        results_path = RESULTS_DIR / "results.json"
        if not results_path.exists():
            log.warning("[resume] %s not found — starting fresh", results_path)
            return 0

        with open(results_path) as f:
            saved = json.load(f)

        trials_data = saved.get("trials", [])
        if not trials_data:
            log.warning("[resume] results.json has no trials — starting fresh")
            return 0

        self._history = [
            {"trial": t["trial"], "params": t["params"], "score": t["score"]}
            for t in trials_data
        ]
        self._best_score  = float(saved.get("best_score", -np.inf))
        self._best_params = saved.get("best_params")
        self._best_trial  = int(saved.get("best_trial", -1))

        gp_history_path = RESULTS_DIR / "gp_history.json"
        if gp_history_path.exists():
            with open(gp_history_path) as f:
                self._gp_history = json.load(f)

        last_trial_num = trials_data[-1]["trial"]
        last_trial_dir = RESULTS_DIR / f"trial_{last_trial_num:03d}"
        hp_path        = last_trial_dir / "hp_trials.joblib"
        if hp_path.exists():
            try:
                import joblib
                self._hp_trials = joblib.load(hp_path)
                log.info("[resume] loaded hp_trials (%d entries) from %s",
                         len(self._hp_trials.trials), hp_path)
            except Exception as exc:
                log.warning("[resume] hp_trials load failed (%s) — fresh TPE state", exc)
        else:
            log.warning("[resume] %s not found — fresh TPE state (GP history still loaded)", hp_path)

        self._trial_num = last_trial_num
        log.info("[resume] restored %d trials  best=%.4f (t%03d)",
                 last_trial_num, self._best_score, self._best_trial)
        return last_trial_num

    def _ros_init(self) -> None:
        if not rclpy.ok():
            rclpy.init()
            self._executor = SingleThreadedExecutor()
            log.info("[ROS] rclpy initialised")

    def _ros_shutdown(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()
            log.info("[ROS] rclpy shutdown")

    # ── Per-scenario runner ───────────────────────────────────────────────────

    def _run_scenario(
        self,
        scenario: dict,
        params_yaml: Path,
        trial_dir: Path,
        trial_num: int,
        sc_idx: int,
    ) -> dict:
        sc_name      = scenario["name"]
        scenario_dir = trial_dir / f"scenario_{sc_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        log_prefix     = f"[T{trial_num:03d}/SC{sc_idx}/{sc_name}]"
        failure_reason: Optional[str] = None
        log.info("%s starting", log_prefix)

        bag = RosbagRecorder(scenario_dir / "rosbag")

        try:
            self._sim.launch(params_yaml, scenario)
            bag.start()

            log.info("%s waiting 10s for Gazebo bridge…", log_prefix)
            _heartbeat_sleep(10, log_prefix)
            log.info("%s spawning obstacles…", log_prefix)
            self._sim.spawn_obstacles(scenario)

            remaining_delay = max(PLANNER_DELAY_SEC - 10, 5)
            log.info("%s waiting %ds for planner…", log_prefix, remaining_delay)
            _heartbeat_sleep(remaining_delay, log_prefix)

            monitor = PerformanceMonitor()
            if self._executor:
                self._executor.add_node(monitor)

            goals   = scenario.get("goals", [[scenario.get("goal_x", 0.0), scenario.get("goal_y", 0.0)]])
            timeout = scenario.get("timeout", SCENARIO_TIMEOUT)

            monitor.start(goals[0][0], goals[0][1])
            log.info("%s nav started — %d goal(s), timeout=%ds", log_prefix, len(goals), timeout)

            current_idx     = 0
            goals_reached   = [False] * len(goals)
            end_t           = time.time() + timeout
            global_end_t    = time.time() + GLOBAL_TRIAL_TIMEOUT
            last_log_t      = 0.0
            last_goal_pub_t = time.time()
            nav_start_t     = time.time()
            # initial_dist is set from the first real pose message, not the spawn
            # config — this avoids stale-pose errors when reset fails.
            initial_dist  = float(np.hypot(
                goals[0][0] - scenario.get("robot_x", 0.0),
                goals[0][1] - scenario.get("robot_y", 0.0),
            ))
            initial_dist_set_from_pose = False  # will be overridden on first pose

            while time.time() < end_t and time.time() < global_end_t:
                if self._executor:
                    self._executor.spin_once(timeout_sec=0.05)
                else:
                    rclpy.spin_once(monitor, timeout_sec=0.05)

                now = time.time()

                # Update initial_dist from the first real pose so that if the
                # robot reset failed and the robot is elsewhere, the progress
                # metric is still meaningful.
                if not initial_dist_set_from_pose and monitor.trajectory:
                    _, rx, ry, _ = monitor.trajectory[0]
                    gx0, gy0     = goals[0]
                    initial_dist = float(np.hypot(rx - gx0, ry - gy0))
                    initial_dist_set_from_pose = True
                    log.debug(
                        "%s initial_dist set from first pose: %.2fm  pos=(%.2f,%.2f)",
                        log_prefix, initial_dist, rx, ry,
                    )

                # Republish the current goal every 5 s — guards against the planner
                # missing the one-shot publish (QoS transient / timing race).
                if (now - last_goal_pub_t) >= 5.0:
                    monitor.republish_last_goal()
                    last_goal_pub_t = now

                term_reason = _check_early_termination(
                    monitor, tuple(goals[current_idx]), initial_dist, now, nav_start_t,
                )
                if term_reason:
                    failure_reason = f"early_term: {term_reason}"
                    log.warning("%s early termination — %s", log_prefix, term_reason)
                    break

                if monitor.trajectory:
                    _, x, y, _ = monitor.trajectory[-1]
                    gx, gy = goals[current_idx]
                    dist   = float(np.hypot(x - gx, y - gy))

                    if dist < 0.5:
                        goals_reached[current_idx] = True
                        log.info(
                            "%s goal %d/%d reached  pos=(%.2f,%.2f)  elapsed=%.0fs",
                            log_prefix, current_idx + 1, len(goals), x, y, now - nav_start_t,
                        )
                        current_idx += 1
                        if current_idx >= len(goals):
                            break
                        monitor.publish_goal(goals[current_idx][0], goals[current_idx][1])
                        last_goal_pub_t = now
                        initial_dist = float(np.hypot(
                            goals[current_idx][0] - x, goals[current_idx][1] - y,
                        ))
                        nav_start_t = now  # reset grace period for the new goal

                    if now - last_log_t >= NAV_LOG_INTERVAL:
                        vx = vy = 0.0
                        if monitor.cmd_history:
                            _, vx, vy, _ = monitor.cmd_history[-1]
                        mpc_ok_pct = solve_ms = float("nan")
                        if monitor.mpc_diag:
                            mpc_ok_pct = (
                                sum(1 for d in monitor.mpc_diag[-20:] if d[1] > 0.5)
                                / min(len(monitor.mpc_diag), 20) * 100
                            )
                            solve_ms = monitor.mpc_diag[-1][3]
                        log.info(
                            "%s %.1fs/%.0fs  pos=(%.2f,%.2f)  dist=%.2fm"
                            "  cmd=(%.2f,%.2f)  mpc_ok=%.0f%%  solve=%.1fms",
                            log_prefix, now - nav_start_t, timeout, x, y, dist,
                            vx, vy, mpc_ok_pct, solve_ms,
                        )
                        last_log_t = now

                elif now - last_log_t >= NAV_LOG_INTERVAL:
                    log.info(
                        "%s %.1fs/%.0fs  waiting for pose…",
                        log_prefix, now - nav_start_t, timeout,
                    )
                    last_log_t = now

            if time.time() >= global_end_t:
                failure_reason = failure_reason or "global_trial_timeout"

            goals_reached_frac = sum(goals_reached) / len(goals)
            final_goal         = tuple(goals[-1])
            log.info(
                "%s nav done — goals %d/%d (%.0f%%)  elapsed=%.0fs  failure=%s",
                log_prefix, sum(goals_reached), len(goals),
                goals_reached_frac * 100, time.time() - nav_start_t, failure_reason,
            )

            monitor.stop()
            score, metrics = compute_score(monitor, final_goal, goals_reached_frac=goals_reached_frac)
            metrics["n_goals"]            = len(goals)
            metrics["goals_reached_frac"] = goals_reached_frac
            if failure_reason:
                metrics["failure_reason"] = failure_reason

            if self._executor:
                try:
                    self._executor.remove_node(monitor)
                except Exception:
                    pass
            monitor.destroy_node()

        except Exception as exc:
            failure_reason = str(exc)
            log.error("%s exception: %s", log_prefix, exc, exc_info=True)
            score   = 0.0
            metrics = {"error": str(exc), "score": 0.0, "failure_reason": failure_reason}
        finally:
            bag.stop()

        metrics["scenario"] = sc_name
        metrics["weight"]   = scenario["weight"]
        return metrics

    # ── Objective / trial runner ──────────────────────────────────────────────

    def _objective(self, params: dict) -> dict:
        self._trial_num += 1
        trial_num = self._trial_num
        trial_dir = RESULTS_DIR / f"trial_{trial_num:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        now = time.time()
        if self._run_start_t is None:
            self._run_start_t = now
        completed = trial_num - 1
        eta_str   = ""
        if completed > 0:
            avg_sec = (now - self._run_start_t) / completed
            eta_str = f"  ETA ~{avg_sec * (self._max_evals - completed) / 60:.0f}min"

        n_random = getattr(self, "_n_random", N_RANDOM_INIT)
        mode_tag = "RANDOM INIT" if trial_num <= n_random else "TPE-GUIDED"
        log.info("=" * 60)
        log.info("Trial %03d/%d  [%s]  %s%s", trial_num, self._max_evals, mode_tag, _ts(), eta_str)
        log.info("Params: %s", {k: f"{v:.3f}" for k, v in params.items()})
        log.info("=" * 60)

        try:
            return self._run_trial(params, trial_num, trial_dir)
        except Exception as exc:
            log.error("[trial %03d] unhandled exception — zero score: %s", trial_num, exc, exc_info=True)
            try:
                self._sim.kill()
            except Exception:
                pass
            return {"loss": 0.0, "status": STATUS_OK, "score": 0.0}

    def _run_trial(self, params: dict, trial_num: int, trial_dir: Path) -> dict:
        trial_yaml_data   = build_trial_yaml(self._base_params, params, trial_num)
        trial_params_path = trial_dir / "planner_params.yaml"
        _save_yaml(trial_yaml_data, trial_params_path)

        t0               = time.time()
        scenario_results = []
        aggregate_score  = 0.0

        for sc_idx, scenario in enumerate(SCENARIOS, start=1):
            sc_goals = scenario.get("goals", [[scenario.get("goal_x"), scenario.get("goal_y")]])
            log.info(
                "── Scenario %d/%d: %s  (%d goals)  weight=%.2f ──",
                sc_idx, len(SCENARIOS), scenario["name"], len(sc_goals), scenario["weight"],
            )
            result          = self._run_scenario(scenario, trial_params_path, trial_dir, trial_num, sc_idx)
            weighted        = result.get("score", 0.0) * scenario["weight"]
            aggregate_score += weighted
            scenario_results.append(result)
            log.info(
                "  → sc=%s  goals=%.0f%%  score=%.3f  weighted=%.4f  running=%.4f"
                "  dist=%.2f  obs=%.3f  scans=%d",
                scenario["name"],
                result.get("goals_reached_frac", 0.0) * 100,
                result.get("score", 0.0), weighted, aggregate_score,
                result.get("dist_to_goal", float("nan")),
                result.get("obs_avoidance_score", float("nan")),
                result.get("n_cloud_msgs", 0),
            )

        self._sim.kill()
        elapsed = time.time() - t0

        # ── GP surrogate ──────────────────────────────────────────────────────
        self._history.append({
            "trial":  trial_num,
            "params": {k: float(v) for k, v in params.items()},
            "score":  float(aggregate_score),
        })
        log.info("[GP] fitting on %d observations…", len(self._history))
        gp_state = fit_gp_surrogate(self._history)

        if "param_sensitivity" in gp_state:
            top = sorted(gp_state["param_sensitivity"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            log.info(
                "[GP] active  noise=%.4f  gp_mean=%.4f±%.4f  top5=%s",
                gp_state.get("noise_level", float("nan")),
                gp_state.get("gp_mean_at_best", float("nan")),
                gp_state.get("gp_std_at_best", float("nan")),
                "  ".join(f"{k}={v:.3f}" for k, v in top),
            )
        elif "skipped" in gp_state:
            log.info("[GP] skipped — %s", gp_state["skipped"])
        elif "error" in gp_state:
            log.warning("[GP] fit error — %s", gp_state["error"])

        gp_state["trial"]     = trial_num
        gp_state["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Save per-trial checkpoint (full reloadable data + joblib model)
        save_gp_checkpoint(gp_state, trial_dir)

        # Strip private keys before putting into JSON history
        gp_state_json = _gp_state_for_json(gp_state)
        self._gp_history.append(gp_state_json)
        _save_json(gp_state_json, trial_dir / "gp_surrogate.json")

        # Keep live state for _persist() plot functions
        self._last_gp_state = gp_state

        # ── TPE snapshot ──────────────────────────────────────────────────────
        _save_json(serialize_tpe_state(self._hp_trials, trial_num), trial_dir / "tpe_state.json")
        try:
            import joblib as _jl
            _jl.dump(self._hp_trials, trial_dir / "hp_trials.joblib")
        except Exception as _exc:
            log.warning("[resume] hp_trials save failed (non-fatal): %s", _exc)

        # ── Metadata ──────────────────────────────────────────────────────────
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

        # ── Track best ────────────────────────────────────────────────────────
        prev_best   = self._best_score
        is_new_best = aggregate_score > self._best_score
        if is_new_best:
            self._best_score  = aggregate_score
            self._best_params = {k: float(v) for k, v in params.items()}
            self._best_trial  = trial_num
            shutil.copy(trial_params_path, RESULTS_DIR / "best_planner_params.yaml")
            log.info("*** NEW BEST  score=%.4f (+%.4f)  trial=%03d ***",
                     aggregate_score, aggregate_score - prev_best, trial_num)

        self._persist(trial_num)
        _cleanup_old_bags(RESULTS_DIR, BAG_KEEP_LAST_N_TRIALS)

        gp_status = (
            f"GP=active(n={gp_state.get('n_observations')})"
            if "param_sensitivity" in gp_state
            else f"GP=waiting({gp_state.get('skipped', 'error')})"
        )
        log.info(
            "[trial %03d/%d]  aggregate=%.4f  best=%.4f(t%03d)"
            "  elapsed=%.1fmin  %s",
            trial_num, self._max_evals,
            aggregate_score, self._best_score, self._best_trial,
            elapsed / 60, gp_status,
        )

        return {"loss": -aggregate_score, "status": STATUS_OK, "score": aggregate_score}

    def _persist(self, trial_num: int) -> None:
        summary = {
            "best_score":  float(self._best_score),
            "best_trial":  self._best_trial,
            "best_params": self._best_params,
            "trials": [
                {
                    "trial":   t["trial"],
                    "params":  t["params"],
                    "score":   t["score"],
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

        # New GP plots — require live _reloadable data from last GP fit
        if self._last_gp_state is not None:
            _plot_gp_marginals(
                self._last_gp_state,
                RESULTS_DIR / "gp_marginals.png",
            )
            _plot_gp_fitted_vs_actual(
                self._last_gp_state,
                RESULTS_DIR / "gp_fitted_vs_actual.png",
            )

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self, max_evals: int = MAX_EVALS, n_random: int = N_RANDOM_INIT,
            resume: bool = False) -> None:
        self._n_random  = n_random
        self._max_evals = max_evals

        if resume:
            completed = self._load_checkpoint()
            if completed >= max_evals:
                log.info("[resume] already completed %d/%d trials — nothing to do",
                         completed, max_evals)
                return
            log.info("[resume] continuing from trial %d  (%d remaining)",
                     completed, max_evals - completed)

        self._ros_init()

        try:
            import sklearn
            sklearn_ok  = True
            sklearn_ver = sklearn.__version__
        except ImportError:
            sklearn_ok  = False
            sklearn_ver = "NOT INSTALLED"

        secs_per_trial = sum(
            PLANNER_DELAY_SEC + s.get("timeout", SCENARIO_TIMEOUT) + 10
            for s in SCENARIOS
        )
        log.info("#" * 60)
        log.info("# Bayesian MPC Tuner — Go2 Gazebo (hardened)")
        log.info("# Trials=%d  RandomInit=%d  TPE=%d", max_evals, n_random, max_evals - n_random)
        log.info("# Scenarios/trial=%d  Params=%d", len(SCENARIOS), len(PARAM_NAMES))
        log.info("# Est. time: ~%.1fh", max_evals * secs_per_trial / 3600)
        log.info("# Results: %s", RESULTS_DIR)
        if sklearn_ok:
            log.info("# [OK] scikit-learn %s — GP active from trial %d", sklearn_ver, n_random + 1)
        else:
            log.warning("# [!!] scikit-learn NOT INSTALLED — pip install scikit-learn")
        log.info("# [OK] joblib required for gp_model.joblib — pip install joblib")
        log.info("#" * 60)

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
            log.info("[Interrupted] saving partial results…")
        finally:
            self._persist(self._trial_num)
            self._ros_shutdown()

        log.info("=" * 60)
        log.info("COMPLETE — best=%.4f (trial %d)", self._best_score, self._best_trial)
        if self._best_params:
            for k, v in self._best_params.items():
                log.info("  %s: %.4f", k, v)
        log.info("Deploy:  cp %s/best_planner_params.yaml %s", RESULTS_DIR, BASE_PARAMS)
        log.info("=" * 60)


# ─── Heartbeat sleep (FIX-9, FIX-10) ─────────────────────────────────────────

def _heartbeat_sleep(seconds: float, prefix: str = "", interval: float = 10.0) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        chunk    = min(interval, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining = seconds - elapsed
        if remaining > 0:
            log.debug("%s heartbeat — %.0fs remaining", prefix, remaining)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Bayesian MPC tuner for Go2 (hardened)")
    ap.add_argument("--trials", type=int, default=MAX_EVALS,      help="Total trials")
    ap.add_argument("--random", type=int, default=N_RANDOM_INIT,
                    help="Random init trials (must be > 0 and < --trials)")
    ap.add_argument("--gui",    action="store_true",               help="Launch Gazebo/RViz with GUI")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from the last completed trial in TUNING_RESULTS_DIR")
    args = ap.parse_args()

    BayesianMPCTuner(gui=args.gui).run(
        max_evals=args.trials, n_random=args.random, resume=args.resume,
    )