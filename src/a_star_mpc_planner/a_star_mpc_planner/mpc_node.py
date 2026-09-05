"""
MPC tracker ROS2 node for the Unitree G1 humanoid.

Architecture
------------
  Subscribes:
    <pose_topic>              PoseStamped  — robot pose (position + orientation)
    /lidar/points_filtered PointCloud2  — 3-D lidar hits (world frame)
    /a_star/path           nav_msgs/Path — local A* path

  Publishes:
    /mpc/predicted_path  nav_msgs/Path               — N-step MPC predicted trajectory
    /mpc/next_setpoint   geometry_msgs/PoseStamped   — lookahead setpoint
    /mpc/diagnostics     std_msgs/Float64MultiArray  — [success, cost, solve_ms, avg_ms,
                                                        fails, security, vx_eff]

  Control flow (at mpc_rate_hz):
    1. Build 6-D state [px, py, yaw, vx, vy, wz] — velocity from low-pass pose diff (#3/#4).
    2. Check LiDAR scan age; mark stale if too old (#6).
    3. Predict obstacle positions at horizon-midpoint using frame-to-frame tracking (#10).
    4. Solve MPCTracker → predicted state trajectory.
    5. Adaptive velocity limits: reduce on high failure rate, recover when healthy (#9).
    6. Walk predicted trajectory to find lookahead setpoint; ramp down near goal (#8).
    7. Publish setpoint, predicted path, diagnostics.

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker

# CubicSpline for path smoothing (fix #5); graceful fallback if unavailable
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _yaw_to_quat(yaw: float) -> tuple:
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))  # (qx, qy, qz, qw)


class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('mpc_N',            30)
        self.declare_parameter('mpc_dt',           0.1)
        self.declare_parameter('mpc_tau_v',        0.12)   # actuator lag — fix #3/#4
        self.declare_parameter('mpc_tau_w',        0.10)
        self.declare_parameter('mpc_vx_max',       1.0)
        self.declare_parameter('mpc_vx_min',       0.0)   # <=0; 0 = niente retromarcia
        self.declare_parameter('mpc_vy_max',       0.5)
        self.declare_parameter('mpc_omega_max',    1.5)
        self.declare_parameter('mpc_v_ref',        0.5)
        self.declare_parameter('mpc_Q_x',          20.0)
        self.declare_parameter('mpc_Q_y',          20.0)
        self.declare_parameter('mpc_Q_yaw',         0.5)
        self.declare_parameter('mpc_Q_terminal',   50.0)
        self.declare_parameter('mpc_R_vx',          1.0)
        self.declare_parameter('mpc_R_vx_reverse', 0.0)
        self.declare_parameter('mpc_R_vy',          1.0)
        self.declare_parameter('mpc_R_omega',       0.5)
        self.declare_parameter('mpc_R_jerk',       0.2)
        self.declare_parameter('mpc_W_obs_sigmoid',       500.0)
        self.declare_parameter('mpc_obs_alpha',           8.0)
        self.declare_parameter('mpc_obs_r',               0.5)
        self.declare_parameter('mpc_max_obs_constraints', 15)
        self.declare_parameter('mpc_obs_check_radius',    2.0)
        # Integration scheme of the position channel (lecture notes §2.1.3):
        # 'euler' (eq. 2.9, order 1) or 'midpoint' (eq. 2.10, order 2).
        self.declare_parameter('mpc_integrator', 'euler')
        # Reformulations of Chapter 7. Both off by default: the deployed behaviour
        # stays unchanged.
        self.declare_parameter('mpc_path_mode', 'time')          # 'time' | 'theta'
        self.declare_parameter('mpc_theta_progress_weight', 50.0)
        self.declare_parameter('mpc_terminal_constraint', 'none')  # 'none' | 'equilibrium'
        self.declare_parameter('mpc_terminal_rho', 5.0e3)
        self.declare_parameter('mpc_max_iter',   100)
        self.declare_parameter('mpc_warm_start',  True)
        self.declare_parameter('mpc_rate_hz',       2.0)
        self.declare_parameter('mpc_lookahead_dist', 0.5)
        self.declare_parameter('max_lidar_range',  6.0)
        self.declare_parameter('mpc_path_resample_ds', 0.20)
        self.declare_parameter('mpc_path_smooth_window', 5)
        self.declare_parameter('mpc_setpoint_alpha', 0.35)
        self.declare_parameter('mpc_setpoint_max_step', 0.30)
        self.declare_parameter('mpc_setpoint_reset_dist', 1.25)

        # Velocity estimation (#3/#4)
        self.declare_parameter('vel_filter_alpha', 0.30)

        # LiDAR staleness (#6)
        self.declare_parameter('lidar_max_age_sec', 0.30)

        # Dynamic obstacle prediction (#10)
        self.declare_parameter('obs_predict_frac', 0.50)  # fraction of horizon

        # Adaptive velocity limits (#9)
        self.declare_parameter('adaptive_vel_limits', True)

        # Security protocol
        self.declare_parameter('grid_reso',                  0.25)
        self.declare_parameter('grid_half_width',            5.0)
        self.declare_parameter('grid_std',                   0.2)
        self.declare_parameter('mpc_security_threshold',     0.35)
        self.declare_parameter('mpc_security_escape_radius', 3.0)

        # ── Build MPCConfig and tracker ───────────────────────────────
        cfg = MPCConfig(
            N             = int(self.get_parameter('mpc_N').value),
            dt            = float(self.get_parameter('mpc_dt').value),
            integrator    = str(self.get_parameter('mpc_integrator').value),
            path_mode     = str(self.get_parameter('mpc_path_mode').value),
            theta_progress_weight = float(
                self.get_parameter('mpc_theta_progress_weight').value),
            terminal_constraint = str(
                self.get_parameter('mpc_terminal_constraint').value),
            terminal_rho  = float(self.get_parameter('mpc_terminal_rho').value),
            tau_v         = float(self.get_parameter('mpc_tau_v').value),
            tau_w         = float(self.get_parameter('mpc_tau_w').value),
            vx_max        = float(self.get_parameter('mpc_vx_max').value),
            vx_min        = float(self.get_parameter('mpc_vx_min').value),
            vy_max        = float(self.get_parameter('mpc_vy_max').value),
            omega_max     = float(self.get_parameter('mpc_omega_max').value),
            v_ref         = float(self.get_parameter('mpc_v_ref').value),
            Q_x           = float(self.get_parameter('mpc_Q_x').value),
            Q_y           = float(self.get_parameter('mpc_Q_y').value),
            Q_yaw         = float(self.get_parameter('mpc_Q_yaw').value),
            Q_terminal    = float(self.get_parameter('mpc_Q_terminal').value),
            R_vx          = float(self.get_parameter('mpc_R_vx').value),
            R_vx_reverse  = float(self.get_parameter('mpc_R_vx_reverse').value),
            R_vy          = float(self.get_parameter('mpc_R_vy').value),
            R_omega       = float(self.get_parameter('mpc_R_omega').value),
            R_jerk        = float(self.get_parameter('mpc_R_jerk').value),
            W_obs_sigmoid       = float(self.get_parameter('mpc_W_obs_sigmoid').value),
            obs_alpha           = float(self.get_parameter('mpc_obs_alpha').value),
            obs_r               = float(self.get_parameter('mpc_obs_r').value),
            max_obs_constraints = int(self.get_parameter('mpc_max_obs_constraints').value),
            obs_check_radius    = float(self.get_parameter('mpc_obs_check_radius').value),
            max_iter      = int(self.get_parameter('mpc_max_iter').value),
            warm_start    = bool(self.get_parameter('mpc_warm_start').value),
        )
        self._tracker = MPCTracker(config=cfg)
        self._cfg = cfg

        # ── Security protocol ─────────────────────────────────────────
        self._security_threshold    = float(self.get_parameter('mpc_security_threshold').value)
        self._security_escape_radius = float(self.get_parameter('mpc_security_escape_radius').value)
        self._grid = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
        )
        self._security_mode: bool = False

        self._max_lidar_range     = float(self.get_parameter('max_lidar_range').value)
        self._lookahead_dist      = float(self.get_parameter('mpc_lookahead_dist').value)
        self._path_resample_ds    = float(self.get_parameter('mpc_path_resample_ds').value)
        self._path_smooth_window  = int(self.get_parameter('mpc_path_smooth_window').value)
        self._setpoint_alpha      = float(self.get_parameter('mpc_setpoint_alpha').value)
        self._setpoint_max_step   = float(self.get_parameter('mpc_setpoint_max_step').value)
        self._setpoint_reset_dist = float(self.get_parameter('mpc_setpoint_reset_dist').value)

        # ── Velocity estimation (#3/#4) ───────────────────────────────
        self._vel_filter_alpha = float(self.get_parameter('vel_filter_alpha').value)
        self._prev_pose_sec: Optional[float] = None
        self._prev_pose_xy:  Optional[np.ndarray] = None
        self._prev_pose_yaw: float = 0.0
        self._vx_est = 0.0
        self._vy_est = 0.0
        self._wz_est = 0.0

        # ── LiDAR staleness (#6) ──────────────────────────────────────
        self._lidar_max_age_sec = float(self.get_parameter('lidar_max_age_sec').value)
        self._last_scan_stamp: Optional[rclpy.time.Time] = None

        # ── Dynamic obstacle prediction (#10) ─────────────────────────
        self._obs_predict_frac = float(self.get_parameter('obs_predict_frac').value)
        self._prev_obs_pts:  Optional[np.ndarray] = None   # (M, 2) selected last cycle
        self._prev_obs_time: Optional[float]      = None   # perf_counter seconds

        # ── Adaptive velocity limits (#9) ─────────────────────────────
        self._adaptive_enabled  = bool(self.get_parameter('adaptive_vel_limits').value)
        self._cfg_vx_max        = cfg.vx_max    # configured ceiling
        self._cfg_vy_max        = cfg.vy_max
        self._cfg_omega_max     = cfg.omega_max
        self._adaptive_vx_max   = cfg.vx_max    # current effective limit
        self._adaptive_vy_max   = cfg.vy_max
        self._adaptive_omega_max = cfg.omega_max
        self._recent_solves: deque = deque(maxlen=20)

        # ── Subscriptions state ───────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._a_star_path: list | None = None
        self._a_star_path_raw_len = 0
        self._lidar_points: np.ndarray | None = None
        self._setpoint_filtered_xy:  np.ndarray | None = None
        self._setpoint_filtered_yaw: float | None = None

        # Logging counters
        self._solve_count    = 0
        self._fail_count     = 0
        self._total_solve_ms = 0.0

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        # Name of the pose topic: parametric, it is the only point where the robot
        # enters this node. The G1 publishes on /robot_pose.
        self.declare_parameter('pose_topic', '/robot_pose')
        _pose_topic = self.get_parameter('pose_topic').value
        self._pose_topic = _pose_topic

        self.create_subscription(PoseStamped, _pose_topic,               self._pose_cb,  10)
        self.create_subscription(PointCloud2, '/lidar/points_filtered',  self._lidar_cb, sensor_qos)
        self.create_subscription(Path,        '/a_star/path',            self._path_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub   = self.create_publisher(Path,                '/mpc/predicted_path', 10)
        self._setpoint_pub    = self.create_publisher(PoseStamped,         '/mpc/next_setpoint',  10)
        self._diagnostics_pub = self.create_publisher(Float64MultiArray,   '/mpc/diagnostics',    10)

        # ── Solve timer ───────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._solve_cb)

        self.get_logger().info('MPC node ready')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        """Update pose and estimate body-frame velocity via low-pass pose differentiation."""
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        yaw_new = _quat_to_yaw(qx, qy, qz, qw)
        x_new   = msg.pose.position.x
        y_new   = msg.pose.position.y

        # ── Velocity estimation (#3/#4) ───────────────────────────────
        if (self._prev_pose_sec is not None and
                self._prev_pose_xy is not None):
            dt_pose = now_sec - self._prev_pose_sec
            if 0.01 < dt_pose < 0.5:  # ignore stale or too-fast updates
                dx_w = (x_new - self._prev_pose_xy[0]) / dt_pose
                dy_w = (y_new - self._prev_pose_xy[1]) / dt_pose

                # World → body-frame rotation
                cy = math.cos(yaw_new)
                sy = math.sin(yaw_new)
                vx_raw =  dx_w * cy + dy_w * sy
                vy_raw = -dx_w * sy + dy_w * cy

                # Wrap-aware yaw rate
                dyaw_raw = math.atan2(
                    math.sin(yaw_new - self._prev_pose_yaw),
                    math.cos(yaw_new - self._prev_pose_yaw),
                ) / dt_pose

                # Exponential moving average low-pass filter
                a = self._vel_filter_alpha
                self._vx_est = (1.0 - a) * self._vx_est + a * vx_raw
                self._vy_est = (1.0 - a) * self._vy_est + a * vy_raw
                self._wz_est = (1.0 - a) * self._wz_est + a * dyaw_raw

        self._prev_pose_sec = now_sec
        self._prev_pose_xy  = np.array([x_new, y_new], dtype=float)
        self._prev_pose_yaw = yaw_new

        self._pose = msg
        self._yaw  = yaw_new

        self.get_logger().info(
            f'[MPC-DEBUG] {self._pose_topic} received: '
            f'pos=({x_new:.4f}, {y_new:.4f})  '
            f'yaw={math.degrees(yaw_new):.1f} deg  '
            f'vel_body=({self._vx_est:.2f}, {self._vy_est:.2f}, {self._wz_est:.2f})',
            throttle_duration_sec=2.0,
        )

    def _lidar_cb(self, msg: PointCloud2):
        """Parse LiDAR points, filter by range, and record scan timestamp (#6)."""
        # Store stamp for staleness check
        self._last_scan_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

        try:
            points = list(point_cloud2.read_points(msg, skip_nans=True))
            if points:
                arr = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)
                if self._pose is not None:
                    px = self._pose.pose.position.x
                    py = self._pose.pose.position.y
                    dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                    arr = arr[dists < self._max_lidar_range]
                self._lidar_points = arr if len(arr) > 0 else None
            else:
                self._lidar_points = None
        except Exception as e:
            self.get_logger().warn(f'Lidar error: {e}')

    def _path_cb(self, msg: Path):
        """Store and smooth the latest A* path."""
        if msg.poses:
            raw_path = [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            self._a_star_path_raw_len = len(raw_path)
            smoothed_path = self._smooth_resample_path(raw_path)
            self._a_star_path = smoothed_path

            did_reset = False
            if self._setpoint_filtered_xy is None or self._setpoint_filtered_yaw is None:
                did_reset = True
            else:
                idx = 1 if len(smoothed_path) > 1 else 0
                anchor = np.array([float(smoothed_path[idx][0]),
                                   float(smoothed_path[idx][1])], dtype=float)
                if float(np.linalg.norm(anchor - self._setpoint_filtered_xy)) > self._setpoint_reset_dist:
                    did_reset = True

            if did_reset:
                self._setpoint_filtered_xy  = None
                self._setpoint_filtered_yaw = None

            self.get_logger().info(
                f'[MPC] Received NEW A* path: {len(raw_path)} raw → '
                f'{len(self._a_star_path)} resampled | reset_filter={did_reset}',
                throttle_duration_sec=1.0,
            )
        else:
            self._a_star_path = None
            self._a_star_path_raw_len = 0
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

    # ── Path smoothing (fix #5 — CubicSpline replaces moving average) ──

    def _smooth_resample_path(self, path_xyz: list) -> list:
        """
        Resample A* path and smooth with CubicSpline (fix #5).

        Falls back to linear interpolation when scipy is unavailable or the
        path is too short for a cubic fit (< 4 waypoints).  Endpoints are
        always preserved exactly.
        """
        if not path_xyz or len(path_xyz) < 2:
            return path_xyz

        xy = np.array([(p[0], p[1]) for p in path_xyz], dtype=float)

        # Remove repeated consecutive points
        dxy  = np.diff(xy, axis=0)
        keep = np.hstack(([True], np.linalg.norm(dxy, axis=1) > 1e-4))
        xy   = xy[keep]
        if len(xy) < 2:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z)]

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(arc[-1])
        if total <= 1e-6:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z),
                    (float(xy[-1, 0]), float(xy[-1, 1]), z)]

        ds = max(self._path_resample_ds, 1e-2)
        s  = np.arange(0.0, total + 1e-9, ds)
        if s[-1] < total:
            s = np.append(s, total)

        # ── CubicSpline smoothing (fix #5) ────────────────────────────
        if _SCIPY_OK and len(xy) >= 4:
            # Ensure strictly increasing arc (numerical safety)
            arc_safe = arc.copy()
            for i in range(1, len(arc_safe)):
                if arc_safe[i] <= arc_safe[i - 1]:
                    arc_safe[i] = arc_safe[i - 1] + 1e-6

            cs_x = _CubicSpline(arc_safe, xy[:, 0])
            cs_y = _CubicSpline(arc_safe, xy[:, 1])
            x_s  = cs_x(s)
            y_s  = cs_y(s)
            # Pin endpoints to the original path exactly
            x_s[0]  = xy[0, 0];  y_s[0]  = xy[0, 1]
            x_s[-1] = xy[-1, 0]; y_s[-1] = xy[-1, 1]
        else:
            # Fallback: linear interpolation (also used when scipy is absent)
            x_s = np.interp(s, arc, xy[:, 0])
            y_s = np.interp(s, arc, xy[:, 1])

        z = float(path_xyz[-1][2])
        return [(float(x_s[i]), float(y_s[i]), z) for i in range(len(s))]

    # ── Dynamic obstacle prediction (#10) ─────────────────────────────

    def _predict_obs_positions(
        self,
        obs_2d:       np.ndarray,
        predict_sec:  float,
        current_time: float,
    ) -> np.ndarray:
        """
        Predict where each obstacle will be at predict_sec in the future.

        Uses nearest-neighbour matching between consecutive selected-obstacle
        sets to estimate per-point velocity.  Matches with implausible speed
        (> 3 m/s) or large positional jump are ignored.

        Modifies self._prev_obs_pts / self._prev_obs_time as a side-effect.
        """
        predicted = obs_2d.copy()

        if (self._prev_obs_pts is not None and
                self._prev_obs_time is not None and
                len(self._prev_obs_pts) > 0):
            frame_dt = current_time - self._prev_obs_time
            if 0.05 < frame_dt < 1.0:
                for i, pt in enumerate(obs_2d):
                    dists = np.linalg.norm(self._prev_obs_pts - pt, axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] < 0.5:                         # plausible correspondence
                        vel   = (pt - self._prev_obs_pts[j]) / frame_dt
                        speed = float(np.linalg.norm(vel))
                        if speed < 3.0:                        # cap at brisk walking speed
                            predicted[i] = pt + vel * predict_sec

        self._prev_obs_pts  = obs_2d.copy()
        self._prev_obs_time = current_time
        return predicted

    # ── Security escape helper ─────────────────────────────────────────

    def _find_escape_target(
        self,
        grid:     FixedGaussianGridMap,
        robot_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """BFS to find the nearest free cell outside the inflated obstacle zone."""
        ix0, iy0 = grid.world_to_index(float(robot_xy[0]), float(robot_xy[1]))
        if ix0 is None:
            return None

        visited: set = set()
        queue:   deque = deque()
        queue.append((ix0, iy0))
        visited.add((ix0, iy0))

        while queue:
            ix, iy = queue.popleft()
            if float(grid.gmap[ix, iy]) < self._security_threshold:
                wx, wy = grid.index_to_world(ix, iy)
                return np.array([wx, wy], dtype=float)
            for dix in (-1, 0, 1):
                for diy in (-1, 0, 1):
                    if dix == 0 and diy == 0:
                        continue
                    nix, niy = ix + dix, iy + diy
                    if (nix, niy) in visited:
                        continue
                    if not (0 <= nix < grid.cells and 0 <= niy < grid.cells):
                        continue
                    wx, wy = grid.index_to_world(nix, niy)
                    if np.hypot(wx - robot_xy[0], wy - robot_xy[1]) > self._security_escape_radius:
                        continue
                    visited.add((nix, niy))
                    queue.append((nix, niy))

        return None

    # ── Main solve callback ────────────────────────────────────────────

    def _solve_cb(self):
        if self._pose is None or self._a_star_path is None:
            self.get_logger().warn('[MPC] Waiting for pose and path…', throttle_duration_sec=5.0)
            return

        # ── 6-D state (#3/#4) ─────────────────────────────────────────
        state = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
            self._yaw,
            self._vx_est,
            self._vy_est,
            self._wz_est,
        ])

        # ── LiDAR staleness check (#6) ────────────────────────────────
        obs_2d: Optional[np.ndarray] = None
        scan_age_sec = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            if self._last_scan_stamp is not None:
                now_ros   = self.get_clock().now()
                scan_age_sec = (
                    now_ros - self._last_scan_stamp
                ).nanoseconds * 1e-9

            if scan_age_sec <= self._lidar_max_age_sec:
                obs_2d = self._lidar_points[:, :2]
            else:
                self.get_logger().warn(
                    f'[MPC] LiDAR scan stale ({scan_age_sec*1e3:.0f} ms > '
                    f'{self._lidar_max_age_sec*1e3:.0f} ms) — skipping obstacles',
                    throttle_duration_sec=1.0,
                )

        # ── Dynamic obstacle prediction (#10) ─────────────────────────
        if obs_2d is not None and len(obs_2d) > 0:
            predict_sec = self._obs_predict_frac * self._cfg.N * self._cfg.dt
            import time as _time_mod
            obs_2d = self._predict_obs_positions(
                obs_2d, predict_sec, _time_mod.perf_counter()
            )

        # ── Obstacle proximity log (world frame + robot-relative) ─────
        robot_xy  = state[:2]
        robot_yaw = state[2]
        cy, sy    = math.cos(robot_yaw), math.sin(robot_yaw)

        if obs_2d is not None and len(obs_2d) > 0:
            dists   = np.linalg.norm(obs_2d - robot_xy, axis=1)
            n_near  = min(5, len(obs_2d))
            near_idx = np.argsort(dists)[:n_near]
            parts   = []
            for idx in near_idx:
                wx, wy = float(obs_2d[idx, 0]), float(obs_2d[idx, 1])
                dx_w   = wx - float(robot_xy[0])
                dy_w   = wy - float(robot_xy[1])
                # World-delta → body frame
                dx_b   =  dx_w * cy + dy_w * sy
                dy_b   = -dx_w * sy + dy_w * cy
                d      = float(dists[idx])
                bearing_deg = math.degrees(math.atan2(dy_b, dx_b))
                parts.append(
                    f'world=({wx:.2f},{wy:.2f}) '
                    f'body=({dx_b:+.2f},{dy_b:+.2f}) '
                    f'd={d:.2f}m bear={bearing_deg:+.0f}°'
                )
            self.get_logger().warn(
                f'[MPC-OBS] {len(obs_2d)} pts in range | '
                f'nearest {n_near}: ' + ' | '.join(parts),
                throttle_duration_sec=0.5,
            )
        else:
            self.get_logger().warn(
                '[MPC-OBS] NO obstacles fed to MPC this cycle '
                f'(stale={scan_age_sec*1e3:.0f}ms, lidar_pts='
                f'{len(self._lidar_points) if self._lidar_points is not None else 0})',
                throttle_duration_sec=0.5,
            )

        # ── Security protocol ─────────────────────────────────────────
        in_inflated    = False
        escape_target: Optional[np.ndarray] = None
        occ_at_robot   = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            self._grid.update(self._lidar_points, state[:2])
            occ_at_robot = self._grid.get_probability(float(state[0]), float(state[1]))
            if occ_at_robot >= self._security_threshold:
                in_inflated    = True
                escape_target  = self._find_escape_target(self._grid, state[:2])

        prev_security    = self._security_mode
        self._security_mode = in_inflated
        if in_inflated and not prev_security:
            self._tracker._prev_u = None
            self._tracker._prev_x = None
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

        mpc_path = self._a_star_path
        if in_inflated:
            z = float(self._a_star_path[-1][2]) if self._a_star_path else 0.0
            if escape_target is not None:
                mpc_path = [
                    (float(state[0]), float(state[1]), z),
                    (float(escape_target[0]), float(escape_target[1]), z),
                ]
                self.get_logger().warn(
                    f'[MPC-SECURITY] occ={occ_at_robot:.3f} — escape → '
                    f'({escape_target[0]:.2f}, {escape_target[1]:.2f})',
                    throttle_duration_sec=0.5,
                )
            else:
                self.get_logger().warn(
                    f'[MPC-SECURITY] occ={occ_at_robot:.3f} — no free cell, holding A* path',
                    throttle_duration_sec=0.5,
                )

        # ── Solve MPC ─────────────────────────────────────────────────
        result = self._tracker.solve(state, mpc_path, obstacle_points_2d=obs_2d)
        result.security_mode = in_inflated

        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms
        if not result.success:
            self._fail_count += 1

        # ── Adaptive velocity limits (#9) ─────────────────────────────
        if self._adaptive_enabled:
            self._recent_solves.append(result.success)
            if len(self._recent_solves) >= 10:
                fail_rate = self._recent_solves.count(False) / len(self._recent_solves)
                if fail_rate > 0.30:
                    # Too many failures — reduce velocity ceiling by 10 %
                    new_vx = max(0.15, self._adaptive_vx_max * 0.90)
                    if new_vx < self._adaptive_vx_max:
                        self._adaptive_vx_max = new_vx
                        self._tracker.update_velocity_limits(vx_max=new_vx)
                        self.get_logger().warn(
                            f'[MPC-ADAPTIVE] High failure rate ({fail_rate:.0%}) — '
                            f'reducing vx_max to {new_vx:.2f} m/s',
                            throttle_duration_sec=2.0,
                        )
                elif fail_rate < 0.05 and self._adaptive_vx_max < self._cfg_vx_max:
                    # Healthy — recover velocity ceiling by 5 %
                    new_vx = min(self._cfg_vx_max, self._adaptive_vx_max * 1.05)
                    self._adaptive_vx_max = new_vx
                    self._tracker.update_velocity_limits(vx_max=new_vx)

        # ── Publish predicted trajectory ──────────────────────────────
        if result.x_pred is not None:
            pred_path        = Path()
            pred_path.header = self._pose.header
            for i in range(len(result.x_pred)):
                p                    = PoseStamped()
                p.header             = self._pose.header
                p.pose.position.x    = float(result.x_pred[i, 0])
                p.pose.position.y    = float(result.x_pred[i, 1])
                p.pose.position.z    = self._pose.pose.position.z
                q                    = _yaw_to_quat(float(result.x_pred[i, 2]))
                p.pose.orientation.x = q[0]
                p.pose.orientation.y = q[1]
                p.pose.orientation.z = q[2]
                p.pose.orientation.w = q[3]
                pred_path.poses.append(p)
            self._pred_path_pub.publish(pred_path)

            # ── Lookahead setpoint with near-goal ramp-down (#8) ──────
            robot_pos   = state[:2]
            path_end    = np.array(self._a_star_path[-1][:2], dtype=float)
            dist_to_end = float(np.linalg.norm(path_end - robot_pos))
            eff_lookahead = min(self._lookahead_dist, max(0.3, dist_to_end * 0.5))

            lookahead_idx = len(result.x_pred) - 1
            found = False
            for i in range(1, len(result.x_pred)):
                if float(np.linalg.norm(result.x_pred[i, :2] - robot_pos)) >= eff_lookahead:
                    lookahead_idx = i
                    found = True
                    break

            if found:
                nxt_xy  = result.x_pred[lookahead_idx, :2]
                nxt_yaw = float(result.x_pred[lookahead_idx, 2])
            else:
                # Fallback: the horizon does not reach eff_lookahead. Aim at the
                # last A* waypoint.
                #
                # The yaw canNOT be the robot's current one: the controller
                # downstream tracks the setpoint orientation, and "keep the
                # orientation you have" means omega = 0. If the fallback is
                # frequent the robot NEVER turns: it goes straight until the goal
                # ends up beside it, and there it gets stuck, because lateral
                # motion is heavily penalised (R_vy >> R_vx) and is not enough to
                # close the error on its own.
                # The correct reference is the direction TOWARDS the waypoint.
                last_wp = self._a_star_path[-1]
                nxt_xy  = np.array([float(last_wp[0]), float(last_wp[1])])
                _d      = nxt_xy - robot_pos
                nxt_yaw = (float(math.atan2(_d[1], _d[0]))
                           if float(np.linalg.norm(_d)) > 1e-6 else self._yaw)

            # Setpoint low-pass filter
            nxt_xy = np.asarray(nxt_xy, dtype=float)
            if self._setpoint_filtered_xy is None:
                self._setpoint_filtered_xy  = nxt_xy.copy()
                self._setpoint_filtered_yaw = nxt_yaw
            else:
                jump      = nxt_xy - self._setpoint_filtered_xy
                jump_norm = float(np.linalg.norm(jump))
                if self._setpoint_max_step > 0.0 and jump_norm > self._setpoint_max_step:
                    nxt_xy = self._setpoint_filtered_xy + jump / (jump_norm + 1e-9) * self._setpoint_max_step
                alpha = float(np.clip(self._setpoint_alpha, 0.0, 1.0))
                self._setpoint_filtered_xy = (1.0 - alpha) * self._setpoint_filtered_xy + alpha * nxt_xy
                yaw_err = math.atan2(
                    math.sin(nxt_yaw - self._setpoint_filtered_yaw),
                    math.cos(nxt_yaw - self._setpoint_filtered_yaw),
                )
                self._setpoint_filtered_yaw = self._setpoint_filtered_yaw + alpha * yaw_err
            nxt_xy  = self._setpoint_filtered_xy
            nxt_yaw = self._setpoint_filtered_yaw

            setpoint                    = PoseStamped()
            setpoint.header             = self._pose.header
            setpoint.pose.position.x    = float(nxt_xy[0])
            setpoint.pose.position.y    = float(nxt_xy[1])
            setpoint.pose.position.z    = self._pose.pose.position.z
            q                           = _yaw_to_quat(nxt_yaw)
            setpoint.pose.orientation.x = q[0]
            setpoint.pose.orientation.y = q[1]
            setpoint.pose.orientation.z = q[2]
            setpoint.pose.orientation.w = q[3]
            self._setpoint_pub.publish(setpoint)

            self.get_logger().info(
                f'[MPC] #{self._solve_count:04d} '
                f'ok={result.success} '
                f'cost={result.cost:8.1f} '
                f'solve={result.solve_time_ms:5.1f} ms '
                f'avg={self._total_solve_ms / self._solve_count:5.1f} ms  '
                f'fails={self._fail_count}  '
                f'security={self._security_mode}  '
                f'vx_eff={self._adaptive_vx_max:.2f}  '
                f'scan_age={scan_age_sec*1e3:.0f} ms  '
                f'path_wpts={len(self._a_star_path)}(raw={self._a_star_path_raw_len})  '
                f'robot=[{state[0]:.2f},{state[1]:.2f}] '
                f'setpt=[{nxt_xy[0]:.2f},{nxt_xy[1]:.2f}]',
                throttle_duration_sec=0.5,
            )

        # ── Diagnostics ───────────────────────────────────────────────
        diag      = Float64MultiArray()
        diag.data = [
            float(1 if result.success else 0),
            result.cost,
            result.solve_time_ms,
            float(self._total_solve_ms / max(self._solve_count, 1)),
            float(self._fail_count),
            float(1 if result.security_mode else 0),
            float(self._adaptive_vx_max),   # [6] current adaptive vx limit
            # [7..12] initial state x0 of the NLP, i.e. EXACTLY what was passed to
            # the solver. It is what makes it possible to re-solve the same problem
            # offline from a bag (metrics/): position and yaw could be derived from
            # /mpc/predicted_path[0], but the velocities are estimated in here (EMA
            # on pose differences) and without these they would never come out.

            float(state[0]), float(state[1]), float(state[2]),
            float(state[3]), float(state[4]), float(state[5]),
            # [13] IPOPT iterations of the last solve (-1 if unavailable)
            float(result.iterations if hasattr(result, 'iterations') else -1),
        ]
        self._diagnostics_pub.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
