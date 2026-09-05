"""
A* planner ROS2 node for the Unitree G1 humanoid.

Architecture
------------
  Subscribes:
    <pose_topic>       PoseStamped  — robot position and orientation
    /lidar/points_filtered   PointCloud2  — 2-D/3-D lidar hits (world frame)
    /global_goal    PoseStamped  — runtime global goal override

  Publishes:
    /a_star/path            nav_msgs/Path           — local A* path to local goal
    /a_star/local_goal      geometry_msgs/PoseStamped — current local goal
    /a_star/occupancy_grid  nav_msgs/OccupancyGrid  — Gaussian grid map
    /a_star/grid_raw        std_msgs/Float32MultiArray — raw grid coordinates

  Control flow:
    A timer fires at replan_rate_hz.  On each tick:
      1. Build FixedGaussianGridMap from latest LiDAR scan.
      2. Run A* from robot position toward global goal.
      3. Publish path, local_goal, and grid.

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header

from a_star_mpc_planner.a_star_planner import AStarPlanner
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from a_star_mpc_planner.persistent_map import PersistentOccupancyMap
from a_star_mpc_planner.geodesic_field import GeodesicField, block_radius


class AStarNode(Node):

    def __init__(self):
        super().__init__('a_star_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('goal_x',                5.0)
        self.declare_parameter('goal_y',                5.0)
        self.declare_parameter('goal_z',                0.0)
        self.declare_parameter('wait_for_goal',       False)
        self.declare_parameter('grid_reso',             0.25)
        self.declare_parameter('grid_half_width',       5.0)
        self.declare_parameter('grid_std',              0.4)
        self.declare_parameter('obstacle_threshold',    0.5)
        self.declare_parameter('obstacle_cost_weight', 10.0)
        self.declare_parameter('replan_rate_hz',        2.0)
        self.declare_parameter('goal_reached_radius',   0.3)
        self.declare_parameter('duplicate_goal_xy_tolerance', 0.05)
        self.declare_parameter('duplicate_goal_z_tolerance', 0.05)
        self.declare_parameter('max_lidar_range',       6.0)
        self.declare_parameter('planning_height',       0.0)
        self.declare_parameter('map_decay_sec',        30.0)
        self.declare_parameter('map_max_cells',     50_000)

        self._goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value,
        ])
        self._wait_for_goal = bool(self.get_parameter('wait_for_goal').value)
        self._goal_initialized = not self._wait_for_goal

        self._planning_height = float(self.get_parameter('planning_height').value)
        self._goal_reached_radius = float(self.get_parameter('goal_reached_radius').value)
        self._duplicate_goal_xy_tolerance = float(
            self.get_parameter('duplicate_goal_xy_tolerance').value
        )
        self._duplicate_goal_z_tolerance = float(
            self.get_parameter('duplicate_goal_z_tolerance').value
        )
        self._max_lidar_range = float(self.get_parameter('max_lidar_range').value)

        # ── local target selection ────────────────────────────────────
        # false = original behaviour (projection of the goal onto the window
        # border along the robot->goal ray). true = argmin of the geodesic
        # distance.
        self.declare_parameter('use_geodesic_target', False)
        # Margin [m] around the robot+goal+known-map box the wavefront is
        # propagated over. Generous on purpose: the way out of a concavity often
        # leaves the rectangle containing robot and goal.
        self.declare_parameter('geodesic_margin', 6.0)
        # Hysteresis: how much an alternative route has to improve to change the
        # decision. Without it, two symmetric routes (the two sides of a corridor)
        # alternate at every replan and the robot sways on the spot.
        self.declare_parameter('target_switch_margin', 0.0)
        self.declare_parameter('commit_to_side', False)

        self._grid_map = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
        )
        self._planner = AStarPlanner(
            obstacle_threshold=float(self.get_parameter('obstacle_threshold').value),
            obstacle_cost_weight=float(self.get_parameter('obstacle_cost_weight').value),
            switch_margin=float(self.get_parameter('target_switch_margin').value),
            commit_to_side=bool(self.get_parameter('commit_to_side').value),
        )
        # Local target selector based on the GEODESIC distance over the
        # accumulated map instead of the euclidean one. See geodesic_field.py:
        # with the euclidean distance a cell at the bottom of a closed pocket
        # looks 3.65 m from the goal while it is 28.79 m away along a path, and
        # the planner sends the robot back in cycle after cycle.
        self._use_geodesic = bool(self.get_parameter('use_geodesic_target').value)
        self._geo_margin = float(self.get_parameter('geodesic_margin').value)
        self._r_block = block_radius(
            float(self.get_parameter('grid_std').value),
            float(self.get_parameter('obstacle_threshold').value))
        self._geo_ms = 0.0
        self._persistent_map = PersistentOccupancyMap(
            grid_reso=float(self.get_parameter('grid_reso').value),
            decay_sec=float(self.get_parameter('map_decay_sec').value),
            max_cells=int(self.get_parameter('map_max_cells').value),
        )

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._lidar_points: np.ndarray | None = None
        self._goal_reached = False

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

        # Frame the path, the grid and the local goal are published in. It used to
        # be hard-wired to 'map', which only exists if a SLAM runs upstream: in
        # this stack there is no prior map and the planning frame is 'odom'. With
        # the wrong frame the messages are published normally but RViz cannot
        # transform them, so NOTHING is displayed and nobody reports an error.

        self.declare_parameter('planning_frame', 'odom')
        self._frame = self.get_parameter('planning_frame').value

        self.create_subscription(PoseStamped, _pose_topic,             self._pose_cb,         10)
        self.create_subscription(PoseStamped, '/global_goal',          self._goal_cb,         10)
        self.create_subscription(PointCloud2, '/lidar/points_filtered',self._lidar_cb,        sensor_qos)

        # ── Publishers ────────────────────────────────────────────────
        self._path_pub  = self.create_publisher(Path,               '/a_star/path',                10)
        self._lgpal_pub = self.create_publisher(PoseStamped,        '/a_star/local_goal',          10)
        self._grid_pub  = self.create_publisher(OccupancyGrid,      '/a_star/occupancy_grid',      10)
        self._raw_pub   = self.create_publisher(Float32MultiArray,   '/a_star/grid_raw',            10)
        self._pmap_pub  = self.create_publisher(PointCloud2,         '/a_star/persistent_obstacles', 10)

        # ── Replan timer ──────────────────────────────────────────────
        rate = float(self.get_parameter('replan_rate_hz').value)
        self.create_timer(1.0 / rate, self._replan_cb)

        if self._goal_initialized:
            self.get_logger().info(
                f'A* node ready | startup goal=({self._goal[0]:.1f}, {self._goal[1]:.1f}) '
                f'| grid={2*self._grid_map.half_width:.0f}m @ {self._grid_map.reso}m/cell'
            )
        else:
            self.get_logger().info(
                f'A* node ready | waiting for /global_goal '
                f'| grid={2*self._grid_map.half_width:.0f}m @ {self._grid_map.reso}m/cell'
            )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg

    def _goal_cb(self, msg: PoseStamped):
        new_goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        if self._goal_initialized:
            xy_delta = float(np.linalg.norm(new_goal[:2] - self._goal[:2]))
            z_delta = abs(float(new_goal[2] - self._goal[2]))
            if (
                xy_delta <= self._duplicate_goal_xy_tolerance
                and z_delta <= self._duplicate_goal_z_tolerance
            ):
                self.get_logger().debug(
                    f'[A*] Ignoring duplicate global goal: '
                    f'xy_delta={xy_delta:.3f} m z_delta={z_delta:.3f} m'
                )
                return

        self._goal = new_goal
        self._goal_reached = False
        self._goal_initialized = True

        self.get_logger().info(
            f'[A*] New global goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f}, {self._goal[2]:.2f})'
        )

    def _lidar_cb(self, msg: PointCloud2):
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

            # Accumulate confirmed obstacle positions into the persistent map
            now = self.get_clock().now().nanoseconds * 1e-9
            self._persistent_map.update(self._lidar_points, now)
        except Exception as e:
            self.get_logger().warn(f'Lidar parsing error: {e}')

    def _replan_cb(self):
        """
        Replan every timer tick (now at 10 Hz for continuous online adaptation).

        KEY: This callback runs at the configured replan_rate_hz (default 10 Hz).
        At EACH call:
          1. We use the LATEST robot pose from self._pose
          2. We use the LATEST LiDAR scan from self._lidar_points
          3. Local occupancy grid is RE-CENTERED on robot's CURRENT position
          4. A* is RE-RUN from robot's CURRENT position to goal
          5. New path is PUBLISHED for MPC to track

        This ensures that as the robot moves, the planning continuously adapts
        to new sensor data and pose, rather than following a stale plan.
        """
        # DEBUG: show what data is (or isn't) available
        if self._pose is None or self._lidar_points is None:
            self.get_logger().warn(
                f'[A*-DEBUG] Blocked: pose={self._pose is not None}  '
                f'lidar={self._lidar_points is not None}',
                throttle_duration_sec=3.0,
            )
            return

        if not self._goal_initialized:
            self.get_logger().info(
                '[A*] Waiting for first goal on /global_goal ...',
                throttle_duration_sec=5.0,
            )
            return

        drone_xyz = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
            self._pose.pose.position.z,
        ])
        drone_xy = drone_xyz[:2]

        # DEBUG: log the actual pose being used for planning
        self.get_logger().info(
            f'[A*-DEBUG] Planning from pose=({drone_xy[0]:.4f}, {drone_xy[1]:.4f})  '
            f'goal=({self._goal[0]:.2f}, {self._goal[1]:.2f})',
            throttle_duration_sec=2.0,
        )

        # Goal-reached guard — stop replanning once close enough
        dist_to_goal = float(np.linalg.norm(drone_xy - self._goal[:2]))
        if dist_to_goal <= self._goal_reached_radius:
            if not self._goal_reached:
                self.get_logger().info(
                    f'[A*] Goal reached! dist={dist_to_goal:.2f} m'
                )
                self._goal_reached = True
            return
        self._goal_reached = False

        # === ONLINE REPLANNING: Update grid centered on CURRENT robot position ===
        # Merge live LiDAR with persistent obstacle memory so walls that left
        # sensor range are still represented — this prevents local minima in
        # U-shaped corridors and around convex obstacles.
        hw = self._grid_map.half_width
        persistent_pts = self._persistent_map.get_points_in_window(
            drone_xy[0] - hw, drone_xy[1] - hw,
            drone_xy[0] + hw, drone_xy[1] + hw,
        )
        if persistent_pts is not None and self._lidar_points is not None:
            merged = np.vstack([self._lidar_points, persistent_pts])
        elif persistent_pts is not None:
            merged = persistent_pts
        else:
            merged = self._lidar_points
        self._grid_map.update(merged, drone_xy)

        stamp = self.get_clock().now().to_msg()

        # === ONLINE REPLANNING: Run A* from CURRENT robot position ===
        # Plan directly to the current global goal; do not route through
        # nav-graph waypoints, so a previous explored graph node cannot block
        # or pull the robot backward when a new goal is sent.
        geo = None
        if self._use_geodesic:
            # The field is propagated over the WHOLE accumulated map, not just the
            # A* window: it is precisely the information outside the window (the
            # back of the alley, the end of the wall) that makes the geodesic
            # different from the euclidean, and restricting it would make it useless.
            _t0 = time.perf_counter()
            known = self._persistent_map.get_points_in_window(
                -1e6, -1e6, 1e6, 1e6)
            if known is not None and len(known):
                try:
                    geo = GeodesicField(
                        np.asarray(known)[:, :2], self._goal[:2], drone_xy,
                        reso=self._grid_map.reso, r_block=self._r_block,
                        margin=self._geo_margin)
                except Exception as exc:      # must never stop navigation
                    self.get_logger().warn(
                        f'[A*] campo geodetico non calcolabile: {exc}',
                        throttle_duration_sec=5.0)
                    geo = None
            self._geo_ms = (time.perf_counter() - _t0) * 1e3

        path = self._planner.plan(self._grid_map, drone_xy, self._goal[:2],
                                  None, geo)

        if path:
            # Publish path
            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self._frame
            for wx, wy in path:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self._frame
                pose.pose.position.x = float(wx)
                pose.pose.position.y = float(wy)
                pose.pose.position.z = self._goal[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self._path_pub.publish(path_msg)

            # Publish local goal (last waypoint)
            local_goal_msg = PoseStamped()
            local_goal_msg.header.stamp = stamp
            local_goal_msg.header.frame_id = self._frame
            local_goal_msg.pose.position.x = float(path[-1][0])
            local_goal_msg.pose.position.y = float(path[-1][1])
            local_goal_msg.pose.position.z = self._goal[2]
            local_goal_msg.pose.orientation.w = 1.0
            self._lgpal_pub.publish(local_goal_msg)

            self.get_logger().info(
                f'[A*] NAVIGATING  dist_to_goal={dist_to_goal:.2f} m  '
                f'robot=({drone_xy[0]:.2f},{drone_xy[1]:.2f})  '
                f'goal=({self._goal[0]:.2f},{self._goal[1]:.2f})  '
                f'path={len(path)} wpts  local_goal=({path[-1][0]:.2f},{path[-1][1]:.2f})'
                + (f'  geo={self._geo_ms:.0f}ms' if self._use_geodesic else ''),
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn('[A*] No path found', throttle_duration_sec=2.0)

        # Publish occupancy grid (correct ROS row-major: x=column, y=row -> transpose)
        if self._grid_map.gmap is not None:
            ogm = OccupancyGrid()
            ogm.header.stamp = stamp
            ogm.header.frame_id = self._frame
            ogm.info.resolution = self._grid_map.reso
            ogm.info.width = self._grid_map.cells
            ogm.info.height = self._grid_map.cells
            ogm.info.origin.position.x = self._grid_map.minx
            ogm.info.origin.position.y = self._grid_map.miny
            ogm.info.origin.orientation.w = 1.0
            scaled = (self._grid_map.gmap.T.flatten() * 100.0).clip(0, 100).astype(np.int8)
            ogm.data = scaled.tolist()
            self._grid_pub.publish(ogm)

            # Publish raw grid for MPC node
            raw_msg = Float32MultiArray()
            gm = self._grid_map
            meta = [float(gm.minx), float(gm.miny), float(gm.reso), float(gm.cells)]
            raw_msg.data = meta + gm.gmap.flatten(order='C').astype(np.float32).tolist()
            self._raw_pub.publish(raw_msg)

        # Publish all persistent obstacle cells as a PointCloud2 for RViz
        all_cells = self._persistent_map.get_points_in_window(
            -1e9, -1e9, 1e9, 1e9  # unbounded — publish everything stored
        )
        if all_cells is not None:
            hdr = Header(stamp=stamp, frame_id=self._frame)
            pc = point_cloud2.create_cloud_xyz32(hdr, all_cells[:, :3].tolist())
            self._pmap_pub.publish(pc)


def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
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
