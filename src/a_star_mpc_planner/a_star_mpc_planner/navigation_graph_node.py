"""
navigation_graph_node.py
Sparse topological navigation graph for rolling-horizon A* + MPC.

As the robot explores, nodes are added at visited free-space positions and
connected by edges that were clear of obstacles when created.  Dijkstra on
the graph produces a globally-informed waypoint sequence that the A* local
planner uses as its intermediate target, replacing the naive ray-to-boundary
approach when large obstacles block the direct path.

No vision or semantic scoring is used — traversability is determined purely
from the LiDAR occupancy grid published by the A* node.

Subscribes:
  <pose_topic>          PoseStamped        — robot position
  /global_goal       PoseStamped        — final navigation goal
  /a_star/grid_raw   Float32MultiArray  — rolling occupancy grid for edge validation

Publishes:
  /nav_graph/waypoint   PoseStamped   — next intermediate goal for A* node
  /nav_graph/markers    MarkerArray   — graph visualisation (nodes / edges / active path)

Parameters:
  nav_graph_node_spacing   : minimum distance between graph nodes [m]  (default 1.5)
  nav_graph_edge_radius    : max distance to attempt a new edge [m]    (default 4.0)
  nav_graph_waypoint_reach : distance at which a waypoint is considered reached [m] (default 1.0)
  nav_graph_obs_threshold  : occupancy probability treated as obstacle (default 0.45)

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

import heapq

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


class NavGraphNode(Node):

    def __init__(self):
        super().__init__('nav_graph_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('nav_graph_node_spacing',   1.5)
        self.declare_parameter('nav_graph_edge_radius',    4.0)
        self.declare_parameter('nav_graph_waypoint_reach', 1.0)
        self.declare_parameter('nav_graph_obs_threshold',  0.45)

        self._node_spacing = float(self.get_parameter('nav_graph_node_spacing').value)
        self._edge_radius  = float(self.get_parameter('nav_graph_edge_radius').value)
        self._wp_reach     = float(self.get_parameter('nav_graph_waypoint_reach').value)
        self._obs_thresh   = float(self.get_parameter('nav_graph_obs_threshold').value)

        # ── Graph state ───────────────────────────────────────────────
        self._nodes:  dict[int, np.ndarray] = {}   # id -> (x, y)
        self._edges:  dict[int, set]        = {}   # id -> {neighbour ids}
        self._next_id: int                  = 0

        # ── Planning state ────────────────────────────────────────────
        self._robot_pos:    np.ndarray | None = None
        self._global_goal:  np.ndarray | None = None
        self._waypoint_seq: list[int]         = []   # ordered node ids
        self._wp_idx:       int               = 0

        # ── Grid state (from /a_star/grid_raw) ───────────────────────
        # Format: [minx, miny, reso, cells, gmap.flatten(C-order)]
        # gmap[ix, iy] accessed as flat[ix * cells + iy]
        self._grid:       np.ndarray | None = None
        self._grid_minx:  float             = 0.0
        self._grid_miny:  float             = 0.0
        self._grid_reso:  float             = 0.25
        self._grid_cells: int               = 0

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

        self.create_subscription(PoseStamped,       _pose_topic,        self._pose_cb, 10)
        self.create_subscription(PoseStamped,       '/global_goal',     self._goal_cb, 10)
        self.create_subscription(Float32MultiArray, '/a_star/grid_raw', self._grid_cb, sensor_qos)

        # ── Publishers ────────────────────────────────────────────────
        self._wp_pub      = self.create_publisher(PoseStamped, '/nav_graph/waypoint', 10)
        self._markers_pub = self.create_publisher(MarkerArray, '/nav_graph/markers',  10)

        # ── 2 Hz update timer ─────────────────────────────────────────
        self.create_timer(0.5, self._update_cb)

        self.get_logger().info(
            f'NavGraphNode ready | spacing={self._node_spacing}m '
            f'edge_r={self._edge_radius}m wp_reach={self._wp_reach}m'
        )

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._robot_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        q = msg.pose.orientation

    def _goal_cb(self, msg: PoseStamped) -> None:
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        if self._global_goal is None or not np.allclose(new_goal, self._global_goal, atol=0.05):
            self._global_goal  = new_goal
            self._waypoint_seq = []
            self._wp_idx       = 0
            self.get_logger().info(
                f'[NavGraph] New goal ({new_goal[0]:.2f}, {new_goal[1]:.2f}) — graph path reset'
            )

    def _grid_cb(self, msg: Float32MultiArray) -> None:
        d = msg.data
        if len(d) < 5:
            return
        self._grid_minx  = float(d[0])
        self._grid_miny  = float(d[1])
        self._grid_reso  = float(d[2])
        self._grid_cells = int(d[3])
        n = self._grid_cells
        expected = 4 + n * n
        if len(d) < expected:
            return
        self._grid = np.array(d[4:expected], dtype=np.float32).reshape((n, n))

    # ──────────────────────────────────────────────────────────────────
    # Main update (2 Hz)
    # ──────────────────────────────────────────────────────────────────

    def _update_cb(self) -> None:
        if self._robot_pos is None or self._global_goal is None or self._grid is None:
            return
        self._try_add_node()
        self._replan_graph_path()
        self._publish_waypoint()
        self._publish_markers()

    # ──────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────

    def _try_add_node(self) -> None:
        pos = self._robot_pos

        # Skip if too close to any existing node
        if self._nodes:
            positions = np.array(list(self._nodes.values()))
            if np.linalg.norm(positions - pos, axis=1).min() < self._node_spacing:
                return

        # Skip if current position is occupied
        if not self._is_free_world(pos):
            return

        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = pos.copy()
        self._edges[nid] = set()

        # Connect to reachable neighbours with obstacle-free line of sight
        for oid, opos in self._nodes.items():
            if oid == nid:
                continue
            if np.linalg.norm(pos - opos) <= self._edge_radius and self._line_is_free(pos, opos):
                self._edges[nid].add(oid)
                self._edges[oid].add(nid)

        self.get_logger().info(
            f'[NavGraph] Node {nid} @ ({pos[0]:.2f},{pos[1]:.2f})  '
            f'total={len(self._nodes)}  neighbours={len(self._edges[nid])}',
            throttle_duration_sec=2.0,
        )

    # ──────────────────────────────────────────────────────────────────
    # Graph search (Dijkstra)
    # ──────────────────────────────────────────────────────────────────

    def _replan_graph_path(self) -> None:
        if self._global_goal is None:
            return

        positions = self._nodes  # reference, not a copy

        # If no nodes exist, clear path and let A* plan directly to goal
        if not positions:
            self._waypoint_seq = []
            self._wp_idx       = 0
            return

        # Nearest node to robot
        start_id = min(positions, key=lambda i: np.linalg.norm(positions[i] - self._robot_pos))

        # Check if goal is reachable directly from start node (or close enough)
        start_to_goal_dist = float(np.linalg.norm(self._nodes[start_id] - self._global_goal))
        
        # If goal is very close to current best node, publish that node as final waypoint
        if start_to_goal_dist < self._edge_radius and self._line_is_free(self._nodes[start_id], self._global_goal):
            self._waypoint_seq = [start_id]
            self._wp_idx       = 0
            return

        # If goal is in local grid, prefer to navigate toward it via A* instead of graph
        if self._goal_in_grid():
            # Goal is known and reachable — let A* plan directly
            self._waypoint_seq = []
            self._wp_idx       = 0
            return

        # Find nearest node to goal for multi-hop path
        goal_id  = min(positions, key=lambda i: np.linalg.norm(positions[i] - self._global_goal))

        if start_id == goal_id:
            # No other nodes — fallback to direct A* planning toward goal
            self._waypoint_seq = []
            self._wp_idx       = 0
            return

        path = self._dijkstra(start_id, goal_id)
        if path and path != self._waypoint_seq:
            self._waypoint_seq = path
            self._wp_idx       = 0

    def _dijkstra(self, start: int, goal: int) -> list[int] | None:
        dist:  dict[int, float]       = {start: 0.0}
        prev:  dict[int, int | None]  = {start: None}
        heap:  list[tuple]            = [(0.0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float('inf')):
                continue
            if u == goal:
                break
            for v in self._edges.get(u, set()):
                w  = float(np.linalg.norm(self._nodes[u] - self._nodes[v]))
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if goal not in prev:
            return None

        path, node = [], goal
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    # ──────────────────────────────────────────────────────────────────
    # Waypoint publication
    # ──────────────────────────────────────────────────────────────────

    def _publish_waypoint(self) -> None:
        """
        Publish the next waypoint in the graph path.
        If no graph path exists, suppress waypoint publication and let A* plan directly to goal.
        """
        if not self._waypoint_seq:
            # No graph waypoints — A* will navigate directly toward the global goal
            return

        # Advance past already-reached waypoints
        while (self._wp_idx < len(self._waypoint_seq) - 1 and
               np.linalg.norm(
                   self._robot_pos - self._nodes[self._waypoint_seq[self._wp_idx]]
               ) < self._wp_reach):
            self._wp_idx += 1

        wp_pos = self._nodes[self._waypoint_seq[self._wp_idx]]

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(wp_pos[0])
        msg.pose.position.y = float(wp_pos[1])
        msg.pose.orientation.w = 1.0
        self._wp_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────

    def _is_free_world(self, pos: np.ndarray) -> bool:
        if self._grid is None:
            return True
        ix = int((pos[0] - self._grid_minx) / self._grid_reso)
        iy = int((pos[1] - self._grid_miny) / self._grid_reso)
        if not (0 <= ix < self._grid_cells and 0 <= iy < self._grid_cells):
            return True  # outside known grid — assume free (unknown space)
        return float(self._grid[ix, iy]) < self._obs_thresh

    def _line_is_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Sample the occupancy grid along a→b at half-cell resolution."""
        dist = float(np.linalg.norm(b - a))
        n    = max(2, int(dist / (self._grid_reso * 0.5)) + 1)
        for i in range(n + 1):
            t = i / n
            if not self._is_free_world(a + t * (b - a)):
                return False
        return True

    def _goal_in_grid(self) -> bool:
        """True when the final goal is inside the current local grid."""
        if self._grid is None or self._global_goal is None:
            return False
        ix = int((self._global_goal[0] - self._grid_minx) / self._grid_reso)
        iy = int((self._global_goal[1] - self._grid_miny) / self._grid_reso)
        return 0 <= ix < self._grid_cells and 0 <= iy < self._grid_cells

    # ──────────────────────────────────────────────────────────────────
    # RViz visualisation
    # ──────────────────────────────────────────────────────────────────

    def _publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        # Nodes — green spheres
        nm = Marker()
        nm.header.frame_id = 'map'
        nm.header.stamp    = now
        nm.ns     = 'nav_graph_nodes'
        nm.id     = 0
        nm.type   = Marker.SPHERE_LIST
        nm.action = Marker.ADD
        nm.scale.x = nm.scale.y = nm.scale.z = 0.2
        nm.color   = ColorRGBA(r=0.2, g=0.85, b=0.2, a=0.85)
        for pos in self._nodes.values():
            nm.points.append(Point(x=float(pos[0]), y=float(pos[1]), z=0.1))
        ma.markers.append(nm)

        # Edges — blue lines (each edge drawn once)
        em = Marker()
        em.header.frame_id = 'map'
        em.header.stamp    = now
        em.ns     = 'nav_graph_edges'
        em.id     = 1
        em.type   = Marker.LINE_LIST
        em.action = Marker.ADD
        em.scale.x = 0.05
        em.color   = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.5)
        seen: set[tuple] = set()
        for nid, nbrs in self._edges.items():
            for mid in nbrs:
                key = (min(nid, mid), max(nid, mid))
                if key in seen:
                    continue
                seen.add(key)
                p1, p2 = self._nodes[nid], self._nodes[mid]
                em.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=0.1))
                em.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=0.1))
        ma.markers.append(em)

        # Active graph path — orange strip
        pm = Marker()
        pm.header.frame_id = 'map'
        pm.header.stamp    = now
        pm.ns     = 'nav_graph_path'
        pm.id     = 2
        pm.type   = Marker.LINE_STRIP
        pm.action = Marker.ADD
        pm.scale.x = 0.12
        pm.color   = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)
        for nid in self._waypoint_seq:
            pos = self._nodes[nid]
            pm.points.append(Point(x=float(pos[0]), y=float(pos[1]), z=0.2))
        ma.markers.append(pm)

        self._markers_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = NavGraphNode()
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
