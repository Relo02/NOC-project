"""
mission_runner_node.py

Plays a YAML waypoint mission against the A*+MPC planner. Publishes one goal
at a time on /global_goal, waits until the robot is within reach_radius of
the current waypoint (computed from /odom), then advances. Logs success and
per-leg timing for offline benchmarking.

Mission YAML format (see config/example_mission.yaml):

  frame_id: odom
  reach_radius: 0.30
  leg_timeout_sec: 120.0
  waypoints:
    - { x:  2.0, y:  0.0, yaw: 0.0 }
    - { x:  4.0, y:  1.5, yaw: 1.57 }
    - { x:  0.0, y:  0.0, yaw: 0.0, name: "home" }
"""

import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def _yaw_to_quat(yaw):
    """Z-axis quaternion (x, y, z, w) from a yaw angle in radians."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class MissionRunnerNode(Node):

    def __init__(self):
        super().__init__("robot_real_mission_runner")

        self.declare_parameter("mission_file",     "")
        self.declare_parameter("global_goal_topic","/global_goal")
        self.declare_parameter("odom_topic",       "/odom")
        self.declare_parameter("start_delay_sec",  3.0)
        self.declare_parameter("repeat",           False)

        self.mission_file   = self.get_parameter("mission_file").value
        self.goal_topic     = self.get_parameter("global_goal_topic").value
        self.odom_topic     = self.get_parameter("odom_topic").value
        self.start_delay    = float(self.get_parameter("start_delay_sec").value)
        self.repeat         = bool(self.get_parameter("repeat").value)

        if not self.mission_file:
            self.get_logger().info(
                "mission_file is empty — mission runner idle. "
                "Use goal_relay_node + RViz instead."
            )
            return

        if not os.path.isfile(self.mission_file):
            self.get_logger().error(f"mission_file not found: {self.mission_file}")
            return

        with open(self.mission_file, "r") as f:
            mission = yaml.safe_load(f) or {}

        self.frame_id        = mission.get("frame_id", "odom")
        self.reach_radius    = float(mission.get("reach_radius", 0.30))
        self.leg_timeout_sec = float(mission.get("leg_timeout_sec", 120.0))
        self.waypoints       = mission.get("waypoints") or []

        if not self.waypoints:
            self.get_logger().error(f"No waypoints in {self.mission_file}")
            return

        self.idx = 0
        self.cur_pose = None        # (x, y) in odom frame
        self.leg_start_time = None

        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, 10
        )

        self.get_logger().info(
            f"Loaded mission with {len(self.waypoints)} waypoints from {self.mission_file}; "
            f"frame={self.frame_id}, reach_radius={self.reach_radius:.2f} m, "
            f"leg_timeout={self.leg_timeout_sec:.0f} s, repeat={self.repeat}"
        )
        self.get_logger().info(f"Mission start in {self.start_delay:.1f} s...")

        self._tick_timer = self.create_timer(0.5, self._tick)
        self._t0 = time.time()
        self._published_idx = -1

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.cur_pose = (p.x, p.y)

    def _tick(self):
        if not hasattr(self, "waypoints") or not self.waypoints:
            return

        # Initial delay (let TF / hw_bridge warm up)
        if (time.time() - self._t0) < self.start_delay:
            return

        # Publish current waypoint once (latched-style)
        if self._published_idx != self.idx:
            wp = self.waypoints[self.idx]
            self._publish_goal(wp)
            self._published_idx = self.idx
            self.leg_start_time = time.time()
            name = wp.get("name", f"#{self.idx + 1}")
            self.get_logger().info(
                f"[{self.idx + 1}/{len(self.waypoints)}] heading to "
                f"{name} (x={wp['x']:+.2f}, y={wp['y']:+.2f})"
            )

        # Check progress
        wp = self.waypoints[self.idx]
        if self.cur_pose is not None:
            dx, dy = wp["x"] - self.cur_pose[0], wp["y"] - self.cur_pose[1]
            dist = math.hypot(dx, dy)

            if dist <= self.reach_radius:
                leg_dt = time.time() - self.leg_start_time
                self.get_logger().info(
                    f"[{self.idx + 1}/{len(self.waypoints)}] reached "
                    f"(dist={dist:.2f} m, time={leg_dt:.1f} s)"
                )
                self._advance()
                return

        # Leg timeout
        if self.leg_start_time is not None and \
                (time.time() - self.leg_start_time) > self.leg_timeout_sec:
            self.get_logger().warn(
                f"[{self.idx + 1}/{len(self.waypoints)}] TIMED OUT after "
                f"{self.leg_timeout_sec:.0f} s — advancing anyway"
            )
            self._advance()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _advance(self):
        self.idx += 1
        if self.idx >= len(self.waypoints):
            if self.repeat:
                self.idx = 0
                self.get_logger().info("Mission complete — looping (repeat=true)")
            else:
                self.get_logger().info("Mission complete.")
                self._tick_timer.cancel()
                return

    def _publish_goal(self, wp):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(wp["x"])
        msg.pose.position.y = float(wp["y"])
        msg.pose.position.z = float(wp.get("z", 0.0))
        qx, qy, qz, qw = _yaw_to_quat(float(wp.get("yaw", 0.0)))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionRunnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == "__main__":
    main()
