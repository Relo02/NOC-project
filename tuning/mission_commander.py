#!/usr/bin/env python3
"""Publish sequenced goals on /goal_pose and wait for robot progress on /go2/pose."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.node import Node


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class MissionCommander(Node):
    def __init__(
        self,
        goal_topic: str,
        pose_topic: str,
        frame_id: str,
        goal_radius: float,
        goal_timeout_sec: float,
        publish_hz: float,
        republish_sec: float,
        status_log_sec: float,
    ) -> None:
        super().__init__('mission_commander')
        self._goal_pub = self.create_publisher(PoseStamped, goal_topic, 10)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10)

        self._frame_id = frame_id
        self._goal_radius = goal_radius
        self._goal_timeout_sec = goal_timeout_sec
        self._publish_period = 1.0 / max(0.5, publish_hz)
        self._republish_sec = republish_sec
        self._status_log_sec = status_log_sec

        self._pose: PoseStamped | None = None

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _distance_to(self, wp: Waypoint) -> float | None:
        if self._pose is None:
            return None
        px = self._pose.pose.position.x
        py = self._pose.pose.position.y
        return math.hypot(px - wp.x, py - wp.y)

    def _publish_goal(self, wp: Waypoint) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(wp.x)
        msg.pose.position.y = float(wp.y)
        msg.pose.position.z = 0.0
        msg.pose.orientation = yaw_to_quat(wp.yaw)
        self._goal_pub.publish(msg)

    def execute_waypoint(self, wp: Waypoint, idx: int) -> bool:
        start = time.time()
        last_publish = 0.0
        last_status = 0.0
        self.get_logger().info(
            f'[Mission] Goal {idx}: x={wp.x:.2f} y={wp.y:.2f} yaw={wp.yaw:.2f} rad'
        )

        while rclpy.ok() and (time.time() - start) <= self._goal_timeout_sec:
            now = time.time()
            rclpy.spin_once(self, timeout_sec=0.1)
            if last_publish == 0.0 or (
                self._republish_sec > 0.0 and (now - last_publish) >= self._republish_sec
            ):
                self._publish_goal(wp)
                last_publish = now

            dist = self._distance_to(wp)
            if dist is not None and dist <= self._goal_radius:
                self.get_logger().info(
                    f'[Mission] Goal {idx} reached (dist={dist:.3f} m <= {self._goal_radius:.3f} m)'
                )
                return True
            if (
                dist is not None
                and self._status_log_sec > 0.0
                and (now - last_status) >= self._status_log_sec
            ):
                elapsed = now - start
                self.get_logger().info(
                    f'[Mission] Goal {idx} waiting: dist={dist:.3f} m elapsed={elapsed:.1f}s'
                )
                last_status = now
            time.sleep(self._publish_period)

        dist = self._distance_to(wp)
        if dist is None:
            self.get_logger().warning(f'[Mission] Goal {idx} timeout (no pose received).')
        else:
            self.get_logger().warning(
                f'[Mission] Goal {idx} timeout (dist={dist:.3f} m > {self._goal_radius:.3f} m).'
            )
        return False


def parse_waypoints(raw: str) -> list[Waypoint]:
    points: list[Waypoint] = []
    for chunk in raw.split(';'):
        txt = chunk.strip()
        if not txt:
            continue
        items = [x.strip() for x in txt.split(',')]
        if len(items) not in (2, 3):
            raise ValueError(f'Invalid waypoint format: {txt}')
        x = float(items[0])
        y = float(items[1])
        yaw = float(items[2]) if len(items) == 3 else 0.0
        points.append(Waypoint(x=x, y=y, yaw=yaw))
    if not points:
        raise ValueError('No waypoints parsed from input.')
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--waypoints', required=True, help='Format: x,y,yaw;x,y,yaw;...')
    ap.add_argument('--goal-topic', default='/goal_pose')
    ap.add_argument('--pose-topic', default='/go2/pose')
    ap.add_argument('--frame-id', default='map')
    ap.add_argument('--goal-radius', type=float, default=0.5)
    ap.add_argument('--goal-timeout-sec', type=float, default=120.0)
    ap.add_argument('--startup-delay-sec', type=float, default=2.0)
    ap.add_argument('--post-goal-settle-sec', type=float, default=1.5)
    ap.add_argument('--publish-hz', type=float, default=4.0)
    ap.add_argument(
        '--republish-sec',
        type=float,
        default=2.0,
        help='Republish the active goal at most every N seconds. Use <=0 to publish once.',
    )
    ap.add_argument('--status-log-sec', type=float, default=10.0)
    args = ap.parse_args()

    waypoints = parse_waypoints(args.waypoints)

    rclpy.init()
    node = MissionCommander(
        goal_topic=args.goal_topic,
        pose_topic=args.pose_topic,
        frame_id=args.frame_id,
        goal_radius=args.goal_radius,
        goal_timeout_sec=args.goal_timeout_sec,
        publish_hz=args.publish_hz,
        republish_sec=args.republish_sec,
        status_log_sec=args.status_log_sec,
    )

    if args.startup_delay_sec > 0.0:
        node.get_logger().info(f'[Mission] Startup delay: {args.startup_delay_sec:.1f}s')
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < args.startup_delay_sec:
            rclpy.spin_once(node, timeout_sec=0.1)

    all_ok = True
    try:
        for idx, wp in enumerate(waypoints, start=1):
            ok = node.execute_waypoint(wp, idx)
            all_ok = all_ok and ok
            if args.post_goal_settle_sec > 0.0:
                t1 = time.time()
                while rclpy.ok() and (time.time() - t1) < args.post_goal_settle_sec:
                    rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
