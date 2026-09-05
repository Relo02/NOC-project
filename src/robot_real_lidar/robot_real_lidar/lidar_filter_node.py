"""
lidar_filter_node.py

Adapter between the Unitree G1's Livox Mid-360 LiDAR and the a_star_mpc_planner.

Pipeline
--------
  /livox/lidar    (sensor_msgs/PointCloud2 in `mid360_link` frame)
        │
        ├─ range filter        (drop points outside [r_min, r_max])
        ├─ height filter       (drop ground / ceiling)
        ├─ voxel downsample    (deterministic grid hashing, no PCL dependency)
        ├─ tf2 transform       (mid360_link → odom)
        ▼
  /lidar/points_filtered (sensor_msgs/PointCloud2 in `odom` frame)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException


def _quat_to_rotmat(qx, qy, qz, qw):
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


class LidarFilterNode(Node):

    def __init__(self):
        super().__init__("robot_real_lidar_filter")

        # ── Topics & frames ───────────────────────────────────────────────
        self.declare_parameter("raw_topic",       "/livox/lidar")
        self.declare_parameter("filtered_topic",  "/lidar/points_filtered")
        self.declare_parameter("source_frame",    "mid360_link")
        self.declare_parameter("target_frame",    "odom")

        # ── Filtering parameters ──────────────────────────────────────────
        self.declare_parameter("r_min",           0.20)   # [m]
        self.declare_parameter("r_max",           6.00)   # [m]
        self.declare_parameter("z_min",          -0.20)   # [m] in TARGET frame
        self.declare_parameter("z_max",           1.50)   # [m]
        self.declare_parameter("voxel_size",      0.05)   # [m]; <=0 disables
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("tf_lookup_timeout", 0.05)  # [s]

        self.raw_topic        = self.get_parameter("raw_topic").value
        self.filtered_topic   = self.get_parameter("filtered_topic").value
        self.source_frame     = self.get_parameter("source_frame").value
        self.target_frame     = self.get_parameter("target_frame").value
        self.r_min            = float(self.get_parameter("r_min").value)
        self.r_max            = float(self.get_parameter("r_max").value)
        self.z_min            = float(self.get_parameter("z_min").value)
        self.z_max            = float(self.get_parameter("z_max").value)
        self.voxel_size       = float(self.get_parameter("voxel_size").value)
        self.publish_period   = 1.0 / float(self.get_parameter("publish_rate_hz").value)
        self.tf_timeout       = float(self.get_parameter("tf_lookup_timeout").value)

        # ── TF buffer ─────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS interface ─────────────────────────────────────────────────
        self.sub = self.create_subscription(
            PointCloud2, self.raw_topic, self._on_cloud, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(
            PointCloud2, self.filtered_topic, qos_profile_sensor_data
        )

        # Rate-limit publication so we don't republish at the raw sensor rate
        # if the planner doesn't need it.
        self._last_pub_time = self.get_clock().now()

        self.get_logger().info(
            f"robot_real_lidar_filter: {self.raw_topic} ({self.source_frame}) "
            f"→ {self.filtered_topic} ({self.target_frame}); "
            f"r=[{self.r_min:.2f}, {self.r_max:.2f}] m, "
            f"z=[{self.z_min:.2f}, {self.z_max:.2f}] m, "
            f"voxel={self.voxel_size:.2f} m"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_cloud(self, msg: PointCloud2):
        # Rate limit
        now = self.get_clock().now()
        elapsed = (now - self._last_pub_time).nanoseconds * 1e-9
        if elapsed < self.publish_period:
            return
        self._last_pub_time = now

        # Lookup TF (source → target) at the cloud timestamp
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout),
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup failed ({self.source_frame} → {self.target_frame}): {ex}",
                throttle_duration_sec=2.0,
            )
            return

        # Read XYZ from the raw cloud.
        # read_points() returns a STRUCTURED array (dtype with named fields), which
        # cannot be cast to float32: the .view()+.reshape() used before failed with
        # "Cannot cast array data from dtype([('x',...".
        # read_points_numpy() returns a (N, 3) float32 array directly.
        pts = pc2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ).astype(np.float32, copy=False)
        if pts.size == 0:
            return
        pts = pts.reshape(-1, 3)

        # ── Range filter (in source frame) ──
        r = np.linalg.norm(pts, axis=1)
        mask = (r >= self.r_min) & (r <= self.r_max)
        pts = pts[mask]
        if pts.size == 0:
            return

        # ── Transform to target frame ──
        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_to_rotmat(q.x, q.y, q.z, q.w)
        pts_world = (R @ pts.T).T + np.array([t.x, t.y, t.z], dtype=np.float64)

        # ── Height filter (in target frame) ──
        mask = (pts_world[:, 2] >= self.z_min) & (pts_world[:, 2] <= self.z_max)
        pts_world = pts_world[mask]
        if pts_world.size == 0:
            return

        # ── Voxel downsample ──
        if self.voxel_size > 0.0:
            keys = np.floor(pts_world / self.voxel_size).astype(np.int64)
            _, unique_idx = np.unique(
                keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791,
                return_index=True,
            )
            pts_world = pts_world[unique_idx]

        # ── Publish ──
        out = pc2.create_cloud_xyz32(
            header=self._make_header(msg.header.stamp), points=pts_world.astype(np.float32)
        )
        self.pub.publish(out)

    def _make_header(self, stamp):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = stamp
        h.frame_id = self.target_frame
        return h


def main(args=None):
    rclpy.init(args=args)
    node = LidarFilterNode()
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
