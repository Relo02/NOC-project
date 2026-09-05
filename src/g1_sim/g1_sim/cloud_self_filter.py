"""
cloud_self_filter — remove the robot's own support rig (harness + gantry pole)
from the LiDAR cloud before it feeds SLAM / the costmaps.

Why
---
On the real G1 the Mid-360 sees the **support structure** it hangs from: the
harness clips at the sides of the head and the gantry/winch pole behind. These
are NOT environment — if mapped they become phantom obstacles glued to the
robot. They sit at a FIXED pose relative to the (suspended, stationary) robot,
so we can carve them out with static exclusion zones in a robot-fixed frame.

What it does
------------
Subscribes the raw cloud (`/livox/lidar`) and republishes it (SAME frame/fields,
only points removed) on `/livox/lidar_filtered`, dropping every point that is
either:

  1. within `self_radius` HORIZONTALLY of the sensor origin (a cylinder that
     swallows the close harness + any near self-reflection), and/or
  2. inside any axis-aligned `boxes` defined in `filter_frame` (default
     `base_link`: x forward, y left, z up) — use these for directional rig parts
     like the gantry pole BEHIND the robot, so you don't nuke real near
     obstacles in front.
  3. below a floor cut that RISES WITH RANGE, evaluated in the gravity-aligned
     `floor_frame` (see below).

The range-dependent floor cut
-----------------------------
A walking G1 does not hold the LiDAR level, and `base_stab` only removes the tilt
it manages to estimate. Whatever is left, θ, lifts a floor return at horizontal
range R by R·sin θ — so the floor climbs into the /scan height band at distance
even when it is safely below it nearby. With the band's lower edge 0.19 m above
the floor, 1.4° is enough at 8 m, 1.8° at 6 m, but 3.6° would be needed at 3 m.
Those distant floor points then drift in and out radially as the robot rocks,
which the obstacle tracker sees as a medium-speed cluster travelling coherently
for a second or two: a phantom person, far away, appearing and disappearing.

A single height threshold cannot fix this — the value that clears the floor at
8 m throws away real low obstacles at 1 m. So the cut is a plane that tilts:

    reject where z < floor_z0 + floor_slope * R          (in floor_frame)

floor_slope is the tangent of the residual tilt being defended against (0.05 ≈
3°) and floor_z0 is where the cut sits at the sensor. Above that line a real
obstacle still has to be, so the cost is paid by SHORT things FAR away: at 8 m the
cut sits ~0.49 m above the floor, which a person clears by a metre.

Note this changes /scan only where the scan is REGENERATED from the filtered
cloud (`regen_scan:=true` on the replay launch, or the live pipeline). A bag's
recorded /scan was flattened before this node existed and is unaffected.

Both are off by default (pass-through). Tune them live and watch
`/livox/lidar_filtered` in RViz:

    ros2 param set /cloud_self_filter self_radius 0.9
    # boxes = flat [xmin,xmax,ymin,ymax,zmin,zmax, ...] in filter_frame
    ros2 param set /cloud_self_filter boxes "[-1.2,-0.3,-0.4,0.4,-0.5,1.5]"   # pole behind

Then point pointcloud_to_laserscan / SLAM at `/livox/lidar_filtered`.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException


def _quat_to_rot(qx, qy, qz, qw):
    """3x3 rotation matrix from a quaternion."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


class CloudSelfFilter(Node):
    def __init__(self):
        super().__init__('cloud_self_filter')
        self.declare_parameter('cloud_in', '/livox/lidar')
        self.declare_parameter('cloud_out', '/livox/lidar_filtered')
        # Frame the radius / boxes are evaluated in (robot-fixed, z up).
        self.declare_parameter('filter_frame', 'base_link')
        # Cylinder (horizontal) radius around the sensor; 0 = disabled.
        self.declare_parameter('self_radius', 0.0)
        # Flat list [xmin,xmax,ymin,ymax,zmin,zmax] * N in filter_frame.
        self.declare_parameter('boxes', [], ParameterDescriptor(dynamic_typing=True))
        # ── RANGE-DEPENDENT FLOOR CUT (see the module header) ────────────────
        # Gravity-aligned frame the cut is evaluated in. It CANNOT be filter_frame
        # (base_link pitches with the robot, which is the whole problem); base_stab
        # is the stabilized one. Empty disables the cut.
        self.declare_parameter('floor_frame', 'base_stab')
        # [m] Height of the cut AT the sensor, in floor_frame. The floor itself
        # sits ~0.79 m below base_link when standing, so -0.70 keeps 9 cm of
        # clearance for genuinely low near obstacles.
        self.declare_parameter('floor_z0', -0.70)
        # [m per m] How fast the cut rises with horizontal range. This is the
        # tangent of the residual tilt it defends against: 0.05 ≈ 3°. Raise it if
        # the floor still leaks into /scan at range; lower it if low obstacles far
        # away stop being seen.
        self.declare_parameter('floor_slope', 0.05)

        self.cloud_out_topic = self.get_parameter('cloud_out').value
        self.filter_frame = self.get_parameter('filter_frame').value
        self.floor_frame = self.get_parameter('floor_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, self.get_parameter('cloud_in').value,
                                 self._cloud_cb, sub_qos)
        self.pub = self.create_publisher(PointCloud2, self.cloud_out_topic, pub_qos)
        self.get_logger().info(
            f"cloud_self_filter: {self.get_parameter('cloud_in').value} minus support rig "
            f"→ {self.cloud_out_topic} (filter_frame={self.filter_frame})")

    def _boxes(self):
        """Current boxes param as an (N,6) array, or empty."""
        raw = self.get_parameter('boxes').value
        if raw is None:
            return np.empty((0, 6))
        arr = np.asarray(list(raw), dtype=np.float64).reshape(-1)
        n = (arr.size // 6) * 6
        return arr[:n].reshape(-1, 6) if n else np.empty((0, 6))

    def _to_frame(self, pts, msg, frame):
        """(points in `frame`, sensor origin in `frame`), or (None, None) with no TF.

        The floor cut needs the SAME points in a second, gravity-aligned frame, so
        the transform is factored out rather than duplicated."""
        try:
            tf = self.tf_buffer.lookup_transform(frame, msg.header.frame_id,
                                                 rclpy.time.Time())
        except TransformException:
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        origin = np.array([t.x, t.y, t.z])
        return pts @ _quat_to_rot(q.x, q.y, q.z, q.w).T + origin, origin

    def _cloud_cb(self, msg: PointCloud2):
        radius = float(self.get_parameter('self_radius').value)
        boxes = self._boxes()
        # Nothing to do → pass through untouched.
        if radius <= 0.0 and boxes.shape[0] == 0:
            self.pub.publish(msg)
            return

        step = msg.point_step
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, step)
        if raw.shape[0] == 0:
            self.pub.publish(msg)
            return

        offs = {f.name: f.offset for f in msg.fields}
        if not all(k in offs for k in ('x', 'y', 'z')):
            self.pub.publish(msg)   # unexpected layout → don't touch it
            return

        def comp(name):
            o = offs[name]
            return raw[:, o:o + 4].copy().view(np.float32).reshape(-1)

        pts = np.column_stack((comp('x'), comp('y'), comp('z'))).astype(np.float64)

        # Sensor-frame points → filter_frame (z up). Static when the robot is
        # stationary, but we look it up per-cloud so it also tracks a moving base.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.filter_frame, msg.header.frame_id, rclpy.time.Time())
        except TransformException:
            self.pub.publish(msg)   # no TF yet → pass through (don't drop the cloud)
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_to_rot(q.x, q.y, q.z, q.w)
        fp = pts @ R.T + np.array([t.x, t.y, t.z])   # points in filter_frame

        remove = np.zeros(fp.shape[0], dtype=bool)

        # (1) cylinder around the sensor origin (its position in filter_frame = t)
        if radius > 0.0:
            dx = fp[:, 0] - t.x
            dy = fp[:, 1] - t.y
            remove |= (dx * dx + dy * dy) < (radius * radius)

        # (2) axis-aligned exclusion boxes
        for (x0, x1, y0, y1, z0, z1) in boxes:
            remove |= ((fp[:, 0] >= x0) & (fp[:, 0] <= x1) &
                       (fp[:, 1] >= y0) & (fp[:, 1] <= y1) &
                       (fp[:, 2] >= z0) & (fp[:, 2] <= z1))

        # (3) range-dependent floor cut, in the GRAVITY-ALIGNED frame
        z0 = float(self.get_parameter('floor_z0').value)
        slope = float(self.get_parameter('floor_slope').value)
        if self.floor_frame and slope >= 0.0:
            gp, gt = self._to_frame(pts, msg, self.floor_frame)
            if gp is not None:
                rng = np.hypot(gp[:, 0] - gt[0], gp[:, 1] - gt[1])
                remove |= gp[:, 2] < (z0 + slope * rng)

        kept = raw[~remove]
        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(kept.shape[0])
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = step
        out.row_step = step * out.width
        out.is_dense = msg.is_dense
        out.data = kept.tobytes()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CloudSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
