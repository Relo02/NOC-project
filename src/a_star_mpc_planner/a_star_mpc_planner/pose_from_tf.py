"""
pose_from_tf — publishes the robot pose as a PoseStamped, reading it from TF.

ALTERNATIVE pose source to odom_to_pose_node: instead of republishing /odom, it
queries TF at a fixed rate. Useful when the pose comes from a chain of transforms
(EKF, SLAM) and not from a single Odometry topic.

Default of this stack: `odom -> base_link`, i.e. the planning frame, because
there is no prior map here (A* works on the moving-horizon Gaussian grid). In the
original project the defaults were `map -> base_footprint`, because there the
pose came from slam_toolbox in localization mode.

WHY THE FREEZE CHECK

The lookup asks for ``Time()`` = "the latest available", so when odometry stops
feeding TF the buffer keeps returning the same transform FOREVER. Republishing it
with the current stamp would be indistinguishable from a healthy pose:
downstream, the planner would keep replanning from a pose that no longer moves,
and the robot would walk against a stale pose.

Hence: publishing continues only as long as the transform's OWN stamp advances.
If it stays put for tf_freeze_timeout, publishing stops, the pose topic goes
quiet, and the timeout downstream stops the robot.

The comparison is between consecutive stamps, measuring how long they have been
still on the LOCAL clock: a remote stamp is never compared with the local clock,
so it holds even when the producer of the pose is on the other side of the link
and the two clocks disagree.

It catches a FROZEN pose, not a WRONG one: odometry that drifts keeps its stamps
advancing and passes this check.

Ripreso dal progetto Unitree-G1 (policypilot/navigation/pose_from_tf.py).
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


class PoseFromTf(Node):
    def __init__(self):
        super().__init__('pose_from_tf')
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('pose_topic', '/robot_pose')
        # Stop publishing once the transform's own stamp has stood still this
        # long (see the module docstring). 0 or less disables the check.
        self.declare_parameter('tf_freeze_timeout', 0.5)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.freeze_timeout = float(self.get_parameter('tf_freeze_timeout').value)

        self._last_stamp_ns = None    # stamp of the last transform we saw
        self._last_change = None      # local time that stamp last advanced
        self._frozen = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        rate = float(self.get_parameter('rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)
        self._warned = False
        self.get_logger().info(
            f"pose_from_tf: {self.map_frame}->{self.base_frame} → "
            f"{self.get_parameter('pose_topic').value} "
            f"(freeze timeout {self.freeze_timeout:.2f}s)")

    def _is_frozen(self, tf) -> bool:
        """True once the transform's own stamp has stood still longer than
        tf_freeze_timeout — i.e. TF is being re-served, not re-computed."""
        if self.freeze_timeout <= 0.0:
            return False
        stamp_ns = Time.from_msg(tf.header.stamp).nanoseconds
        now = self.get_clock().now()
        if stamp_ns != self._last_stamp_ns:
            self._last_stamp_ns = stamp_ns
            self._last_change = now
            if self._frozen:
                self._frozen = False
                self.get_logger().warn(
                    f"TF {self.map_frame}->{self.base_frame} is advancing again "
                    f"— /robot_pose resumed. The robot navigated blind for part "
                    f"of the gap: check where it actually is before re-arming.")
            return False
        if self._last_change is None:
            self._last_change = now
            return False
        held = (now - self._last_change).nanoseconds * 1e-9
        if held <= self.freeze_timeout:
            return False
        if not self._frozen:
            self._frozen = True
            self.get_logger().error(
                f"*** TF {self.map_frame}->{self.base_frame} FROZEN for "
                f"{held:.2f}s (same stamp re-served) — the odometry stopped "
                f"feeding TF. Muting /robot_pose so the MPC stops instead of "
                f"walking against a stale pose. ***")
        else:
            self.get_logger().error(
                f"    {self.map_frame}->{self.base_frame} still frozen "
                f"({held:.1f}s)", throttle_duration_sec=2.0)
        return True

    def _tick(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as e:
            if not self._warned:
                self.get_logger().warn(
                    f"waiting for TF {self.map_frame}->{self.base_frame}: {e}",
                    throttle_duration_sec=3.0)
                self._warned = True
            return
        if self._is_frozen(tf):
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseFromTf()
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
