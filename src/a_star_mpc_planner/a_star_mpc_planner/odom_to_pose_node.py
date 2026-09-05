"""
odom_to_pose_node.py
Republishes nav_msgs/Odometry from /odom as geometry_msgs/PoseStamped on <pose_topic>.

This bridge is needed because the A* and MPC nodes subscribe to <pose_topic>
(PoseStamped) while the plant publishes /odom (Odometry).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class OdomToPoseNode(Node):

    def __init__(self):
        super().__init__('odom_to_pose_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.declare_parameter('pose_topic', '/robot_pose')
        _pose_topic = self.get_parameter('pose_topic').value
        self._pub = self.create_publisher(PoseStamped, _pose_topic, 10)
        self.create_subscription(Odometry, '/odom/raw', self._odom_cb, sensor_qos)
        self._msg_count = 0

        self.get_logger().info(f'odom_to_pose_node ready: /odom/raw -> {_pose_topic}')
        self.get_logger().warn('[ODOM_BRIDGE] Waiting for first /odom/raw message...')

    def _odom_cb(self, msg: Odometry):
        self._msg_count += 1
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # Log first message explicitly so we know the bridge is alive
        if self._msg_count == 1:
            self.get_logger().warn(
                f'[ODOM_BRIDGE] First /odom/raw received! '
                f'pos=({x:.3f}, {y:.3f})  vel=({vx:.3f}, {vy:.3f})'
            )

        # Throttled log: shows whether position is actually updating
        self.get_logger().info(
            f'[ODOM_BRIDGE] #{self._msg_count:05d}  '
            f'pos=({x:.4f}, {y:.4f})  '
            f'vel=({vx:.4f}, {vy:.4f})',
            throttle_duration_sec=2.0,
        )

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose   = msg.pose.pose
        self._pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToPoseNode()
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
