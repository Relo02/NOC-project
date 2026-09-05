"""
goal_relay_node.py

Bridges RViz's "2D Goal Pose" tool (/goal_pose) onto the topic the
A* planner subscribes to (/global_goal). Also re-stamps the header
to the configured target frame (defaults to odom) so the planner
doesn't choke on a stale "map" frame when SLAM isn't running.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped


class GoalRelayNode(Node):

    def __init__(self):
        super().__init__("robot_real_goal_relay")

        self.declare_parameter("input_topic",    "/goal_pose")
        self.declare_parameter("output_topic",   "/global_goal")
        self.declare_parameter("override_frame", "odom")
        self.declare_parameter("force_frame",    True)

        self.input_topic    = self.get_parameter("input_topic").value
        self.output_topic   = self.get_parameter("output_topic").value
        self.override_frame = self.get_parameter("override_frame").value
        self.force_frame    = bool(self.get_parameter("force_frame").value)

        self.sub = self.create_subscription(
            PoseStamped, self.input_topic, self._on_goal, 10
        )
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)

        self.get_logger().info(
            f"goal_relay: {self.input_topic} → {self.output_topic} "
            f"(force_frame={self.force_frame}, override_frame={self.override_frame})"
        )

    def _on_goal(self, msg: PoseStamped):
        out = PoseStamped()
        out.pose = msg.pose
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.override_frame if self.force_frame else msg.header.frame_id
        self.pub.publish(out)
        self.get_logger().info(
            f"relayed goal x={out.pose.position.x:+.2f} y={out.pose.position.y:+.2f} "
            f"frame={out.header.frame_id}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown() may already have been called by the signal handler:
        # calling it again raises RCLError and makes the node exit with code 1 even
        # when the shutdown is orderly.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
