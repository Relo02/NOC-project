#!/usr/bin/env python3
"""
key_teleop — continuous keyboard teleop publishing geometry_msgs/Twist.

Unlike teleop_twist_keyboard (one Twist per keypress), this node holds the
last commanded velocity and republishes it at a fixed rate, so the robot
keeps moving until you explicitly stop it with SPACE. Ideal for driving the
static G1 around the map with gz_pose_teleop.

Keys:
    w / s : forward / backward
    a / d : strafe left / right
    q / e : rotate left / right
  SPACE   : stop (zero all velocities)
    z / x : decrease / increase linear speed
    c / v : decrease / increase angular speed
  CTRL-C  : quit
"""

import sys
import termios
import tty
import select
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

BINDINGS = {
    'w': (1.0, 0.0, 0.0),
    's': (-1.0, 0.0, 0.0),
    'a': (0.0, 1.0, 0.0),
    'd': (0.0, -1.0, 0.0),
    'q': (0.0, 0.0, 1.0),
    'e': (0.0, 0.0, -1.0),
}

HELP = """
key_teleop — guida continua
---------------------------
  w/s : avanti / indietro
  a/d : traslazione a sinistra / destra
  q/e : rotazione a sinistra / destra
  SPAZIO : ferma
  z/x : velocità lineare -/+
  c/v : velocità angolare -/+
  CTRL-C : esci
"""


class KeyTeleop(Node):
    def __init__(self):
        super().__init__('key_teleop')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('linear_speed', 0.6)
        self.declare_parameter('angular_speed', 0.6)

        topic = self.get_parameter('cmd_vel_topic').value
        self.rate = float(self.get_parameter('publish_rate').value)
        self.lin = float(self.get_parameter('linear_speed').value)
        self.ang = float(self.get_parameter('angular_speed').value)

        self.pub = self.create_publisher(Twist, topic, 10)
        self.vx = self.vy = self.wz = 0.0
        self._running = True

        print(HELP)
        self.get_logger().info(
            f"Publishing on {topic} | linear={self.lin:.2f} angular={self.ang:.2f}")

        # Republish the held velocity at a fixed rate (so motion continues)
        self.timer = self.create_timer(1.0 / self.rate, self.publish)

        # Read the keyboard in a dedicated thread to avoid starving the
        # ROS executor / timer callback.
        self.settings = termios.tcgetattr(sys.stdin)
        self.key_thread = threading.Thread(target=self._key_loop, daemon=True)
        self.key_thread.start()

    def _key_loop(self):
        try:
            tty.setraw(sys.stdin.fileno())
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                key = sys.stdin.read(1)
                self._handle_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def _handle_key(self, key):
        if key == '\x03':  # CTRL-C
            self._running = False
            rclpy.shutdown()
            return
        if key == ' ':
            self.vx = self.vy = self.wz = 0.0
        elif key in BINDINGS:
            fx, fy, fw = BINDINGS[key]
            self.vx = fx * self.lin
            self.vy = fy * self.lin
            self.wz = fw * self.ang
        elif key in ('z', 'x'):
            self.lin = max(0.05, self.lin * (0.9 if key == 'z' else 1.1))
            sys.stdout.write(f"\r\nlinear speed = {self.lin:.2f} m/s\r\n")
        elif key in ('c', 'v'):
            self.ang = max(0.05, self.ang * (0.9 if key == 'c' else 1.1))
            sys.stdout.write(f"\r\nangular speed = {self.ang:.2f} rad/s\r\n")

    def publish(self):
        msg = Twist()
        msg.linear.x = self.vx
        msg.linear.y = self.vy
        msg.angular.z = self.wz
        self.pub.publish(msg)

    def stop(self):
        self._running = False
        self.pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
