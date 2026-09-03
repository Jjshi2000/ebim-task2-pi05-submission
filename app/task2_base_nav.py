#!/usr/bin/env python3
"""Approach the worktable using the official EBiM mobile-base ROS topics."""
from __future__ import annotations

import argparse
import math


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--odom-topic", default="/swerve_drive_controller/odom")
    p.add_argument("--pose-topic", default="/mobile_base/pose")
    p.add_argument("--cmd-topic", default="/swerve_drive_controller/cmd_vel")
    p.add_argument("--target-x", type=float)
    p.add_argument("--target-y", type=float)
    p.add_argument("--target-yaw", type=float)
    p.add_argument("--forward-distance", type=float, default=0.0,
                   help="metres forward from the startup pose; 0 disables navigation")
    p.add_argument("--lateral-distance", type=float, default=0.0)
    p.add_argument("--relative-yaw", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--max-linear", type=float, default=0.25)
    p.add_argument("--max-angular", type=float, default=0.45)
    p.add_argument("--position-tolerance", type=float, default=0.05)
    p.add_argument("--yaw-tolerance", type=float, default=0.08)
    p.add_argument("--odom-timeout", type=float, default=1.0)
    p.add_argument("--hold-after-goal", type=float, default=1.0)
    args = p.parse_args()
    if args.rate <= 0 or args.max_linear < 0 or args.max_angular < 0:
        p.error("rate must be > 0 and speed limits must be >= 0")
    explicit = all(v is not None for v in (args.target_x, args.target_y, args.target_yaw))
    partial = any(v is not None for v in (args.target_x, args.target_y, args.target_yaw))
    if partial and not explicit:
        p.error("target-x, target-y and target-yaw must be supplied together")
    if explicit and args.forward_distance:
        p.error("use either explicit target or forward-distance")

    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                     reliability=ReliabilityPolicy.BEST_EFFORT)

    class Navigator(Node):
        def __init__(self):
            super().__init__("task2_base_navigator")
            self.pose = None
            self.target = None
            self.goal_time = None
            self.pub = self.create_publisher(TwistStamped, args.cmd_topic, 10)
            self.create_subscription(Odometry, args.odom_topic, self.odom_cb, qos)
            if args.pose_topic:
                self.create_subscription(PoseStamped, args.pose_topic, self.pose_cb, qos)
            self.timer = self.create_timer(1.0 / args.rate, self.tick)
            self.get_logger().info("navigation active: odom=%s cmd=%s", args.odom_topic, args.cmd_topic)

        @staticmethod
        def yaw(q):
            return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        def odom_cb(self, m):
            p = m.pose.pose.position
            self.pose = (p.x, p.y, self.yaw(m.pose.pose.orientation), self.get_clock().now().nanoseconds * 1e-9)

        def pose_cb(self, m):
            if self.pose and self.get_clock().now().nanoseconds * 1e-9 - self.pose[3] < args.odom_timeout:
                return
            p = m.pose.position
            self.pose = (p.x, p.y, self.yaw(m.pose.orientation), self.get_clock().now().nanoseconds * 1e-9)

        def publish(self, x, y, wz):
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "base_link"
            m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = float(x), float(y), float(wz)
            self.pub.publish(m)

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.pose is None or now - self.pose[3] > args.odom_timeout:
                self.publish(0, 0, 0)
                return
            if self.target is None:
                x, y, yaw = self.pose[:3]
                if explicit:
                    self.target = (args.target_x, args.target_y, args.target_yaw)
                elif args.forward_distance > 0:
                    c, s = math.cos(yaw), math.sin(yaw)
                    self.target = (x + c * args.forward_distance - s * args.lateral_distance,
                                   y + s * args.forward_distance + c * args.lateral_distance,
                                   wrap_to_pi(yaw + args.relative_yaw))
                else:
                    self.get_logger().info("no navigation distance requested; holding base")
                    self.publish(0, 0, 0)
                    rclpy.shutdown()
                    return
                self.get_logger().info("resolved target=(%.3f, %.3f, %.3f)", *self.target)
            x, y, yaw = self.pose[:3]
            ex, ey = self.target[0] - x, self.target[1] - y
            dist = math.hypot(ex, ey)
            eyaw = wrap_to_pi(self.target[2] - yaw)
            if dist <= args.position_tolerance and abs(eyaw) <= args.yaw_tolerance:
                self.publish(0, 0, 0)
                if self.goal_time is None:
                    self.goal_time = now
                    self.get_logger().info("table approach complete; holding zero velocity")
                if now - self.goal_time >= args.hold_after_goal:
                    self.get_logger().info("navigation finished")
                    rclpy.shutdown()
                return
            self.goal_time = None
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = 0.8 * (c * ex + s * ey), 0.8 * (-s * ex + c * ey)
            speed = math.hypot(vx, vy)
            if speed > args.max_linear:
                vx, vy = vx * args.max_linear / speed, vy * args.max_linear / speed
            wz = max(-args.max_angular, min(args.max_angular, 1.2 * eyaw))
            self.publish(vx, vy, wz)

    rclpy.init()
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish(0, 0, 0)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
