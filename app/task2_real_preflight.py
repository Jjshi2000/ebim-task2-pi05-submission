#!/usr/bin/env python3
"""Read-only validation of the official EBiM Task 2 real-robot ROS interface."""
from __future__ import annotations

import argparse
import json
import math
import time


INPUT_TYPES = {
    "/left/franka_robot_state_broadcaster/measured_joint_states": "sensor_msgs/msg/JointState",
    "/right/franka_robot_state_broadcaster/measured_joint_states": "sensor_msgs/msg/JointState",
    "/left/franka_robot_state_broadcaster/external_joint_torques": "sensor_msgs/msg/JointState",
    "/right/franka_robot_state_broadcaster/external_joint_torques": "sensor_msgs/msg/JointState",
    "/left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": "geometry_msgs/msg/WrenchStamped",
    "/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": "geometry_msgs/msg/WrenchStamped",
    "/left/gripper/joint_states": "sensor_msgs/msg/JointState",
    "/right/gripper/joint_states": "sensor_msgs/msg/JointState",
    "/spine/joint_states": "sensor_msgs/msg/JointState",
    "/swerve_drive_controller/odom": "nav_msgs/msg/Odometry",
    "/head_camera/zed_node/rgb/color/rect/image": "sensor_msgs/msg/Image",
    "/wrist_camera_left/camera/color/image_raw": "sensor_msgs/msg/Image",
    "/wrist_camera_right/camera/color/image_raw": "sensor_msgs/msg/Image",
}

COMMAND_TYPES = {
    "/left/gello/joint_states": "sensor_msgs/msg/JointState",
    "/right/gello/joint_states": "sensor_msgs/msg/JointState",
    "/left/gripper/gripper_client/target_gripper_width_percent": "std_msgs/msg/Float32",
    "/right/gripper/gripper_client/target_gripper_width_percent": "std_msgs/msg/Float32",
    "/spine/target_height": "std_msgs/msg/Float32",
    "/swerve_drive_controller/cmd_vel": "geometry_msgs/msg/TwistStamped",
}

CAMERA_GEOMETRY = {
    "/head_camera/zed_node/rgb/color/rect/image": (720, 1280),
    "/wrist_camera_left/camera/color/image_raw": (480, 640),
    "/wrist_camera_right/camera/color/image_raw": (480, 640),
}


def finite(values) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--skip-command-subscribers",
        action="store_true",
        help="do not require the real controllers to subscribe to command topics",
    )
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")

    import rclpy
    from geometry_msgs.msg import WrenchStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    rclpy.init()
    node = Node("task2_real_preflight")
    samples = {}
    subscriptions = []

    def save(topic):
        return lambda message: samples.setdefault(topic, message)

    for topic, type_name in INPUT_TYPES.items():
        msg_type = {
            "sensor_msgs/msg/JointState": JointState,
            "geometry_msgs/msg/WrenchStamped": WrenchStamped,
            "nav_msgs/msg/Odometry": Odometry,
            "sensor_msgs/msg/Image": Image,
        }[type_name]
        subscriptions.append(node.create_subscription(msg_type, topic, save(topic), qos_profile_sensor_data))

    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline and len(samples) < len(INPUT_TYPES):
        rclpy.spin_once(node, timeout_sec=0.1)

    visible = {name: set(types) for name, types in node.get_topic_names_and_types()}
    errors = []
    details = {}
    for topic, expected_type in INPUT_TYPES.items():
        actual_types = sorted(visible.get(topic, set()))
        if expected_type not in actual_types:
            errors.append(f"{topic}: expected type {expected_type}, visible={actual_types}")
        message = samples.get(topic)
        if message is None:
            errors.append(f"{topic}: no message received within {args.timeout:.1f}s")
            continue

        if expected_type == "sensor_msgs/msg/JointState":
            details[topic] = {"names": list(message.name), "position": list(message.position)}
            if not finite(message.position):
                errors.append(f"{topic}: position is empty or non-finite")
            if "measured_joint_states" in topic and len(message.position) < 7:
                errors.append(f"{topic}: expected at least 7 arm positions, got {len(message.position)}")
        elif expected_type == "geometry_msgs/msg/WrenchStamped":
            wrench = message.wrench
            values = [wrench.force.x, wrench.force.y, wrench.force.z,
                      wrench.torque.x, wrench.torque.y, wrench.torque.z]
            details[topic] = {"wrench": values}
            if not finite(values):
                errors.append(f"{topic}: wrench is non-finite")
        elif expected_type == "sensor_msgs/msg/Image":
            actual = (int(message.height), int(message.width))
            details[topic] = {"height": actual[0], "width": actual[1], "encoding": message.encoding}
            if actual != CAMERA_GEOMETRY[topic]:
                errors.append(f"{topic}: geometry {actual}, expected {CAMERA_GEOMETRY[topic]}")
            if message.encoding.lower() not in ("rgb8", "bgr8", "rgba8", "bgra8", "8uc3", "8uc4"):
                errors.append(f"{topic}: unsupported image encoding {message.encoding!r}")
        else:
            pose = message.pose.pose
            twist = message.twist.twist
            values = [pose.position.x, pose.position.y, pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w, twist.linear.x, twist.linear.y,
                      twist.angular.z]
            details[topic] = {"odom": values}
            if not finite(values):
                errors.append(f"{topic}: odometry is non-finite")

    command_subscribers = {topic: node.count_subscribers(topic) for topic in COMMAND_TYPES}
    if not args.skip_command_subscribers:
        for topic, expected_type in COMMAND_TYPES.items():
            count = command_subscribers[topic]
            actual_types = sorted(visible.get(topic, set()))
            if expected_type not in actual_types:
                errors.append(f"{topic}: expected type {expected_type}, visible={actual_types}")
            if count < 1:
                errors.append(f"{topic}: no controller subscriber visible")

    result = {
        "ok": not errors,
        "received_inputs": len(samples),
        "expected_inputs": len(INPUT_TYPES),
        "command_subscribers": command_subscribers,
        "details": details,
        "errors": errors,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")

    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
