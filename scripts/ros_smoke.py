#!/usr/bin/env python3
"""Isolated ROS plumbing tests. Run on a private ROS domain without real hardware."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" if (ROOT / "app").is_dir() else ROOT))
import cv2
import numpy as np
from navigation.config import load_config
from navigation.demo_bank import DemoBank
from navigation.base_motion import BaseMotion


def fake_rig(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Image, LaserScan
    from geometry_msgs.msg import Twist, TwistStamped, WrenchStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32
    from std_srvs.srv import SetBool
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from controller_manager_msgs.msg import ControllerState
    from tf2_ros import StaticTransformBroadcaster

    cfg = load_config(ROOT / "config/task2.real.yaml")
    bank = DemoBank(cfg["navigation"]["visual"]["bank_path"], cfg["navigation"]["visual"])
    rclpy.init()
    node = Node("fake_task2_rig")
    q = {s: list(bank.metadata["start_pose"][s]) for s in ("left", "right")}
    target = {s: list(v) for s, v in q.items()}
    pose = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]
    last_arm = {}
    report = dict(arm_commands=0, base_commands=0, nonzero_base=0, max_arm_gap_s=0,
                  stamp_errors=0, name_errors=0, last_base=[0, 0, 0], activations=0)
    controllers = {s: "inactive" for s in q}
    frozen = [False]
    pubs = {}
    offset = 12345.0

    def stamp():
        return rclpy.time.Time(seconds=time.time() + offset).to_msg()

    def arm(side, msg):
        now = time.monotonic()
        if side in last_arm:
            report["max_arm_gap_s"] = max(report["max_arm_gap_s"], now - last_arm[side])
        last_arm[side] = now
        report["arm_commands"] += 1
        if msg.name != [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]:
            report["name_errors"] += 1
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if abs(ts - (time.time() + offset)) > .4:
            report["stamp_errors"] += 1
        target[side] = list(msg.position)

    def base(msg):
        t = msg.twist if hasattr(msg, "twist") else msg
        velocity[:] = [t.linear.x, t.linear.y, t.angular.z]
        report["base_commands"] += 1
        report["nonzero_base"] += int(any(velocity))
        report["last_base"] = velocity.copy()

    def list_controllers(side, req, res):
        res.controller = [ControllerState(name="joint_impedance_controller", state=controllers[side])]
        return res

    def switch(side, req, res):
        res.ok = req.activate_controllers == ["joint_impedance_controller"]
        if res.ok:
            controllers[side] = "active"
            report["activations"] += 1
        return res

    def freeze(req, res):
        frozen[0], res.success = req.data, True
        return res

    for side in q:
        ns = f"/{side}/follower" if args.layout == "follower" else f"/{side}"
        pubs[side] = node.create_publisher(JointState, f"/{side}/franka_robot_state_broadcaster/measured_joint_states", 5)
        pubs[side + "_grip"] = node.create_publisher(JointState, ns + "/gripper/joint_states", 5)
        pubs[side + "_wrench"] = node.create_publisher(WrenchStamped, f"/{side}/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame", 5)
        node.create_subscription(JointState, ns + "/gello/joint_states", lambda msg, s=side: arm(s, msg), 5)
        node.create_subscription(Float32, ns + "/gripper/gripper_client/target_gripper_width_percent", lambda msg: None, 5)
        node.create_service(ListControllers, f"/{side}/controller_manager/list_controllers", lambda req, res, s=side: list_controllers(s, req, res))
        node.create_service(SwitchController, f"/{side}/controller_manager/switch_controller", lambda req, res, s=side: switch(s, req, res))
    base_type = TwistStamped if args.layout == "follower" else Twist
    node.create_subscription(base_type, "/swerve_drive_controller/cmd_vel", base, 5)
    node.create_subscription(Float32, "/spine/target_height", lambda msg: None, 5)
    node.create_service(SetBool, "/test/freeze_head", freeze)
    pubs["spine"] = node.create_publisher(JointState, "/spine/joint_states", 5)
    pubs["odom"] = node.create_publisher(Odometry, "/swerve_drive_controller/odom", 5)
    for direction in ("front", "rear"):
        pubs[direction] = node.create_publisher(LaserScan, f"/lidar_{direction}/scan", 5)
    names = {"head": "/head_camera/zed_node/rgb/color/rect/image" if args.layout == "follower" else "/head_camera/zed/rgb/image_rect_color",
             "wrist_left": "/wrist_camera_left/camera/color/image_raw",
             "wrist_right": "/wrist_camera_right/color/image_raw"}
    for name, topic in names.items():
        pubs[name] = node.create_publisher(Image, topic, 2)
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = "base_link", "laser"
    transform.transform.rotation.w = 1.0
    transform.header.stamp = stamp()
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transform)
    head = cv2.cvtColor(cv2.resize(bank.references[0].gray, (1280, 720)), cv2.COLOR_GRAY2BGRA)
    images = {"head": head, "wrist_left": np.zeros((480, 640, 3), np.uint8), "wrist_right": np.zeros((480, 640, 3), np.uint8)}

    def tick():
        ts = stamp()
        for side in q:
            q[side] = target[side].copy()
            msg = JointState()
            msg.header.stamp = ts
            msg.name, msg.position = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)], q[side]
            pubs[side].publish(msg)
            msg.name, msg.position = [side + "_robotiq_85_left_knuckle_joint"], [0.0]
            pubs[side + "_grip"].publish(msg)
            w = WrenchStamped()
            w.header.stamp = ts
            pubs[side + "_wrench"].publish(w)
        msg = JointState()
        msg.header.stamp, msg.name, msg.position = ts, ["spine_joint"], [.434]
        pubs["spine"].publish(msg)
        yaw = pose[2]
        pose[0] += .05 * (math.cos(yaw) * velocity[0] - math.sin(yaw) * velocity[1])
        pose[1] += .05 * (math.sin(yaw) * velocity[0] + math.cos(yaw) * velocity[1])
        pose[2] += .05 * velocity[2]
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = ts, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = pose[:2]
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = velocity
        pubs["odom"].publish(odom)
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = ts, "laser"
        scan.angle_min, scan.angle_max, scan.angle_increment = -math.pi, math.pi, math.pi / 180
        scan.range_min, scan.range_max, scan.ranges = .05, 10., [5.] * 361
        pubs["front"].publish(scan)
        pubs["rear"].publish(scan)
        for name, image in images.items():
            if name == "head" and frozen[0]:
                continue
            msg = Image()
            msg.header.stamp = ts
            msg.height, msg.width = image.shape[:2]
            msg.encoding = "bgra8" if name == "head" else "rgb8"
            msg.step, msg.data = image.shape[1] * image.shape[2], image.tobytes()
            pubs[name].publish(msg)

    node.create_timer(.05, tick)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        node.destroy_node()


def run(args):
    from real_io import RealIO
    from std_srvs.srv import SetBool
    from navigation.types import SafeStop
    cfg = load_config(ROOT / "config/task2.real.yaml")
    cv2.setNumThreads(2)
    fake = subprocess.Popen([sys.executable, __file__, "--rig", "--layout", args.layout, "--report", args.report])
    io = None
    try:
        io = RealIO(cfg, dry_run=True)
        io.wait_ready()
        assert not io.pubs
        io.close()
        io = None
        time.sleep(.3)
        io = RealIO(cfg)
        io.wait_ready()
        bank = DemoBank(cfg["navigation"]["visual"]["bank_path"], cfg["navigation"]["visual"])
        io.prepare(bank.metadata["start_pose"])
        io.verify_start_pose(bank.metadata["start_pose"])
        motion = BaseMotion(io, cfg["navigation"])
        motion.execute_base_delta(.02, 0, 0)
        motion.execute_base_delta(0, -.015, 0)
        time.sleep(.7)
        assert not io.fault, io.fault
        if args.policy:
            from navigation.coordinator import Coordinator
            from task2_real_runner import policy_loop
            assert Coordinator(io, bank, cfg["navigation"]).run(verify_only=True)
            action = [*bank.metadata["start_pose"]["left"], 1., *bank.metadata["start_pose"]["right"], 1., 434.]
            class Client:
                def request(self, observation):
                    time.sleep(.15)
                    return np.repeat(np.array(action, np.float32)[None], 50, axis=0)
                def close(self):
                    pass
            cfg["policy"]["max_episode_s"] = 2.0
            events = []
            policy_loop(io, Client(), cfg, lambda event, **fields: events.append(event), bank)
            assert "policy_started" in events and "policy_chunk" in events, events
            assert io.fault == "policy loop ended", io.fault
        elif args.layout == "follower":
            io.call(SetBool, "/test/freeze_head", SetBool.Request(data=True))
            time.sleep(1.0)
            assert io.fault and ("camera" in io.fault or "head" in io.fault), io.fault
        else:
            io.command(.025, 0, 0)
            time.sleep(.4)
            assert io.fault and "lease" in io.fault, io.fault
        motion.final_stop()
        print("ROS scenario passed:", args.layout, io.fault, flush=True)
    finally:
        if io is not None:
            io.close()
        fake.terminate()
        fake.wait(timeout=10)
    report = json.loads(Path(args.report).read_text())
    assert report["nonzero_base"] > 0 and report["arm_commands"] > 10, report
    assert report["max_arm_gap_s"] < .5 and not report["stamp_errors"] and not report["name_errors"], report
    assert not any(report["last_base"]) and report["activations"] == 2, report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rig", action="store_true")
    ap.add_argument("--layout", choices=("follower", "plain"), default="follower")
    ap.add_argument("--policy", action="store_true", help="exercise handoff and mock inference playback")
    ap.add_argument("--report", default="out/ros_smoke.json")
    args = ap.parse_args()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    fake_rig(args) if args.rig else run(args)
