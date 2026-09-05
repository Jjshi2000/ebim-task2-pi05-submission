from __future__ import annotations

import math
import threading
import time

import numpy as np

from navigation.safety import SafetyMonitor, direction_clearance, scan_points, transform_scan
from navigation.types import Pose, SafeStop
from real_interfaces import SourceClock, arm_sample, gripper_sample, resolve_channels, spine_metres
from command_stream import CommandStream
from task2_pi05_eval_node import decode_image, build_real_state, LEFT_JOINTS, RIGHT_JOINTS, RIGHT_GRIPPER_DRIVER


class RealIO:
    """ROS observations plus an isolated command process with independent leases."""
    clock = staticmethod(time.monotonic)

    def __init__(self, cfg: dict, dry_run=False, log=lambda *a, **k: None):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, JointState, LaserScan
        from geometry_msgs.msg import WrenchStamped, Twist, TwistStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32
        from tf2_ros import Buffer, TransformListener
        self.cfg, self.dry_run, self.log = cfg, dry_run, log
        self.lock = threading.RLock()
        self.data, self.stamps, self.clocks = {}, {}, {}
        self.fault = None
        self.closed = False
        self.base_target = (0.0, 0.0, 0.0)
        self.base_until = 0.0
        self.arm_targets = {}
        self.arm_names = {}
        self.gripper_targets = {}
        self.spine_target = None
        self.camera_seq = 0
        self.track_since = {}
        self.guard_enabled = False
        self.policy_active = False
        self.policy_until = 0.0
        self.fault_hold_frozen = False
        self.prepared = False
        self.controllers_ready = False
        self.controller_setup = not dry_run
        self.stop_tick = threading.Event()
        self.safety = SafetyMonitor(cfg["navigation"])
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("task2_real_runner")
        deadline = self.clock() + cfg["rig"]["discovery_timeout_s"]
        while True:
            self.channels, missing = resolve_channels(self.node.get_topic_names_and_types(), cfg["rig"]["topics"],
                                                      cfg["navigation"]["safety"]["require_rear_lidar"],
                                                      bootstrap=not dry_run)
            if not missing:
                break
            if self.clock() > deadline:
                self.node.destroy_node()
                rclpy.shutdown()
                raise SafeStop("topic discovery: " + ", ".join(missing))
            rclpy.spin_once(self.node, timeout_sec=0.1)
        log("topics", channels=self.channels)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.node)
        self.types = {"sensor_msgs/msg/JointState": JointState, "sensor_msgs/msg/Image": Image,
                      "sensor_msgs/msg/LaserScan": LaserScan, "geometry_msgs/msg/WrenchStamped": WrenchStamped,
                      "nav_msgs/msg/Odometry": Odometry, "std_msgs/msg/Float32": Float32,
                      "geometry_msgs/msg/TwistStamped": TwistStamped, "geometry_msgs/msg/Twist": Twist}
        self.pubs = {}
        self.stream = None
        for key, (topic, typ) in self.channels.items():
            if key.endswith("_cmd"):
                continue
            else:
                self.clocks[key] = SourceClock()
                endpoints = self.node.get_publishers_info_by_topic(topic)
                reliability = (ReliabilityPolicy.RELIABLE if endpoints and all(
                    e.qos_profile.reliability == ReliabilityPolicy.RELIABLE for e in endpoints)
                    else ReliabilityPolicy.BEST_EFFORT)
                qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=reliability,
                                 durability=DurabilityPolicy.VOLATILE)
                self.node.create_subscription(self.types[typ], topic, lambda m, k=key: self.callback(k, m), qos)
                log("sensor_qos", channel=key, reliability=str(reliability), depth=1)
        if not dry_run:
            try:
                self.check_graph(startup=True, commands_only=True)
                self.stream = CommandStream(self.channels, cfg)
            except BaseException:
                self.node.destroy_node()
                rclpy.shutdown()
                raise
        self.node.create_timer(cfg["rig"]["graph_check_period_s"], self.check_graph)
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()
        self.tick_thread = threading.Thread(target=self.watchdog, daemon=True)
        self.tick_thread.start()

    def watchdog(self):
        while not self.stop_tick.is_set():
            self.tick()
            self.stop_tick.wait(1 / self.cfg["policy"]["rate_hz"])

    def spin(self):
        try:
            self.rclpy.spin(self.node)
        except BaseException as exc:
            if not self.closed:
                self.fault = f"ROS executor stopped: {exc}"

    def callback(self, key, msg):
        try:
            now = self.clock()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # Decode outside the state lock; the command watchdog must not wait on pixels.
            decoded = None
            if key in ("head", "wrist_left", "wrist_right"):
                decoded = decode_image(dict(height=msg.height, width=msg.width, encoding=msg.encoding,
                                            step=msg.step, data=msg.data))
            with self.lock:
                source = self.clocks[key]
                if not source.observe(stamp, now, self.cfg["navigation"]["safety"]["require_stamped_sensors"]):
                    return
                if key in ("left", "right"):
                    names, value = arm_sample(msg.name, msg.position)
                    self.arm_names[key] = names
                elif key in ("left_gripper", "right_gripper"):
                    value = gripper_sample(msg.name, msg.position)
                elif key == "spine":
                    if len(msg.position) != 1:
                        raise SafeStop("ambiguous spine sample")
                    value = spine_metres(float(msg.position[0]), self.cfg["rig"]["spine_state_unit"])
                elif key.endswith("wrench"):
                    w = msg.wrench
                    value = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
                    if not np.isfinite(value).all():
                        raise SafeStop("non-finite wrench")
                elif key in ("head", "wrist_left", "wrist_right"):
                    value = decoded
                    if key == "head":
                        self.camera_seq += 1
                elif key == "odom":
                    p, q, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
                    norm = sum(v * v for v in (q.x, q.y, q.z, q.w))
                    if not math.isfinite(norm) or abs(norm - 1.0) > 0.05:
                        raise SafeStop("invalid odometry quaternion")
                    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                    value = (Pose(p.x, p.y, yaw), (t.linear.x, t.linear.y, t.angular.z))
                    if not np.isfinite([p.x, p.y, yaw, *value[1]]).all():
                        raise SafeStop("non-finite odometry")
                    old = self.data.get("odom_frame")
                    frame = (msg.header.frame_id, msg.child_frame_id)
                    if old is not None and frame != old:
                        raise SafeStop("odometry coordinate frame changed")
                    self.data["odom_frame"] = frame
                else:
                    from rclpy.time import Time
                    sc = self.cfg["navigation"]["safety"]
                    raw = scan_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                      msg.range_min, msg.range_max)
                    if key == "lidar_front":
                        self.data["front_range_m"] = direction_clearance(
                            [raw], 0., sc["front_half_fov_rad"], sc["min_scan_coverage"])
                    target = sc["base_frame"]
                    if target == "auto":
                        target = self.data.get("odom_frame", ("", ""))[1]
                    extrinsic = sc["lidar_extrinsics"].get(msg.header.frame_id)
                    if extrinsic is not None:
                        if len(extrinsic) != 3:
                            raise SafeStop("LiDAR extrinsic must be measured x, y, yaw")
                        x, y, yaw = extrinsic
                        value = scan_points(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min,
                                            msg.range_max, x, y, yaw)
                    else:
                        if not target or not self.tf.can_transform(target, msg.header.frame_id, Time()):
                            return  # Missing TF keeps the scan stale; no guessed orientation.
                        transform = self.tf.lookup_transform(target, msg.header.frame_id, Time()).transform
                        q, p = transform.rotation, transform.translation
                        value = transform_scan(raw, p.x, p.y, (q.x, q.y, q.z, q.w))
                self.data[key], self.stamps[key] = value, source.effective_arrival(now)
        except BaseException as exc:
            self.fault = f"{key}: {exc}"

    def check_graph(self, startup=False, commands_only=False, require_commands=False):
        if self.closed:
            return
        try:
            for key, (topic, typ) in self.channels.items():
                if key.endswith("_cmd"):
                    subscribers = self.node.get_subscriptions_info_by_topic(topic)
                    if any(s.topic_type != typ for s in subscribers):
                        raise SafeStop(f"controller subscriber type mismatch: {topic}")
                    if (require_commands or not self.controller_setup) and not subscribers:
                        raise SafeStop(f"controller subscriber lost: {topic}")
                    foreign = [p.node_name for p in self.node.get_publishers_info_by_topic(topic)
                               if p.node_name not in (self.node.get_name(), self.stream.name if self.stream else "")
                               or p.node_namespace != self.node.get_namespace()]
                    if foreign:
                        raise SafeStop(f"foreign command publisher on {topic}: {foreign}")
                elif not commands_only and self.node.count_publishers(topic) < 1:
                    raise SafeStop(f"sensor publisher lost: {topic}")
        except BaseException as exc:
            if startup:
                raise SafeStop(str(exc)) from exc
            if self.guard_enabled:
                self.fault = str(exc)

    def snapshot(self):
        with self.lock:
            scans = [self.data[k] for k in ("lidar_front", "lidar_rear") if k in self.data]
            required = ["lidar_front"] + (["lidar_rear"] if self.cfg["navigation"]["safety"]["require_rear_lidar"] else [])
            scan_stamp = min((self.stamps.get(k, -1e9) for k in required))
            if "lidar_rear" in self.stamps:
                scan_stamp = min(scan_stamp, self.stamps["lidar_rear"])
            sc = self.cfg["navigation"]["safety"]
            front = self.data.get("front_range_m")
            def clearance(direction):
                if abs(direction) < 1e-9:
                    if front is None:
                        raise SafeStop("missing front LaserScan range")
                    return front
                return direction_clearance(scans, direction, sc["direction_half_fov_rad"], sc["min_scan_coverage"]) - sc["footprint_radius_m"]
            return dict(pose=self.data.get("odom", (None, None))[0],
                        velocity=self.data.get("odom", (None, (0, 0, 0)))[1],
                        odom_stamp=self.stamps.get("odom"), camera_stamp=self.stamps.get("head"),
                        camera_seq=self.camera_seq, image=self.data.get("head"), lidar_stamp=scan_stamp,
                        clearance=clearance,
                        fault=self.fault)

    def observation(self):
        with self.lock:
            now = self.clock()
            required = ("left", "right", "right_gripper", "left_wrench", "right_wrench", "spine")
            for key in required + ("head", "wrist_left", "wrist_right"):
                limit = self.cfg["rig"]["image_max_age_s"] if key in ("head", "wrist_left", "wrist_right") else self.cfg["rig"]["state_max_age_s"]
                if key not in self.stamps or now - self.stamps[key] > limit:
                    raise SafeStop(f"missing/stale policy input: {key}")
            times = [self.stamps[k] for k in ("head", "wrist_left", "wrist_right")]
            if max(times) - min(times) > self.cfg["rig"]["image_max_skew_s"]:
                raise SafeStop(f"camera arrival skew: ages={[round(now - t, 3) for t in times]}")
            joints = dict(zip(LEFT_JOINTS, self.data["left"]))
            joints.update(zip(RIGHT_JOINTS, self.data["right"]))
            joints[RIGHT_GRIPPER_DRIVER] = self.data["right_gripper"]
            state = build_real_state(joints, {s: self.data[s + "_wrench"] for s in ("left", "right")})
            if state is None:
                raise SafeStop("invalid 42D state")
            return state, {k: self.data[k] for k in ("head", "wrist_left", "wrist_right")}, joints

    def wait_ready(self, full_graph=False):
        end = self.clock() + self.cfg["navigation"]["sensor_wait_s"]
        last = "no sensors"
        while self.clock() < end:
            if self.fault:
                raise SafeStop(self.fault)
            try:
                self.observation()
                SafetyMonitor(self.cfg["navigation"]).check(self.snapshot(), self.clock())
                self.check_graph(startup=True, require_commands=full_graph)
                if self.fault:
                    raise SafeStop(self.fault)
                self.guard_enabled = True
                return
            except SafeStop as exc:
                last = str(exc)
                time.sleep(0.05)
        raise SafeStop("sensor startup timeout: " + last)

    def hold_current(self, side):
        topic, typ = self.channels[side + "_cmd"]
        known = dict(self.node.get_topic_names_and_types())
        if topic in known and set(known[topic]) != {typ}:
            raise SafeStop(f"arm command topic has incompatible types: {topic}")
        # A controller may subscribe only in on_activate. Start the current-pose
        # stream before activation (HKUST), then require the subscription after
        # activation and before enabling any posture movement.
        self.check_graph(startup=True)
        with self.lock:
            if self.clock() - self.stamps.get(side, -1e9) > self.cfg["rig"]["state_max_age_s"]:
                raise SafeStop(f"cannot hold stale {side} arm")
            self.arm_targets[side] = list(self.data[side])
        self.sleep(1.0)
        if side not in self.stream.published_sides:
            raise SafeStop(f"arm keepalive was not confirmed: {side}")

    def prepare_controllers(self):
        if self.dry_run or self.controllers_ready:
            return
        from rig_lifecycle import RigLifecycle
        if self.arm_targets:
            raise SafeStop("controller startup may only run before setting arm targets")
        self.controller_setup = True
        lifecycle = RigLifecycle(self)
        lifecycle.base()
        self.wait_ready()
        lifecycle.arms()
        # DDS discovery can lag the controller service response.
        self.wait_ready(full_graph=True)
        self.controller_setup = False
        self.controllers_ready = True

    def sleep(self, seconds):
        time.sleep(seconds)
        if self.fault:
            raise SafeStop(self.fault)

    def command(self, vx, vy, wz):
        if self.dry_run and any((vx, vy, wz)):
            raise SafeStop("nonzero command requested during dry run")
        with self.lock:
            if any((vx, vy, wz)):
                if self.fault:
                    raise SafeStop(self.fault)
                self.safety.check(self.snapshot(), self.clock(), (vx, vy, wz))
            self.base_target = (float(vx), float(vy), float(wz))
            self.base_until = self.clock() + self.cfg["navigation"]["safety"]["command_lease_s"]
            if not any(self.base_target):
                self.publish_base(self.base_target)

    def publish_base(self, command):
        # Publication belongs exclusively to the isolated stream process.
        self.base_target = tuple(float(v) for v in command)

    def tick(self):
        if self.closed:
            return
        with self.lock:
            now = self.clock()
            try:
                if self.guard_enabled:
                    self.safety.check(self.snapshot(), now, self.base_target)
                    self.observation()
                if any(self.base_target) and now > self.base_until:
                    raise SafeStop("base command lease expired")
                if self.policy_active and now > self.policy_until:
                    raise SafeStop("policy command lease expired")
                if self.prepared and abs(self.data["spine"] - self.spine_target / 1000) > self.cfg["rig"]["spine_tolerance_m"]:
                    raise SafeStop("spine left the verified height")
                for side, q in self.arm_targets.items():
                    measured = self.data.get(side)
                    if measured is not None and max(abs(a - b) for a, b in zip(q, measured)) > self.cfg["rig"]["tracking_error_rad"]:
                        self.track_since.setdefault(side, now)
                        if now - self.track_since[side] > self.cfg["rig"]["tracking_timeout_s"]:
                            raise SafeStop(f"{side} arm tracking watchdog")
                    else:
                        self.track_since.pop(side, None)
            except BaseException as exc:
                self.fault = str(exc)
            command = (0.0, 0.0, 0.0) if self.fault else self.base_target
            try:
                self.publish_base(command)
                if self.dry_run:
                    return
                if self.fault and not self.fault_hold_frozen:
                    for side in self.arm_targets:
                        if now - self.stamps.get(side, -1e9) <= self.cfg["rig"]["state_max_age_s"]:
                            self.arm_targets[side] = list(self.data[side])
                    self.fault_hold_frozen = True
                arms = {side: (list(target), list(self.arm_names[side]), self.clocks[side].remote_time(now, 0) - now)
                        for side, target in self.arm_targets.items()}
                measured = {side: (list(self.data[side]), self.stamps[side]) for side in arms}
                self.stream.update(dict(arms=arms, measured=measured, base=command, base_until=self.base_until,
                    grips=dict(self.gripper_targets), spine=self.spine_target, fault=self.fault,
                    policy_active=self.policy_active, policy_until=self.policy_until,
                    stamp_bias_s=self.cfg["rig"]["arm_stamp_bias_s"],
                    state_max_age_s=self.cfg["rig"]["state_max_age_s"]))
            except BaseException as exc:
                self.fault = f"publisher error: {exc}"

    def set_arms(self, left, right, grips=None):
        with self.lock:
            if self.fault:
                raise SafeStop(self.fault)
            if not np.isfinite([*left, *right]).all() or len(left) != 7 or len(right) != 7:
                raise SafeStop("invalid arm command")
            lower, upper = self.cfg["rig"]["joint_lower_rad"], self.cfg["rig"]["joint_upper_rad"]
            for q in (left, right):
                if any(v < lo or v > hi for v, lo, hi in zip(q, lower, upper)):
                    raise SafeStop("arm command outside configured FR3 joint limits")
            if self.dry_run:
                return
            self.arm_targets = {"left": list(left), "right": list(right)}
            self.policy_until = self.clock() + self.cfg["policy"]["max_hold_s"]
            if grips is not None:
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in grips):
                    raise SafeStop("invalid gripper command")
                self.gripper_targets = dict(zip(("left", "right"), grips))

    def call(self, service_type, name, request, timeout=5.0, optional=False):
        client = self.node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                if optional:
                    return None
                raise SafeStop(f"service missing: {name}")
            future = client.call_async(request)
            end = self.clock() + timeout
            while not future.done() and self.clock() < end:
                self.sleep(0.02)
            if not future.done():
                raise SafeStop(f"service timeout: {name}")
            return future.result()
        finally:
            self.node.destroy_client(client)

    def prepare(self, start_pose):
        self.log("prepare", start_pose=start_pose, spine_mm=self.cfg["rig"]["spine_target_mm"], dry_run=self.dry_run)
        if self.dry_run:
            return
        if not self.controllers_ready:
            self.prepare_controllers()
        self.spine_target = self.cfg["rig"]["spine_target_mm"]
        end = self.clock() + self.cfg["rig"]["spine_timeout_s"]
        while abs(self.data["spine"] - self.spine_target / 1000.0) > self.cfg["rig"]["spine_tolerance_m"]:
            if self.clock() > end:
                raise SafeStop("spine did not reach demonstrated height")
            self.sleep(0.05)
        end = self.clock() + self.cfg["rig"]["start_pose_timeout_s"]
        while self.clock() < end:
            commands = []
            for side in ("left", "right"):
                previous = np.asarray(self.arm_targets[side])
                commands.append(previous + np.clip(np.asarray(start_pose[side]) - previous,
                                                   -self.cfg["rig"]["start_pose_step_rad"], self.cfg["rig"]["start_pose_step_rad"]))
            self.set_arms(*commands, grips=(1.0, 1.0))
            self.sleep(1.0 / self.cfg["policy"]["rate_hz"])
            if all(np.max(np.abs(np.asarray(self.data[s]) - start_pose[s])) <= self.cfg["rig"]["start_pose_tolerance_rad"] for s in ("left", "right")):
                self.prepared = True
                return
        raise SafeStop("arms did not reach demonstrated start pose")

    def verify_start_pose(self, start_pose):
        self.observation()
        for side in ("left", "right"):
            if np.max(np.abs(np.asarray(self.data[side]) - start_pose[side])) > self.cfg["rig"]["start_pose_tolerance_rad"]:
                raise SafeStop(f"{side}: arm no longer at demonstrated start pose")
        if abs(self.data["spine"] - self.cfg["rig"]["spine_target_mm"] / 1000) > self.cfg["rig"]["spine_tolerance_m"]:
            raise SafeStop("spine no longer at demonstrated height")

    def close(self):
        try:
            for _ in range(self.cfg["navigation"]["safety"]["repeated_zero_cmd_count"]):
                self.base_target = (0.0, 0.0, 0.0)
                try:
                    self.publish_base(self.base_target)
                except Exception:
                    pass
                time.sleep(1.0 / self.cfg["navigation"]["rate_hz"])
        finally:
            self.closed = True
            self.stop_tick.set()
            self.tick_thread.join(timeout=2)
            if self.stream is not None:
                self.stream.close()
            if self.rclpy.ok():
                self.rclpy.shutdown()
            self.thread.join(timeout=2)
            self.node.destroy_node()
