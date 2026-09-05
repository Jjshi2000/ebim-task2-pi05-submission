"""Typed real-rig discovery and parsing; no ROS imports in the testable helpers."""
from __future__ import annotations

from collections import deque
import math
import re
import statistics
import time

from navigation.types import SafeStop

JS = "sensor_msgs/msg/JointState"
F32 = "std_msgs/msg/Float32"
IMAGE = "sensor_msgs/msg/Image"


def channel_specs():
    specs = {
        "head": ([r"/head_camera/[^/]+/(?:rgb/color/rect/image|rgb/color/rect/image_rect_color|rgb/image_rect_color|left/image_rect_color)"], (IMAGE,)),
        "odom": ([r"/swerve_drive_controller/odom"], ("nav_msgs/msg/Odometry",)),
        "base_cmd": ([r"/swerve_drive_controller/cmd_vel"], ("geometry_msgs/msg/TwistStamped", "geometry_msgs/msg/Twist")),
        "spine": ([r"/spine/joint_states"], (JS,)),
        "spine_cmd": ([r"/spine/target_height"], (F32,)),
        "lidar_front": ([r"/lidar_front/scan"], ("sensor_msgs/msg/LaserScan",)),
        "lidar_rear": ([r"/lidar_rear/scan"], ("sensor_msgs/msg/LaserScan",)),
    }
    for side in ("left", "right"):
        specs[side] = ([f"/{side}/franka_robot_state_broadcaster/measured_joint_states"], (JS,))
        specs[f"{side}_wrench"] = ([f"/{side}/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"], ("geometry_msgs/msg/WrenchStamped",))
        specs[f"{side}_cmd"] = ([f"/{side}/follower/gello/joint_states", f"/{side}/gello/joint_states"], (JS,))
        specs[f"{side}_gripper"] = ([f"/{side}/follower/gripper/joint_states", f"/{side}/gripper/(?:joint_states|gripper_joint_states)"], (JS,))
        specs[f"{side}_gripper_cmd"] = ([f"/{side}/follower/gripper/gripper_client/target_gripper_width_percent", f"/{side}/gripper/gripper_client/target_gripper_width_percent"], (F32,))
        specs[f"wrist_{side}"] = ([f"/wrist_camera_{side}/(?:[^/]+/)?color/image_raw"], (IMAGE,))
    return specs


def resolve_channels(topics, overrides=None, require_rear=True, bootstrap=False):
    known = {name: set(types) for name, types in topics}
    channels, missing = {}, []
    for key, (patterns, accepted) in channel_specs().items():
        explicit = (overrides or {}).get(key)
        choices = []
        for pattern in ([re.escape(explicit)] if explicit else patterns):
            choices = [(name, next(iter(types))) for name, types in known.items()
                       if re.fullmatch(pattern, name) and len(types) == 1 and types.issubset(accepted)]
            if choices:
                break
        if not choices and bootstrap and key in ("left_cmd", "right_cmd"):
            side = key.split("_")[0]
            # An unloaded controller may not expose its subscription yet. Use
            # the rig's observed gripper namespace, then REQUIRE that subscriber
            # after activation and before moving away from the measured pose.
            grip = channels.get(side + "_gripper_cmd")
            if grip is None:
                grip_names = [n for n in known if n in (
                    f"/{side}/follower/gripper/gripper_client/target_gripper_width_percent",
                    f"/{side}/gripper/gripper_client/target_gripper_width_percent")]
                if len(grip_names) == 1:
                    grip = (grip_names[0], F32)
            topic = explicit
            if not topic and grip:
                topic = f"/{side}/" + ("follower/" if "/follower/" in grip[0] else "") + "gello/joint_states"
            if topic and topic not in known:
                choices = [(topic, JS)]
        if len(choices) == 1:
            channels[key] = choices[0]
        elif key != "lidar_rear" or require_rear:
            missing.append(f"{key} ({'ambiguous' if choices else 'absent/wrong type'})")
    return channels, missing


def arm_sample(names, positions) -> tuple[list[str], list[float]]:
    if len(names) != len(positions):
        raise SafeStop("arm names/positions length mismatch")
    found = {}
    for name, position in zip(names, positions):
        if any(word in name.lower() for word in ("finger", "knuckle")):
            continue
        match = re.search(r"joint_?([1-7])$", name)
        if match:
            index = int(match[1])
            if index in found or not math.isfinite(position):
                raise SafeStop("duplicate or non-finite arm joint")
            found[index] = (str(name), float(position))
    if set(found) != set(range(1, 8)):
        raise SafeStop("arm sample must contain all seven named joints")
    return [found[i][0] for i in range(1, 8)], [found[i][1] for i in range(1, 8)]


def gripper_sample(names, positions) -> float:
    if len(names) != len(positions):
        raise SafeStop("gripper names/positions length mismatch")
    candidates = [(name, float(value)) for name, value in zip(names, positions)
                  if name.endswith("left_knuckle_joint") and "inner" not in name]
    if len(candidates) != 1 or not math.isfinite(candidates[0][1]):
        raise SafeStop("missing/ambiguous/non-finite Robotiq driver knuckle")
    return candidates[0][1]


def spine_metres(value: float, unit: str) -> float:
    if not math.isfinite(value):
        raise SafeStop("non-finite spine state")
    if unit not in ("auto", "m", "mm"):
        raise ValueError("spine unit must be auto, m or mm")
    return value / 1000.0 if unit == "mm" or (unit == "auto" and abs(value) > 5) else value


class SourceClock:
    """Detect frozen stamps and extrapolate controller-host time without local wall-clock assumptions."""
    def __init__(self):
        self.stamp = None
        self.arrival = None
        self.offsets = deque(maxlen=50)
        self.unstamped = False
        self.last_effective = None

    def observe(self, stamp: float, now: float, required=True) -> bool:
        if not math.isfinite(stamp) or stamp < 0:
            raise SafeStop("invalid source timestamp")
        if stamp == 0:
            if required:
                raise SafeStop("unstamped required sensor")
            if self.stamp is not None:
                return False  # A stamped source cannot evade freeze detection by resetting to zero.
            self.arrival, self.unstamped = now, True
            return True
        if self.stamp is not None and stamp <= self.stamp:
            return False
        self.stamp, self.arrival, self.unstamped = stamp, now, False
        self.offsets.append(stamp - now)
        return True

    def remote_time(self, now: float, bias: float) -> float:
        if not self.offsets:
            # HKUST falls back to local wall time until a host clock is known.
            return time.time() + bias
        return now + statistics.median(self.offsets) + bias

    def effective_arrival(self, now: float) -> float:
        if self.unstamped or not self.offsets or self.stamp is None:
            return now
        value = min(now, self.stamp - max(self.offsets))
        self.last_effective = max(value, self.last_effective if self.last_effective is not None else value)
        return self.last_effective
