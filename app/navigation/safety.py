from __future__ import annotations

import math

import numpy as np

from .types import Pose, SafeStop, wrap


def scan_points(ranges, angle_min, angle_increment, range_min, range_max,
                sensor_x=0.0, sensor_y=0.0, sensor_yaw=0.0):
    ranges = np.asarray(ranges, dtype=float)
    if (ranges.ndim != 1 or ranges.size < 3 or
            not all(math.isfinite(v) for v in (angle_min, angle_increment, range_min, range_max,
                                               sensor_x, sensor_y, sensor_yaw)) or
            angle_increment == 0 or not 0 <= range_min < range_max):
        raise SafeStop("invalid LiDAR geometry")
    angles = angle_min + np.arange(len(ranges)) * angle_increment + sensor_yaw
    valid = (np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)) | np.isposinf(ranges)
    # +inf is the LaserScan convention for no return within range_max; NaN is unknown.
    distances = np.where(np.isposinf(ranges), range_max, ranges)
    x = sensor_x + distances * np.cos(angles)
    y = sensor_y + distances * np.sin(angles)
    return np.arctan2(y, x), np.hypot(x, y), valid


def direction_clearance(scans: list[tuple], direction: float, half_fov: float, min_coverage: float) -> float:
    if not scans:
        raise SafeStop("no LiDAR coverage")
    angles = np.concatenate([s[0] for s in scans])
    distances = np.concatenate([s[1] for s in scans])
    valid = np.concatenate([s[2] for s in scans])
    relative = np.arctan2(np.sin(angles - direction), np.cos(angles - direction))
    selected = np.abs(relative) <= half_fov
    if not selected.any() or valid[selected].mean() < min_coverage:
        raise SafeStop("unknown LiDAR sector")
    # Reject scans that see only a sliver of the requested motion corridor.
    edges = np.linspace(-half_fov, half_fov, 9)
    if any(not np.any(selected & valid & (relative >= lo) & (relative <= hi)) for lo, hi in zip(edges[:-1], edges[1:])):
        raise SafeStop("incomplete directional LiDAR coverage")
    return float(distances[selected & valid].min())


def transform_scan(scan, x, y, quaternion):
    """Transform a horizontal scan, including scanners mounted upside down."""
    qx, qy, qz, qw = quaternion
    if not np.isfinite([x, y, *quaternion]).all() or abs(sum(v*v for v in quaternion) - 1) > .05:
        raise SafeStop("invalid LiDAR transform")
    r20, r21, r22 = 2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)
    if max(abs(r20), abs(r21)) > .05 or abs(r22) < .95:
        raise SafeStop("LiDAR scan plane is not horizontal in the base frame")
    angles, distances, valid = scan
    sx, sy = distances*np.cos(angles), distances*np.sin(angles)
    bx = x + (1-2*(qy*qy+qz*qz))*sx + 2*(qx*qy-qw*qz)*sy
    by = y + 2*(qx*qy+qw*qz)*sx + (1-2*(qx*qx+qz*qz))*sy
    return np.arctan2(by, bx), np.hypot(bx, by), valid


class SafetyMonitor:
    def __init__(self, nav: dict):
        self.nav = nav
        self.cfg = nav["safety"]
        self.previous: tuple[Pose, float] | None = None
        self.path_m = 0.0
        self.fault: str | None = None

    def check(self, snap: dict, now: float, command=(0.0, 0.0, 0.0)):
        if self.fault:
            raise SafeStop(self.fault)
        try:
            if snap.get("fault"):
                raise SafeStop(snap["fault"])
            for key, age in (("odom_stamp", self.cfg["odom_max_age_s"]),
                             ("camera_stamp", self.cfg["camera_max_age_s"]),
                             ("lidar_stamp", self.cfg["lidar_max_age_s"])):
                stamp = snap.get(key)
                if stamp is None or not math.isfinite(stamp) or not 0 <= now - stamp <= age:
                    raise SafeStop(f"missing or stale {key}")
            pose = snap.get("pose")
            values = [] if pose is None else [pose.x, pose.y, pose.yaw, *snap["velocity"], *command]
            if not values or not all(math.isfinite(v) for v in values):
                raise SafeStop("non-finite pose, velocity or command")
            stamp = snap["odom_stamp"]
            if self.previous is not None and stamp != self.previous[1]:
                old, old_stamp = self.previous
                dt = stamp - old_stamp
                distance = math.hypot(pose.x - old.x, pose.y - old.y)
                rotation = abs(wrap(pose.yaw - old.yaw))
                if (dt <= 0 or distance > self.cfg["max_odom_jump_m"] or
                        rotation > self.cfg["max_odom_jump_rad"] or
                        distance > self.cfg["max_odom_speed_mps"] * dt + self.cfg["odom_noise_m"] or
                        rotation > self.cfg["max_odom_yaw_rps"] * dt + self.cfg["odom_noise_rad"]):
                    raise SafeStop("odometry discontinuity")
                self.path_m += distance
            self.previous = pose, stamp
            vx, vy, wz = command
            if math.hypot(vx, vy) > self.cfg["max_linear_mps"] + 1e-9 or abs(wz) > self.cfg["max_angular_rps"] + 1e-9:
                raise SafeStop("velocity limit")
            front = snap["clearance"](0.0)
            if front <= self.nav["coarse"]["lidar_stop_m"]:
                raise SafeStop("front LiDAR hard stop")
            if vx or vy:
                if snap["clearance"](math.atan2(vy, vx)) <= self.cfg["clearance_m"]:
                    raise SafeStop("motion direction obstructed")
            if wz:
                for direction in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                    if snap["clearance"](float(direction)) <= self.cfg["clearance_m"]:
                        raise SafeStop("rotation clearance obstructed")
        except (KeyError, ValueError, SafeStop) as exc:
            self.fault = str(exc)
            raise SafeStop(self.fault) from exc


def stopped(snap: dict, cfg: dict) -> bool:
    vx, vy, wz = snap["velocity"]
    return math.hypot(vx, vy) <= cfg["stopped_linear_mps"] and abs(wz) <= cfg["stopped_angular_rps"]
