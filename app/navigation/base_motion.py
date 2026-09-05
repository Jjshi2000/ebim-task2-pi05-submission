from __future__ import annotations

import math

from .safety import SafetyMonitor, stopped
from .types import Pose, SafeStop, wrap


class BaseMotion:
    """I/O supplies snapshot, command, clock, and sleep; shared with offline simulation."""
    def __init__(self, io, nav: dict):
        self.io, self.nav = io, nav
        self.cfg = nav["safety"]
        self.safety = SafetyMonitor(nav)
        self.fine_origin: Pose | None = None
        self.fine_path_start = 0.0
        self.deadline = float("inf")

    def snapshot(self, command=(0.0, 0.0, 0.0)) -> dict:
        now = self.io.clock()
        if now > self.deadline:
            raise SafeStop("navigation stage timeout")
        snap = self.io.snapshot()
        self.safety.check(snap, now, command)
        if self.fine_origin is not None:
            p = snap["pose"]
            f = self.nav["fine"]
            if math.hypot(p.x - self.fine_origin.x, p.y - self.fine_origin.y) > f["max_radius_m"]:
                raise SafeStop("fine search radius exceeded")
            if abs(wrap(p.yaw - self.fine_origin.yaw)) > f["max_yaw_rad"]:
                raise SafeStop("fine search yaw envelope exceeded")
            if self.safety.path_m - self.fine_path_start > f["max_path_m"]:
                raise SafeStop("fine search cumulative travel exceeded")
        elif self.safety.path_m > self.nav["coarse"]["max_travel_m"]:
            raise SafeStop("coarse cumulative travel exceeded")
        return snap

    def command(self, vx: float, vy: float, wz: float = 0.0):
        self.snapshot((vx, vy, wz))
        self.io.command(vx, vy, wz)

    def stop(self):
        self.io.command(0.0, 0.0, 0.0)

    def final_stop(self):
        for _ in range(self.cfg["repeated_zero_cmd_count"]):
            try:
                self.stop()
            except Exception:
                pass
            try:
                self.io.sleep(1.0 / self.nav["rate_hz"])
            except Exception:
                pass

    def settle(self, seconds: float):
        end = self.io.clock() + seconds
        try:
            while self.io.clock() < end:
                self.stop()
                self.snapshot()
                self.io.sleep(1.0 / self.nav["rate_hz"])
            if not stopped(self.snapshot(), self.nav["handoff"]):
                raise SafeStop("base did not settle")
        finally:
            self.stop()

    def execute_base_delta(self, dx: float, dy: float, dyaw: float):
        start = self.snapshot()["pose"]
        self.move_to(start.offset(dx, dy, dyaw))

    def move_to(self, target: Pose):
        start = self.snapshot()["pose"]
        if self.fine_origin is not None:
            f = self.nav["fine"]
            if (math.hypot(target.x - self.fine_origin.x, target.y - self.fine_origin.y) > f["max_radius_m"] or
                    abs(wrap(target.yaw - self.fine_origin.yaw)) > f["max_yaw_rad"]):
                raise SafeStop("requested target exceeds fine search envelope")
        deadline = min(self.deadline, self.io.clock() + self.cfg["primitive_timeout_s"])
        delta = start.error_body(target)
        if (not all(math.isfinite(v) for v in delta) or
                math.hypot(delta[0], delta[1]) > 2 * max(self.nav["fine"]["step_x_m"], self.nav["fine"]["step_y_m"]) + self.cfg["tracking_error_m"] or
                abs(delta[2]) > 2 * self.nav["fine"]["step_yaw_rad"] + self.cfg["tracking_error_rad"]):
            raise SafeStop("primitive request exceeds micro-motion envelope")
        try:
            while self.io.clock() < deadline:
                p = self.snapshot()["pose"]
                ex, ey, eyaw = p.error_body(target)
                if math.hypot(ex, ey) <= self.cfg["position_tolerance_m"] and abs(eyaw) <= self.cfg["yaw_tolerance_rad"]:
                    return
                # Bound perpendicular drift away from the requested segment.
                sx, sy, syaw = start.error_body(p)
                length = math.hypot(delta[0], delta[1])
                cross = abs(sx * delta[1] - sy * delta[0]) / length if length > 1e-9 else math.hypot(sx, sy)
                if cross > self.cfg["tracking_error_m"] or abs(syaw) > abs(delta[2]) + self.cfg["tracking_error_rad"]:
                    raise SafeStop("primitive tracking error")
                speed = min(self.cfg["primitive_linear_mps"], math.hypot(ex, ey) * self.nav["rate_hz"] / 2)
                norm = max(math.hypot(ex, ey), 1e-12)
                wz = max(-self.cfg["max_angular_rps"], min(self.cfg["max_angular_rps"], eyaw * self.nav["rate_hz"] / 2))
                self.command(speed * ex / norm, speed * ey / norm, wz)
                self.io.sleep(1.0 / self.nav["rate_hz"])
            raise SafeStop("primitive timeout")
        finally:
            self.stop()
