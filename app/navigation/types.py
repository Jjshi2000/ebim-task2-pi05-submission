from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np


class SafeStop(RuntimeError):
    pass


class State(str, Enum):
    BOOT = "BOOT"
    WAIT_FOR_SENSORS = "WAIT_FOR_SENSORS"
    COARSE_APPROACH = "COARSE_APPROACH"
    VISUAL_SEARCH_INIT = "VISUAL_SEARCH_INIT"
    FINE_VISUAL_ALIGNMENT = "FINE_VISUAL_ALIGNMENT"
    FINAL_SETTLE_AND_VERIFY = "FINAL_SETTLE_AND_VERIFY"
    READY_FOR_PI05 = "READY_FOR_PI05"
    SAFE_STOP = "SAFE_STOP"


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def offset(self, dx: float, dy: float, dyaw: float = 0.0) -> Pose:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose(self.x + c * dx - s * dy, self.y + s * dx + c * dy,
                    wrap(self.yaw + dyaw))

    def error_body(self, target: Pose) -> tuple[float, float, float]:
        dx, dy = target.x - self.x, target.y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy, wrap(target.yaw - self.yaw)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Registration:
    reference_id: int
    valid: bool = False
    reason: str = "not evaluated"
    number_of_keypoints: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 1e6
    H: np.ndarray | None = None
    corners: np.ndarray | None = None
    geometry: np.ndarray | None = None
    overlap: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    rotation: float = 0.0
    spread: float = 0.0
    similarity: float = 0.0
    global_distance: float = 0.0


@dataclass
class Evaluation:
    score: float
    ready: bool
    reference_id: int
    valid_frames: int = 1
    registration: Registration | None = None
    details: dict = field(default_factory=dict)
