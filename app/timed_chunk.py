"""Wall-clock action playback, independent of inference request duration."""
from __future__ import annotations

import math
import numpy as np

from navigation.types import SafeStop


class TimedChunk:
    def __init__(self, cfg):
        self.cfg = cfg
        self.actions = None
        self.origin = 0.0
        self.speed = 1.0
        self.refresh_at = 0.0

    def install(self, actions, observed_at, now):
        actions = np.asarray(actions, dtype=np.float32)
        latency = now - observed_at
        if actions.shape != (50, 17) or not np.isfinite(actions).all():
            raise SafeStop("expected a finite 50x17 action chunk")
        if not math.isfinite(latency) or not 0 <= latency <= self.cfg["max_inference_s"]:
            raise SafeStop("inference latency outside configured limit")
        # Cover observation-to-result plus the next inference, as in HKUST's pacing.
        ticks = latency * self.cfg["rate_hz"]
        speed = min(1.0, (len(actions) - 1) / max(1.0, 2 * ticks * self.cfg["latency_margin"]))
        if speed < self.cfg["min_playback_rate"]:
            raise SafeStop("inference cannot sustain minimum action playback rate")
        self.actions, self.origin, self.speed = actions.copy(), observed_at, speed
        remaining = (len(actions) - 1) / (speed * self.cfg["rate_hz"]) - latency
        execute_s = min(self.cfg["replan_steps"] / (speed * self.cfg["rate_hz"]),
                        max(0, remaining - 1 / self.cfg["rate_hz"]))
        self.refresh_at = now + max(0, execute_s - latency * self.cfg["latency_margin"])

    def index(self, now):
        return max(0, int((now - self.origin) * self.cfg["rate_hz"] * self.speed))

    def action(self, now):
        if self.actions is None:
            raise SafeStop("no action chunk")
        index = self.index(now)
        if index >= len(self.actions):
            end = self.origin + len(self.actions) / (self.cfg["rate_hz"] * self.speed)
            if now - end > self.cfg["max_hold_s"]:
                raise SafeStop("action chunk exhausted beyond hold deadline")
            return self.actions[-1].copy()
        return self.actions[index].copy()

    def replan_due(self, now):
        return now >= self.refresh_at
