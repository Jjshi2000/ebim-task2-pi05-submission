#!/usr/bin/env python3
"""Deterministic black-box pose search replay; no ROS or robot required."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app" if (root / "app").is_dir() else root))
import numpy as np
from navigation.config import load_config
from navigation.coordinator import Coordinator
from navigation.types import Evaluation, Pose, Registration


class SimIO:
    def __init__(self, pose=Pose(0, 0, 0), dry_run=False):
        self.pose, self.dry_run = pose, dry_run
        self.t, self.seq = 10.0, 1
        self.velocity = (0.0, 0.0, 0.0)
        self.commands = []
        self.fault = None
        self.overrides = {}

    def clock(self):
        return self.t

    def snapshot(self):
        return dict(pose=self.pose, velocity=self.velocity, odom_stamp=self.t,
                    camera_stamp=self.t, lidar_stamp=self.t, camera_seq=self.seq,
                    image=np.array([self.pose.x, self.pose.y, self.pose.yaw]),
                    clearance=lambda direction: 5.0, fault=self.fault, **self.overrides)

    def command(self, vx, vy, wz):
        if self.dry_run and any((vx, vy, wz)):
            raise AssertionError("dry run moved")
        self.velocity = (vx, vy, wz)
        self.commands.append(self.velocity)

    def sleep(self, seconds):
        self.pose = self.pose.offset(*(v * seconds for v in self.velocity))
        self.t += seconds
        self.seq += 1


class SyntheticBank:
    def __init__(self, optimum=(0, 0, 0), noise=0, seed=1):
        self.optimum = np.asarray(optimum)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def evaluate(self, image, reference_id=None):
        error = (image - self.optimum) / np.array([0.02, 0.02, 0.03])
        distance = float(np.linalg.norm(error))
        return Evaluation(-distance + self.rng.normal(0, self.noise), distance < 0.35,
                          0, registration=Registration(0, valid=True, reason="synthetic"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/task2.real.yaml")
    ap.add_argument("--output", default="out/search_replay.json")
    args = ap.parse_args()
    nav = load_config(args.config)["navigation"]
    rows = []
    for x, y, yaw in ((.06, -.04, 0), (-.07, .05, 0), (.02, .07, 0), (-.04, -.06, .06)):
        for noise in (0.0, 0.015):
            nav["fine"]["enable_yaw"] = yaw != 0
            io = SimIO(Pose(x, y, yaw))
            events = []
            coordinator = Coordinator(io, SyntheticBank(noise=noise), nav,
                                      lambda event, **fields: events.append(dict(event=event, **fields)))
            ready = coordinator.run()
            rows.append(dict(initial=[x, y, yaw], noise=noise, ready=ready,
                             final=[io.pose.x, io.pose.y, io.pose.yaw], duration_s=io.t - 10,
                             travel_m=coordinator.motion.safety.path_m,
                             probes=sum(e["event"] == "probe" for e in events)))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
