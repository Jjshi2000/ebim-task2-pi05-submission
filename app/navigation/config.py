from __future__ import annotations

import math
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")

    def check(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                check(child, f"{key}.{name}")
        elif isinstance(value, list):
            for child in value:
                check(child, key)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"non-finite configuration: {key}")
    check(cfg)
    nav = cfg["navigation"]
    fine, safety, visual = nav["fine"], nav["safety"], nav["visual"]
    for section in (fine, safety, nav["coarse"], nav["handoff"], visual):
        for key, value in section.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
                raise ValueError(f"{key} must be positive")
    if not 0 < fine["step_shrink_factor"] < 1:
        raise ValueError("step_shrink_factor must be between 0 and 1")
    if not 0 < visual["ratio_test"] < 1 or not 0 < visual["acceptance_quantile"] < 1:
        raise ValueError("invalid visual quantile or ratio test")
    if not 1 <= fine["min_valid_frames"] <= fine["frames_per_evaluation"]:
        raise ValueError("invalid evaluation frame counts")
    for axis, unit in (("x", "m"), ("y", "m"), ("yaw", "rad")):
        if fine[f"min_step_{axis}_{unit}"] > fine[f"step_{axis}_{unit}"]:
            raise ValueError("minimum step exceeds initial step")
    if nav["coarse"]["lidar_stop_m"] >= nav["coarse"]["lidar_slow_m"]:
        raise ValueError("LiDAR stop must be smaller than slowdown distance")
    if nav["coarse"]["cruise_vx"] > safety["max_linear_mps"]:
        raise ValueError("coarse speed exceeds safety limit")
    if cfg["policy"]["rate_hz"] != 20.0:
        raise ValueError("real checkpoint requires the demonstrated 20 Hz cadence")
    if cfg["policy"]["replan_steps"] != 10 or cfg["rig"]["spine_target_mm"] != 434.0:
        raise ValueError("real checkpoint requires replan 10 and spine 434 mm")
    if not 0 < cfg["policy"]["min_playback_rate"] <= 1 or cfg["policy"]["max_episode_s"] < 0:
        raise ValueError("invalid policy playback or duration")
    if cfg["policy"]["latency_margin"] < 1:
        raise ValueError("policy latency margin must cover at least two inference periods")
    for section, names in ((cfg["rig"], ("stream_heartbeat_s", "stream_start_timeout_s")),
                           (cfg["policy"], ("max_inference_s", "max_hold_s", "request_timeout_s"))):
        if any(section[n] <= 0 for n in names):
            raise ValueError("stream and policy deadlines must be positive")
    for frame, extrinsic in safety["lidar_extrinsics"].items():
        if not frame or not isinstance(extrinsic, list) or len(extrinsic) != 3 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in extrinsic):
            raise ValueError("LiDAR extrinsics require a frame and measured [x, y, yaw]")
    for name in ("joint_lower_rad", "joint_upper_rad"):
        if len(cfg["rig"][name]) != 7:
            raise ValueError("FR3 joint limits must contain seven values")
    if any(lo >= hi for lo, hi in zip(cfg["rig"]["joint_lower_rad"], cfg["rig"]["joint_upper_rad"])):
        raise ValueError("invalid FR3 joint limit interval")
    if safety["primitive_linear_mps"] > safety["max_linear_mps"] or nav["coarse"]["slow_vx"] > nav["coarse"]["cruise_vx"]:
        raise ValueError("motion speed exceeds safety limit")
    for owner, name in ((visual, "bank_path"), (cfg["log"], "directory")):
        value = Path(owner[name])
        owner[name] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return cfg
