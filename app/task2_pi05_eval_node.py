#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import socket
import struct
import threading
import time


HOST = "127.0.0.1"
PORT = 8765

DEFAULT_TASK = "Pick up the thermal pad and place it on the target RAM board."
DEFAULT_ROBOT_TYPE = "fr3duo_mobile_task2"
GRIPPER_CLOSED_RAD = 0.8

ISAAC_STATE_DIM = 37
ISAAC_ACTION_DIM = 20
REAL_STATE_DIM = 42
REAL_ACTION_DIM = 17

LEFT_JOINTS = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
REAL_ARM_COMMAND_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"

REAL_ROBOT = False
CLOCK_TOPIC = "/isaac/clock"
FULL_STATES_TOPIC = "/isaac/joint_states_full"
ODOM_TOPIC = "/isaac/odom"
LEFT_EE_TOPIC = "/isaac/left_ee_pose"
RIGHT_EE_TOPIC = "/isaac/right_ee_pose"
SCENE_RESET_TOPIC = "/isaac/task2/scene_reset"
SCENE_RESET_REQUEST_TOPIC = "/isaac/task2/scene_reset_request"
PAD_POINTS_TOPIC = "/isaac/task2/pad_points"

LEFT_ARM_CMD_TOPIC = "/isaac/left_joint_commands"
RIGHT_ARM_CMD_TOPIC = "/isaac/right_joint_commands"
LEFT_GRIPPER_CMD_TOPIC = "/isaac/left_robotiq_joint_commands"
RIGHT_GRIPPER_CMD_TOPIC = "/isaac/right_robotiq_joint_commands"
SPINE_CMD_TOPIC = "/isaac/spine_joint_commands"

REAL_TOPICS = {
    "left_joints": "/left/franka_robot_state_broadcaster/measured_joint_states",
    "right_joints": "/right/franka_robot_state_broadcaster/measured_joint_states",
    "left_external_joint_torques": "/left/franka_robot_state_broadcaster/external_joint_torques",
    "right_external_joint_torques": "/right/franka_robot_state_broadcaster/external_joint_torques",
    "left_external_wrench": "/left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
    "right_external_wrench": "/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
    "spine_joints": "/spine/joint_states",
    "left_gripper": "/left/gripper/joint_states",
    "right_gripper": "/right/gripper/joint_states",
    "odom": "/swerve_drive_controller/odom",
    "left_camera": "/head_camera/zed_node/rgb/color/rect/image",
    "wrist_left": "/wrist_camera_left/camera/color/image_raw",
    "wrist_right": "/wrist_camera_right/camera/color/image_raw",
    "left_gripper_cmd": "/left/gripper/gripper_client/target_gripper_width_percent",
    "right_gripper_cmd": "/right/gripper/gripper_client/target_gripper_width_percent",
    "left_arm_cmd": "/left/gello/joint_states",
    "right_arm_cmd": "/right/gello/joint_states",
    "spine_cmd": "/spine/target_height",
}


# The current Task2 v3 dataset can contain four RGB streams.  A checkpoint
# selects a subset through PI05Config.input_features.  The ROS adapter discovers
# the matching live sensor_msgs/msg/Image topics from the ROS graph rather than
# assuming that the feature key itself is a ROS topic name.
CAMERA_LOGICAL_ALIASES = {
    "head": (
        "head_camera",
        "head",
    ),
    "eval_camera": (
        "eval_camera",
        "evaluation_camera",
        "eval",
    ),
    "wrist_left": (
        "wrist_left",
        "left_wrist",
        "left_wrist_camera",
        "wrist_camera_left",
    ),
    "wrist_right": (
        "wrist_right",
        "right_wrist",
        "right_wrist_camera",
        "wrist_camera_right",
    ),
}

# Preferred exact names for the Task2 Isaac Sim bridge.  Discovery still works
# when the benchmark publishes an alias instead.
PREFERRED_CAMERA_TOPICS = {
    "head": (
        "/isaac/head_camera/image_raw",
        "/head_camera/zed_node/rgb/color/rect/image",
    ),
    "eval_camera": (
        "/isaac/eval_camera/image_raw",
        "/isaac/task2/eval_camera/image_raw",
        "/isaac/evaluation_camera/image_raw",
    ),
    "wrist_left": (
        "/isaac/wrist_left_camera/image_raw",
        "/isaac/left_wrist_camera/image_raw",
        "/isaac/wrist_left/image_raw",
        "/isaac/left_wrist/image_raw",
        "/wrist_camera_left/camera/color/image_raw",
    ),
    "wrist_right": (
        "/isaac/wrist_right_camera/image_raw",
        "/isaac/right_wrist_camera/image_raw",
        "/isaac/wrist_right/image_raw",
        "/isaac/right_wrist/image_raw",
        "/wrist_camera_right/camera/color/image_raw",
    ),
}

BAD_CAMERA_TOPIC_TOKENS = (
    "depth",
    "seg",
    "semantic",
    "instance",
    "bbox",
    "bounding",
    "mask",
    "label",
    "camera_info",
    "pointcloud",
    "points",
)


# Successful Task2 demonstrations (200 episodes, 30 Hz) define the deployment
# envelope. The arm bounds include a small margin beyond the observed extrema.
# Slew limits are the rounded P99.5 within-episode action deltas, preventing a
# stochastic chunk boundary from becoming an instantaneous joint-target jump.
DEMO_LEFT_ARM_MIN = (-1.242, -1.435, -0.030, -2.734, -0.030, 1.402, 0.713)
DEMO_LEFT_ARM_MAX = (0.076, -0.203, 1.775, -1.362, 1.306, 2.535, 1.841)
DEMO_RIGHT_ARM_MIN = (-0.607, -1.857, -2.769, -3.107, -2.117, 1.540, -2.034)
DEMO_RIGHT_ARM_MAX = (2.876, 0.087, 0.081, -0.101, 1.943, 4.641, 0.815)

DEMO_LEFT_ARM_MAX_STEP = (0.010, 0.012, 0.016, 0.010, 0.010, 0.012, 0.010)
DEMO_RIGHT_ARM_MAX_STEP = (0.050, 0.060, 0.090, 0.045, 0.065, 0.070, 0.060)
DEMO_SPINE_MAX_STEP = 0.020

# The public Munich real-robot dataset has 238 episodes at 20 Hz. These are the
# observed action extrema and rounded P99.5 within-episode arm target deltas.
# Gripper targets are independently clamped to [0, 1]. The recorded spine
# target is exactly 434.0 for every frame.
REAL_DEMO_LEFT_ARM_MIN = (0.701, 0.482, -2.505, -2.346, 0.924, 0.718, -1.968)
REAL_DEMO_LEFT_ARM_MAX = (2.317, 0.696, -1.108, -1.339, 2.377, 2.131, -0.029)
REAL_DEMO_RIGHT_ARM_MIN = (0.382, -1.387, -2.901, -2.696, -2.877, 0.594, -1.315)
REAL_DEMO_RIGHT_ARM_MAX = (2.694, -0.450, -0.476, -1.078, 0.173, 3.664, 2.912)
REAL_DEMO_LEFT_ARM_MAX_STEP = (0.020, 0.002, 0.017, 0.010, 0.016, 0.017, 0.019)
REAL_DEMO_RIGHT_ARM_MAX_STEP = (0.044, 0.019, 0.050, 0.025, 0.059, 0.066, 0.070)
REAL_DEMO_SPINE_HEIGHT = 434.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class Task2ActionSafetyFilter:
    """Project absolute policy actions into the demonstrated control envelope."""

    def __init__(
        self,
        *,
        spine_min: float,
        spine_max: float,
        step_scale: float = 1.0,
        contract: str = "isaac37_20",
    ):
        if step_scale <= 0:
            raise ValueError("step_scale must be > 0")
        self.spine_min = float(spine_min)
        self.spine_max = float(spine_max)
        self.step_scale = float(step_scale)
        if contract not in ("isaac37_20", "real42_17"):
            raise ValueError(f"unsupported Task2 action contract: {contract}")
        self.contract = contract
        self.last_action = None
        self.total_limited_steps = 0

    def reset(self) -> None:
        self.last_action = None

    def _measured_action(self, joints: dict[str, float]) -> list[float]:
        required = LEFT_JOINTS + RIGHT_JOINTS
        if self.contract == "real42_17":
            required += [RIGHT_GRIPPER_DRIVER]
        else:
            required += [LEFT_GRIPPER_DRIVER, RIGHT_GRIPPER_DRIVER, SPINE_JOINT]
        missing = [name for name in required if name not in joints]
        if missing:
            raise RuntimeError(
                "cannot initialize action safety without measured joints: "
                + ", ".join(missing)
            )

        if self.contract == "real42_17":
            return (
                [float(joints[name]) for name in LEFT_JOINTS]
                + [1.0]
                + [float(joints[name]) for name in RIGHT_JOINTS]
                + [_clip(float(joints[RIGHT_GRIPPER_DRIVER]), 0.0, 1.0)]
                + [REAL_DEMO_SPINE_HEIGHT]
            )

        left_open = _clip(1.0 - float(joints[LEFT_GRIPPER_DRIVER]) / GRIPPER_CLOSED_RAD, 0.0, 1.0)
        right_open = _clip(1.0 - float(joints[RIGHT_GRIPPER_DRIVER]) / GRIPPER_CLOSED_RAD, 0.0, 1.0)
        return (
            [0.0, 0.0, 0.0]
            + [float(joints[name]) for name in LEFT_JOINTS]
            + [float(joints[name]) for name in RIGHT_JOINTS]
            + [left_open, right_open, float(joints[SPINE_JOINT])]
        )

    def filter(
        self,
        action: list[float],
        measured_joints: dict[str, float],
    ) -> tuple[list[float], dict[str, object]]:
        expected_dim = REAL_ACTION_DIM if self.contract == "real42_17" else ISAAC_ACTION_DIM
        if len(action) != expected_dim or not all(math.isfinite(float(x)) for x in action):
            raise ValueError(f"invalid {expected_dim}D action")
        if self.last_action is None:
            self.last_action = self._measured_action(measured_joints)

        raw = [float(x) for x in action]
        bounded = list(raw)
        if self.contract == "real42_17":
            arm_indices = list(range(0, 7)) + list(range(8, 15))
            arm_bounds = list(zip(REAL_DEMO_LEFT_ARM_MIN, REAL_DEMO_LEFT_ARM_MAX)) + list(
                zip(REAL_DEMO_RIGHT_ARM_MIN, REAL_DEMO_RIGHT_ARM_MAX)
            )
            gripper_indices = (7, 15)
            spine_index = 16
            max_steps = list(REAL_DEMO_LEFT_ARM_MAX_STEP) + list(REAL_DEMO_RIGHT_ARM_MAX_STEP)
            slew_indices = arm_indices
        else:
            bounded[0:3] = [0.0, 0.0, 0.0]
            arm_indices = list(range(3, 17))
            arm_bounds = list(zip(DEMO_LEFT_ARM_MIN, DEMO_LEFT_ARM_MAX)) + list(
                zip(DEMO_RIGHT_ARM_MIN, DEMO_RIGHT_ARM_MAX)
            )
            gripper_indices = (17, 18)
            spine_index = 19
            max_steps = (
                list(DEMO_LEFT_ARM_MAX_STEP)
                + list(DEMO_RIGHT_ARM_MAX_STEP)
                + [DEMO_SPINE_MAX_STEP]
            )
            slew_indices = arm_indices + [spine_index]

        range_limited = []
        for index, (lower, upper) in zip(arm_indices, arm_bounds):
            bounded[index] = _clip(raw[index], lower, upper)
            if not math.isclose(bounded[index], raw[index], abs_tol=1.0e-12):
                range_limited.append(index)

        for index in gripper_indices:
            bounded[index] = _clip(raw[index], 0.0, 1.0)
        bounded[spine_index] = _clip(raw[spine_index], self.spine_min, self.spine_max)
        for index in (*gripper_indices, spine_index):
            if not math.isclose(bounded[index], raw[index], abs_tol=1.0e-12):
                range_limited.append(index)

        safe = list(bounded)
        rate_limited = []
        for index, max_step in zip(slew_indices, max_steps):
            previous = float(self.last_action[index])
            limit = float(max_step) * self.step_scale
            safe[index] = _clip(bounded[index], previous - limit, previous + limit)
            if not math.isclose(safe[index], bounded[index], abs_tol=1.0e-12):
                rate_limited.append(index)

        raw_max_arm_delta = max(
            abs(raw[index] - float(self.last_action[index]))
            for index in arm_indices
        )
        safe_max_arm_delta = max(
            abs(safe[index] - float(self.last_action[index]))
            for index in arm_indices
        )
        self.last_action = safe
        limited = bool(range_limited or rate_limited)
        if limited:
            self.total_limited_steps += 1

        return safe, {
            "limited": limited,
            "range_limited": range_limited,
            "rate_limited": rate_limited,
            "raw_max_arm_delta": raw_max_arm_delta,
            "safe_max_arm_delta": safe_max_arm_delta,
            "total_limited_steps": self.total_limited_steps,
        }


# ---------------------------------------------------------------------------
# Framed localhost protocol
# ---------------------------------------------------------------------------

def _recv_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("socket closed")
        out.extend(chunk)
    return bytes(out)


def send_obj(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def recv_obj(sock):
    size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    if size > 512 * 1024 * 1024:
        raise ValueError(f"refusing oversized packet: {size} bytes")
    return pickle.loads(_recv_exact(sock, size))


# ---------------------------------------------------------------------------
# Camera feature / checkpoint helpers
# ---------------------------------------------------------------------------

def camera_logical_from_feature(feature_key: str) -> str:
    suffix = feature_key.rsplit(".", 1)[-1].lower()

    if suffix == "head":
        return "head"
    if suffix in ("eval", "eval_camera", "evaluation_camera"):
        return "eval_camera"
    if suffix in ("wrist_left", "left_wrist", "left_wrist_camera"):
        return "wrist_left"
    if suffix in ("wrist_right", "right_wrist", "right_wrist_camera"):
        return "wrist_right"

    raise ValueError(
        f"Unsupported Task2 image feature {feature_key!r}. "
        "Supported camera suffixes: head, eval_camera, wrist_left, wrist_right."
    )


def _feature_shape(feature) -> tuple[int, ...] | None:
    shape = getattr(feature, "shape", None)
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def policy_image_keys_from_config(config) -> list[str]:
    features = getattr(config, "image_features", {}) or {}
    keys = list(features.keys()) if hasattr(features, "keys") else list(features)
    if not keys:
        raise RuntimeError("PI0.5 checkpoint config has no visual input features")
    for key in keys:
        camera_logical_from_feature(key)
    return keys


def policy_expected_image_shapes_from_config(config, image_keys: list[str]) -> dict[str, tuple[int, ...] | None]:
    input_features = getattr(config, "input_features", {}) or {}
    image_features = getattr(config, "image_features", {}) or {}
    out = {}
    for key in image_keys:
        feat = input_features.get(key) if hasattr(input_features, "get") else None
        if feat is None and hasattr(image_features, "get"):
            feat = image_features.get(key)
        out[key] = _feature_shape(feat)
    return out


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pi05_action_queue_len(policy) -> int | None:
    """Best-effort introspection for diagnostics only; never changes policy behavior."""
    candidates = [policy]
    get_base_model = getattr(policy, "get_base_model", None)
    if callable(get_base_model):
        try:
            candidates.append(get_base_model())
        except Exception:
            pass
    for obj in candidates:
        queue = getattr(obj, "_action_queue", None)
        if queue is not None:
            try:
                return len(queue)
            except TypeError:
                pass
    return None


def _reset_pipeline(policy, preprocessor, postprocessor, *, seed: int | None = None) -> None:
    """Reset every stateful part of the synchronous LeRobot inference path."""
    if seed is not None:
        _seed_everything(seed)

    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    else:
        get_base_model = getattr(policy, "get_base_model", None)
        if not callable(get_base_model):
            raise RuntimeError("Loaded PI0.5 policy exposes neither reset() nor get_base_model()")
        base = get_base_model()
        reset = getattr(base, "reset", None)
        if not callable(reset):
            raise RuntimeError("Loaded PEFT base policy does not expose reset()")
        reset()

    for name, processor in (("preprocessor", preprocessor), ("postprocessor", postprocessor)):
        reset = getattr(processor, "reset", None)
        if callable(reset):
            reset()
        else:
            raise RuntimeError(f"Loaded {name} has no reset(); checkpoint processor format is incompatible")


def trim_replanned_chunk(actions, actions_consumed_during_inference: int):
    """Drop chunk steps that became obsolete while background inference ran."""
    delay = max(0, min(int(actions_consumed_during_inference), len(actions)))
    return actions[delay:], delay


class AsyncActionChunkExecutor:
    """Run PI0.5 chunk inference off the socket/control thread.

    This is deliberately non-guided RTC for the absolute-action Task2
    checkpoint. A completed plan atomically replaces the old tail after the
    prefix corresponding to actions actually consumed during inference is
    removed.
    """

    def __init__(
        self,
        *,
        policy,
        config,
        preprocessor,
        postprocessor,
        device,
        task,
        robot_type,
        replan_interval,
        prefetch_margin_steps=0,
        debug_actions=False,
        action_ensemble_samples=1,
        action_dim=ISAAC_ACTION_DIM,
    ):
        self.policy = policy
        self.config = config
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.task = task
        self.robot_type = robot_type
        self.replan_interval = int(replan_interval)
        self.prefetch_margin_steps = max(
            0,
            min(int(prefetch_margin_steps), max(0, self.replan_interval - 1)),
        )
        self.replan_trigger_steps = max(
            1, self.replan_interval - self.prefetch_margin_steps
        )
        self.debug_actions = bool(debug_actions)
        self.action_ensemble_samples = max(1, int(action_ensemble_samples))
        self.action_dim = int(action_dim)

        self.cond = threading.Condition()
        self.actions = deque()
        self.last_action = None
        self.latest_observation = None
        self.inference_requested = False
        self.inference_busy = False
        self.stop_requested = False
        self.epoch = 0
        self.control_steps_served = 0
        self.steps_since_replan = 0
        self.hold_count = 0
        self.chunk_id = 0
        self.last_result = {
            "chunk_id": 0,
            "inference_ms": 0.0,
            "inference_delay_steps": 0,
            "observation_sim_time": None,
            "replacement_wall_time": None,
        }
        self.error = None
        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PI05AsyncInference",
        )
        self.worker.start()

    def reset(self, reset_fn):
        # Reset processors/model only after an in-flight CUDA call has exited.
        with self.cond:
            self.epoch += 1
            self.latest_observation = None
            self.inference_requested = False
            self.actions.clear()
            self.last_action = None
            self.control_steps_served = 0
            self.steps_since_replan = 0
            self.hold_count = 0
            self.error = None
            while self.inference_busy:
                self.cond.wait(timeout=0.1)
        reset_fn()
        with self.cond:
            self.last_result = {
                "chunk_id": self.chunk_id,
                "inference_ms": 0.0,
                "inference_delay_steps": 0,
                "observation_sim_time": None,
                "replacement_wall_time": None,
            }

    def close(self):
        with self.cond:
            self.stop_requested = True
            self.cond.notify_all()
        self.worker.join(timeout=3.0)

    def request_inference(self, raw_observation, observation_sim_time=None):
        with self.cond:
            if self.error is not None:
                raise RuntimeError(self.error)
            if self.inference_busy or self.inference_requested:
                return False
            if (
                self.last_action is not None
                and self.steps_since_replan < self.replan_trigger_steps
            ):
                return False
            self.latest_observation = (
                raw_observation,
                observation_sim_time,
                self.control_steps_served,
                self.last_action is not None,
            )
            self.inference_requested = True
            self.cond.notify_all()
            return True

    def get_action(self):
        with self.cond:
            if self.error is not None:
                raise RuntimeError(self.error)

            held = False
            if self.actions:
                action = self.actions.popleft()
                self.last_action = action
            else:
                action = self.last_action
                held = action is not None
                if held:
                    self.hold_count += 1
            self.control_steps_served += 1
            if action is not None:
                self.steps_since_replan += 1

            result = dict(self.last_result)
            result.update(
                {
                    "action": None if action is None else action.tolist(),
                    "queue_remaining": len(self.actions),
                    "inference_busy": self.inference_busy or self.inference_requested,
                    "replan_due": self.last_action is None
                    or self.steps_since_replan >= self.replan_trigger_steps,
                    "steps_since_replan": self.steps_since_replan,
                    "held": held,
                    "hold_count": self.hold_count,
                }
            )
            return result

    def _worker_loop(self):
        import numpy as np
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        while True:
            with self.cond:
                while not self.stop_requested and not self.inference_requested:
                    self.cond.wait()
                if self.stop_requested:
                    return
                (
                    raw_observation,
                    observation_sim_time,
                    consumed_before,
                    had_action_at_request,
                ) = self.latest_observation
                request_epoch = self.epoch
                self.inference_requested = False
                self.inference_busy = True

            t0 = time.perf_counter()
            try:
                autocast_ctx = (
                    torch.autocast(device_type=self.device.type)
                    if self.device.type == "cuda" and bool(getattr(self.config, "use_amp", False))
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    observation = prepare_observation_for_inference(
                        raw_observation,
                        device=self.device,
                        task=self.task,
                        robot_type=self.robot_type,
                    )
                    observation = self.preprocessor(observation)
                    chunks = []
                    for _ in range(self.action_ensemble_samples):
                        sampled = self.policy.predict_action_chunk(observation)
                        sampled = self.postprocessor(sampled)
                        if not torch.is_tensor(sampled) or sampled.ndim != 3:
                            raise TypeError(
                                "PI0.5 chunk postprocessor must return Tensor [B,T,A], "
                                f"got {type(sampled).__name__} shape={getattr(sampled, 'shape', None)}"
                            )
                        if sampled.shape[0] != 1 or sampled.shape[2] != self.action_dim:
                            raise ValueError(
                                f"expected postprocessed Task2 chunk [1,T,{self.action_dim}], "
                                f"got {tuple(sampled.shape)}"
                            )
                        chunks.append(sampled)
                    actions = torch.stack(chunks, dim=0).mean(dim=0)

                if not torch.is_tensor(actions) or actions.ndim != 3:
                    raise TypeError(
                        "PI0.5 chunk postprocessor must return Tensor [B,T,A], "
                        f"got {type(actions).__name__} shape={getattr(actions, 'shape', None)}"
                    )
                if actions.shape[0] != 1 or actions.shape[2] != self.action_dim:
                    raise ValueError(
                        f"expected postprocessed Task2 chunk [1,T,{self.action_dim}], "
                        f"got {tuple(actions.shape)}"
                    )
                chunk = actions[0].detach().float().cpu().numpy()
                if not np.isfinite(chunk).all():
                    raise ValueError("non-finite PI0.5 action chunk")
                inference_ms = (time.perf_counter() - t0) * 1000.0

                with self.cond:
                    if request_epoch == self.epoch:
                        control_steps_during = (
                            self.control_steps_served - consumed_before
                            if had_action_at_request
                            else 0
                        )
                        usable, delay = trim_replanned_chunk(chunk, control_steps_during)
                        self.actions = deque(usable)
                        # The replacement plan is conditioned on the snapshot
                        # taken before inference.  Actions executed while CUDA
                        # was busy still count toward the next observation
                        # boundary; clearing this counter stretches a 10-step
                        # checkpoint horizon to roughly 13 steps indefinitely.
                        self.steps_since_replan = delay
                        self.chunk_id += 1
                        self.last_result = {
                            "chunk_id": self.chunk_id,
                            "inference_ms": inference_ms,
                            "inference_delay_steps": delay,
                            "observation_sim_time": observation_sim_time,
                            "replacement_wall_time": time.time(),
                        }
                        if self.debug_actions and len(usable):
                            print(
                                f"ASYNC CHUNK {self.chunk_id} delay={delay} action0="
                                + np.array2string(usable[0], precision=4, suppress_small=False),
                                flush=True,
                            )
                    self.inference_busy = False
                    self.cond.notify_all()
            except Exception as exc:
                with self.cond:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.inference_busy = False
                    self.cond.notify_all()


def load_pi05_policy_and_processors(checkpoint: str, device: str):
    """Load a LeRobot PI0.5 checkpoint, including PEFT adapter and saved processors.

    The policy config stored next to a LeRobot PEFT checkpoint records the original
    PI0.5 feature contract.  The PEFT adapter_config.json records the base model.
    The policy_preprocessor / policy_postprocessor files carry the exact Task2
    quantile normalization statistics used during training.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    ckpt = Path(checkpoint).expanduser().resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"PI0.5 policy config missing: {ckpt / 'config.json'}")

    config = PreTrainedConfig.from_pretrained(str(ckpt))
    if getattr(config, "type", None) != "pi05":
        raise ValueError(f"checkpoint policy type is {getattr(config, 'type', None)!r}, expected 'pi05'")

    # Inference-time overrides only.  They do not alter the learned feature,
    # normalization, action-chunk, or LoRA configuration.
    config.device = device
    if hasattr(config, "compile_model"):
        config.compile_model = False
    if hasattr(config, "gradient_checkpointing"):
        config.gradient_checkpointing = False
    if bool(getattr(config, "use_relative_actions", False)):
        raise RuntimeError(
            "This Task2 synchronous node intentionally rejects PI0.5 relative-action checkpoints. "
            "Your current Task2 LoRA training uses absolute actions; a relative-action checkpoint needs "
            "chunk-level postprocessing/RTC to avoid per-tick re-anchoring drift."
        )
    if getattr(config, "rtc_config", None) is not None and getattr(config.rtc_config, "enabled", False):
        raise RuntimeError(
            "This Task2 node implements LeRobot synchronous select_action inference. "
            "The checkpoint has RTC enabled; disable RTC in the checkpoint/config or use a dedicated RTC executor."
        )

    adapter_cfg_path = ckpt / "adapter_config.json"
    adapter_weights_path = ckpt / "adapter_model.safetensors"
    is_peft = bool(getattr(config, "use_peft", False) or adapter_cfg_path.is_file())
    config.use_peft = is_peft
    config.pretrained_path = ckpt

    if is_peft:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "PEFT is required for this LoRA checkpoint. Install the LeRobot peft extra in the lerobot env."
            ) from exc

        if not adapter_cfg_path.is_file():
            raise FileNotFoundError(f"LoRA adapter config missing: {adapter_cfg_path}")
        if not adapter_weights_path.is_file():
            raise FileNotFoundError(f"LoRA adapter weights missing: {adapter_weights_path}")

        peft_config = PeftConfig.from_pretrained(str(ckpt))
        base_path = getattr(peft_config, "base_model_name_or_path", None)
        if not base_path:
            raise RuntimeError("adapter_config.json has no base_model_name_or_path")

        print(f"Loading PI0.5 base policy: {base_path}")
        base_policy = PI05Policy.from_pretrained(
            pretrained_name_or_path=base_path,
            config=config,
            revision=getattr(peft_config, "revision", None),
        )
        policy = PeftModel.from_pretrained(
            base_policy,
            str(ckpt),
            config=peft_config,
            is_trainable=False,
        )
    else:
        model_path = ckpt / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is not marked PEFT and has no full model weights: {model_path}"
            )
        policy = PI05Policy.from_pretrained(str(ckpt), config=config)

    policy.to(torch.device(device))
    policy.eval()

    # Never rebuild PI0.5 normalization from ad-hoc stats at deployment time.
    # LeRobot checkpoints save the exact processor pipelines next to the model.
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(ckpt),
            preprocessor_overrides={
                "device_processor": {"device": str(torch.device(device))},
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the saved PI0.5 pre/post processors from the checkpoint. "
            "Do not bypass this: Task2 PI0.5 uses QUANTILES for state/action and language tokenization. "
            f"checkpoint={ckpt}"
        ) from exc

    image_keys = policy_image_keys_from_config(config)
    expected_shapes = policy_expected_image_shapes_from_config(config, image_keys)

    state_feature = getattr(config, "robot_state_feature", None)
    state_shape = _feature_shape(state_feature)
    action_feature = getattr(config, "action_feature", None)
    action_shape = _feature_shape(action_feature)
    supported_contracts = {
        ((ISAAC_STATE_DIM,), (ISAAC_ACTION_DIM,)),
        ((REAL_STATE_DIM,), (REAL_ACTION_DIM,)),
    }
    if (state_shape, action_shape) not in supported_contracts:
        raise ValueError(
            "Unsupported Task2 checkpoint contract: "
            f"state={state_shape}, action={action_shape}; expected "
            f"({ISAAC_STATE_DIM},{ISAAC_ACTION_DIM}) for Isaac or "
            f"({REAL_STATE_DIM},{REAL_ACTION_DIM}) for the official real dataset"
        )

    return {
        "policy": policy,
        "config": config,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": torch.device(device),
        "image_keys": image_keys,
        "expected_shapes": expected_shapes,
        "state_shape": state_shape,
        "action_shape": action_shape,
        "is_peft": is_peft,
        "checkpoint": ckpt,
    }


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def decode_image(pkt):
    import numpy as np

    h = int(pkt["height"])
    w = int(pkt["width"])
    step = int(pkt["step"])
    enc = str(pkt["encoding"]).lower()

    if h <= 0 or w <= 0 or step <= 0:
        raise ValueError(f"invalid image geometry h={h} w={w} step={step}")

    if enc in ("rgb8", "bgr8", "8uc3"):
        ch = 3
    elif enc in ("rgba8", "bgra8", "8uc4"):
        ch = 4
    else:
        raise ValueError(
            f"Unsupported ROS image encoding {enc!r}; "
            "expected rgb8/bgr8/rgba8/bgra8/8UC3/8UC4"
        )

    raw = np.frombuffer(pkt["data"], dtype=np.uint8)
    expected_bytes = h * step
    if raw.size != expected_bytes:
        raise ValueError(
            f"image byte count mismatch: got {raw.size}, expected {expected_bytes}"
        )

    row = raw.reshape(h, step)
    needed = w * ch
    if needed > step:
        raise ValueError(
            f"image row step {step} is smaller than width*channels {needed}"
        )

    img = row[:, :needed].reshape(h, w, ch)[:, :, :3]

    # 8UC3 is treated as RGB. Isaac's normal bridge should publish rgb8/bgr8,
    # so the ambiguous encoding is only a compatibility fallback.
    if enc in ("bgr8", "bgra8"):
        img = img[:, :, ::-1]

    # np.frombuffer(bytes) is read-only; force an owned writable array before torch.from_numpy.
    return np.ascontiguousarray(img).copy()


def validate_live_image_shape(feature_key, image, expected_shape):
    """
    LeRobot PolicyFeature visual shapes are normally C,H,W.
    Accept H,W,C too for compatibility, but never resize silently.
    """
    if expected_shape is None or len(expected_shape) != 3:
        return

    h, w, c = image.shape
    actual_chw = (c, h, w)

    if expected_shape[0] in (1, 3, 4):
        expected_chw = tuple(expected_shape)
    elif expected_shape[-1] in (1, 3, 4):
        expected_chw = (
            expected_shape[-1],
            expected_shape[0],
            expected_shape[1],
        )
    else:
        return

    if actual_chw != expected_chw:
        raise ValueError(
            f"{feature_key}: live camera shape CHW={actual_chw} does not match "
            f"checkpoint feature shape {expected_shape}. "
            "Refusing to resize silently because train/deploy geometry must match."
        )


# ---------------------------------------------------------------------------
# Inference process (lerobot Conda env; no rclpy required)
# ---------------------------------------------------------------------------

def run_inference(args):
    import numpy as np
    import torch

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required in --mode inference")
    if not args.task.strip():
        raise SystemExit("--task must be a non-empty Task2 language instruction")

    _seed_everything(args.seed)
    bundle = load_pi05_policy_and_processors(args.checkpoint, args.device)
    policy = bundle["policy"]
    config = bundle["config"]
    preprocessor = bundle["preprocessor"]
    postprocessor = bundle["postprocessor"]
    device = bundle["device"]
    image_keys = bundle["image_keys"]
    expected_shapes = bundle["expected_shapes"]
    state_shape = bundle["state_shape"]
    action_shape = bundle["action_shape"]
    state_dim = int(state_shape[0])
    action_dim = int(action_shape[0])

    # PI0.5 predicts chunk_size actions but n_action_steps is the intended
    # closed-loop replanning cadence. Keep the complete chunk as latency
    # fallback while requesting a new observation-conditioned chunk every
    # n_action_steps. This never mutates the checkpoint on disk.
    chunk_size = int(getattr(config, "chunk_size", 0))
    requested_horizon = (
        args.n_action_steps_override
        if args.n_action_steps_override is not None
        else int(getattr(config, "n_action_steps", 0))
    )

    if requested_horizon is not None:
        n = int(requested_horizon)
        if n < 1 or n > chunk_size:
            raise ValueError(
                f"--n-action-steps-override must be in [1,{chunk_size}], got {n}"
            )
        config.n_action_steps = n
        get_base_model = getattr(policy, "get_base_model", None)
        if callable(get_base_model):
            base = get_base_model()
            base_cfg = getattr(base, "config", None)
            if base_cfg is not None and hasattr(base_cfg, "n_action_steps"):
                base_cfg.n_action_steps = n

    replan_interval = int(getattr(config, "n_action_steps", 0))
    # With long horizons, start CUDA inference before the old queue is empty.
    # The native 10-step checkpoint already has enough tail actions to hide
    # inference latency; longer deployment overrides need about ten steps.
    prefetch_margin_steps = (
        0 if replan_interval <= 20 else min(12, max(1, replan_interval // 5))
    )

    _reset_pipeline(
        policy,
        preprocessor,
        postprocessor,
        seed=args.seed if not args.no_reseed_on_reset else None,
    )

    print("PI0.5 Task2 inference")
    print(f"  checkpoint: {bundle['checkpoint']}")
    print(f"  device: {device}")
    print(f"  peft/LoRA: {bundle['is_peft']}")
    print(f"  dtype: {getattr(config, 'dtype', None)}")
    print(f"  chunk_size: {getattr(config, 'chunk_size', None)}")
    print(f"  n_action_steps: {getattr(config, 'n_action_steps', None)}")
    print(f"  async replan interval: {replan_interval}")
    print(f"  num_inference_steps: {getattr(config, 'num_inference_steps', None)}")
    print(f"  task: {args.task}")
    print(f"  robot_type: {args.robot_type}")
    print(f"  state/action contract: {state_dim}D/{action_dim}D")
    print("  checkpoint image features:")
    for key in image_keys:
        print(
            f"    {key} -> logical={camera_logical_from_feature(key)} "
            f"shape={expected_shapes.get(key)}"
        )

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"listening on {args.host}:{args.port}")

    executor = AsyncActionChunkExecutor(
        policy=policy,
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        task=args.task,
        robot_type=args.robot_type,
        replan_interval=replan_interval,
        prefetch_margin_steps=prefetch_margin_steps,
        debug_actions=args.debug_actions,
        action_ensemble_samples=args.action_ensemble_samples,
        action_dim=action_dim,
    )

    def reset_executor(seed=None):
        reset_seed = args.seed if seed is None else int(seed)
        executor.reset(
            lambda: _reset_pipeline(
                policy,
                preprocessor,
                postprocessor,
                seed=reset_seed if not args.no_reseed_on_reset else None,
            )
        )

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("ROS adapter connected:", addr)

        send_obj(
            conn,
            {
                "ok": True,
                "op": "hello",
                "policy": "pi05",
                "requires_observation": True,
                "peft": bool(bundle["is_peft"]),
                "image_keys": image_keys,
                "expected_shapes": {
                    k: (list(v) if v is not None else None)
                    for k, v in expected_shapes.items()
                },
                "state_shape": list(state_shape),
                "action_shape": list(action_shape),
                "task": args.task,
                "robot_type": args.robot_type,
                "chunk_size": int(getattr(config, "chunk_size", 0)),
                "n_action_steps": int(getattr(config, "n_action_steps", 0)),
                "async_execution": True,
                "replan_interval": replan_interval,
            },
        )

        first_image_log = True

        try:
            while True:
                req = recv_obj(conn)
                op = req.get("op")

                if op == "reset":
                    reset_seed = int(req.get("seed", args.seed))
                    reset_executor(reset_seed)
                    send_obj(conn, {"ok": True, "seed": reset_seed})
                    continue

                if op not in ("infer", "cached"):
                    raise ValueError(f"unknown request op={op!r}")

                t0 = time.perf_counter()
                inference_accepted = False
                if op == "infer":
                    state = np.asarray(req["state"], dtype=np.float32)
                    if state.shape != state_shape or not np.isfinite(state).all():
                        raise ValueError(
                            f"invalid Task2 state: shape={state.shape}, finite={np.isfinite(state).all()}"
                        )

                    images_pkt = req.get("images")
                    if not isinstance(images_pkt, dict):
                        raise ValueError("infer request has no 'images' dict")
                    missing = [key for key in image_keys if key not in images_pkt]
                    if missing:
                        raise ValueError(f"missing required camera packet(s): {missing}")

                    raw_observation = {"observation.state": state}
                    shape_log = {}
                    for key in image_keys:
                        image = decode_image(images_pkt[key])
                        validate_live_image_shape(key, image, expected_shapes.get(key))
                        shape_log[key] = tuple(image.shape)
                        raw_observation[key] = image

                    if first_image_log:
                        print("first live camera shapes (H,W,C):")
                        for key in image_keys:
                            print(f"  {key}: {shape_log[key]}")
                        first_image_log = False

                    inference_accepted = executor.request_inference(
                        raw_observation,
                        observation_sim_time=req.get("sim_time"),
                    )

                result = executor.get_action()
                send_obj(
                    conn,
                    {
                        "ok": True,
                        **result,
                        "request_ms": (time.perf_counter() - t0) * 1000.0,
                        "inference_accepted": inference_accepted,
                        "need_action": result["action"] is None,
                    },
                )

        except (ConnectionError, EOFError, BrokenPipeError, ConnectionResetError):
            print("ROS adapter disconnected")
        except Exception as exc:
            print(f"inference request error: {type(exc).__name__}: {exc}")
            try:
                send_obj(
                    conn,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            except Exception:
                pass
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# State/action contract
# ---------------------------------------------------------------------------

def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy * qy + qz * qz),
    )


def candidates(name):
    yield name
    if "fr3v2_joint" in name:
        yield name.replace("fr3v2_joint", "fr3v2_1_joint")
    if name == LEFT_GRIPPER_DRIVER:
        yield "left_fr3v2_finger_joint1"
    if name == RIGHT_GRIPPER_DRIVER:
        yield "right_fr3v2_finger_joint1"


def resolve(joints, name, default=math.nan):
    for n in candidates(name):
        v = joints.get(n)
        if v is not None and math.isfinite(v):
            return float(v)
    return default


def open_fraction(driver):
    if not math.isfinite(driver):
        return math.nan
    return max(
        0.0,
        min(1.0, 1.0 - driver / GRIPPER_CLOSED_RAD),
    )


def build_state(joints, ee, odom):
    s = [math.nan] * 37

    if ee["left"] is not None:
        s[0:7] = ee["left"]
    if ee["right"] is not None:
        s[7:14] = ee["right"]

    for i, n in enumerate(LEFT_JOINTS + RIGHT_JOINTS):
        s[14 + i] = resolve(joints, n)

    # Do not silently substitute 0 for missing live state.
    s[28] = resolve(joints, SPINE_JOINT)
    s[29] = open_fraction(resolve(joints, LEFT_GRIPPER_DRIVER))
    s[30] = open_fraction(resolve(joints, RIGHT_GRIPPER_DRIVER))

    if odom is not None:
        x, y, qx, qy, qz, qw, vx, vy, wz = odom
        s[31:34] = [x, y, quat_to_yaw(qx, qy, qz, qw)]
        s[34:37] = [vx, vy, wz]

    return s if all(math.isfinite(x) for x in s) else None


def build_real_state(joints, wrenches):
    """Build the official converted Munich dataset's exact 42D state order."""
    left = [resolve(joints, name) for name in LEFT_JOINTS]
    right = [resolve(joints, name) for name in RIGHT_JOINTS]
    left_wrench = wrenches.get("left")
    right_wrench = wrenches.get("right")
    if left_wrench is None or right_wrench is None:
        return None

    # In the released dataset, both external-joint-torque blocks and the left
    # gripper state are constant zero. Preserve those training values exactly.
    state = (
        left
        + [0.0]
        + [0.0] * 7
        + list(left_wrench)
        + right
        + [resolve(joints, RIGHT_GRIPPER_DRIVER)]
        + [0.0] * 7
        + list(right_wrench)
    )
    if len(state) != REAL_STATE_DIM:
        raise AssertionError(f"real Task2 state has unexpected length {len(state)}")
    return state if all(math.isfinite(float(x)) for x in state) else None


def split_task2_action(action, *, real, spine_min=0.0, spine_max=0.54, real_spine_height=434.0):
    expected_dim = REAL_ACTION_DIM if real else ISAAC_ACTION_DIM
    if len(action) != expected_dim or not all(math.isfinite(float(x)) for x in action):
        raise ValueError(f"invalid {expected_dim}D action")
    if real:
        return {
            "left_arm": list(action[0:7]),
            "left_gripper": _clip(action[7], 0.0, 1.0),
            "right_arm": list(action[8:15]),
            "right_gripper": _clip(action[15], 0.0, 1.0),
            "spine": float(real_spine_height),
        }
    return {
        "left_arm": list(action[3:10]),
        "left_gripper": _clip(action[17], 0.0, 1.0),
        "right_arm": list(action[10:17]),
        "right_gripper": _clip(action[18], 0.0, 1.0),
        "spine": _clip(action[19], spine_min, spine_max),
    }


# ---------------------------------------------------------------------------
# ROS camera discovery
# ---------------------------------------------------------------------------

def parse_camera_topic_overrides(items):
    """
    --camera-topic may be repeated:
      --camera-topic eval_camera=/some/topic
      --camera-topic observation.images.wrist_right=/some/topic
    """
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"--camera-topic expects KEY=/topic, got {item!r}"
            )
        key, topic = item.split("=", 1)
        key = key.strip()
        topic = topic.strip()
        if not key or not topic.startswith("/"):
            raise ValueError(
                f"invalid --camera-topic {item!r}; expected KEY=/absolute/topic"
            )

        logical = (
            camera_logical_from_feature(key)
            if key.startswith("observation.")
            else key
        )
        if logical not in CAMERA_LOGICAL_ALIASES:
            raise ValueError(
                f"unknown camera override key {key!r}; "
                f"supported logical keys={list(CAMERA_LOGICAL_ALIASES)}"
            )
        out[logical] = topic
    return out


def camera_topic_score(topic: str, logical: str) -> int | None:
    lower = topic.lower()

    if any(token in lower for token in BAD_CAMERA_TOPIC_TOKENS):
        return None

    score = 0

    if topic in PREFERRED_CAMERA_TOPICS.get(logical, ()):
        score += 10000

    if lower.endswith("/image_raw"):
        score += 500
    elif "image_raw" in lower:
        score += 400
    elif lower.endswith("/image"):
        score += 250
    elif "image" in lower:
        score += 100
    else:
        return None

    if "/isaac/" in lower:
        score += 100

    matched_alias = False
    for alias in CAMERA_LOGICAL_ALIASES[logical]:
        if alias in lower:
            score += 300 + len(alias)
            matched_alias = True

    if logical == "wrist_left":
        if "left" in lower:
            score += 120
        if "wrist" in lower:
            score += 120
        if "right" in lower:
            score -= 1000

    if logical == "wrist_right":
        if "right" in lower:
            score += 120
        if "wrist" in lower:
            score += 120
        if "left" in lower:
            score -= 1000

    if logical == "head" and "head" in lower:
        score += 120

    if logical == "eval_camera" and "eval" in lower:
        score += 120

    return score if matched_alias else None


def discover_camera_topics(node, logicals, overrides, timeout_s):
    """
    Resolve required logical cameras to live sensor_msgs/msg/Image topics.

    Exact benchmark topic strings do not have to be encoded in the policy.
    The feature key comes from the checkpoint; the topic comes from the ROS graph.
    """
    unresolved = set(logicals)
    result = {}

    for logical in list(unresolved):
        if logical in overrides:
            result[logical] = overrides[logical]
            unresolved.remove(logical)

    deadline = time.monotonic() + timeout_s
    last_image_topics = []

    while unresolved and time.monotonic() < deadline:
        names_and_types = node.get_topic_names_and_types()

        image_topics = []
        for topic, types in names_and_types:
            if "sensor_msgs/msg/Image" in types:
                image_topics.append(topic)
        last_image_topics = sorted(image_topics)

        newly_resolved = []
        for logical in sorted(unresolved):
            scored = []
            for topic in image_topics:
                score = camera_topic_score(topic, logical)
                if score is not None:
                    scored.append((score, topic))

            scored.sort(key=lambda x: (-x[0], x[1]))
            if not scored:
                continue

            if len(scored) >= 2 and scored[0][0] == scored[1][0]:
                # Ambiguous at the same score: do not guess.
                continue

            result[logical] = scored[0][1]
            newly_resolved.append(logical)

        for logical in newly_resolved:
            unresolved.remove(logical)

        if unresolved:
            time.sleep(0.25)

    if unresolved:
        details = "\n".join(f"  {x}" for x in last_image_topics)
        raise RuntimeError(
            "Could not uniquely discover required RGB camera topic(s): "
            f"{sorted(unresolved)}.\n"
            "Start Task2 Isaac Sim with --record and make sure the camera "
            "publishers are visible to the Jazzy container.\n"
            "Visible sensor_msgs/msg/Image topics:\n"
            f"{details or '  <none>'}\n"
            "If naming is unusual, pass e.g. "
            "--camera-topic eval_camera=/exact/topic"
        )

    # A single ROS topic must not accidentally satisfy two distinct features.
    rev = {}
    for logical, topic in result.items():
        if topic in rev and rev[topic] != logical:
            raise RuntimeError(
                f"camera discovery mapped both {rev[topic]} and {logical} "
                f"to {topic}; use --camera-topic overrides"
            )
        rev[topic] = logical

    return result


# ---------------------------------------------------------------------------
# ROS process (Jazzy container; no LeRobot required)
# ---------------------------------------------------------------------------

def run_ros(args):
    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32, Float32MultiArray, String

    real = args.ros_profile == "real"
    # The real platform publishes arm/spine/gripper state on separate topics;
    # Isaac exposes one aggregate state stream.  Keep both wiring profiles in
    # this adapter so the same container can be evaluated in either setting.
    if real:
        state_topics = {
            "left": args.real_left_joint_states,
            "right": args.real_right_joint_states,
            "left_gripper": args.real_left_gripper_states,
            "right_gripper": args.real_right_gripper_states,
            "left_wrench": args.real_left_wrench,
            "right_wrench": args.real_right_wrench,
        }
        command_topics = {
            "left": args.real_left_arm_command,
            "right": args.real_right_arm_command,
            "left_gripper": args.real_left_gripper_command,
            "right_gripper": args.real_right_gripper_command,
            "spine": args.real_spine_command,
        }
    else:
        state_topics = {"left": FULL_STATES_TOPIC, "right": FULL_STATES_TOPIC,
                        "spine": FULL_STATES_TOPIC, "left_gripper": FULL_STATES_TOPIC,
                        "right_gripper": FULL_STATES_TOPIC}
        command_topics = {"left": LEFT_ARM_CMD_TOPIC, "right": RIGHT_ARM_CMD_TOPIC,
                          "left_gripper": LEFT_GRIPPER_CMD_TOPIC,
                          "right_gripper": RIGHT_GRIPPER_CMD_TOPIC,
                          "spine": SPINE_CMD_TOPIC}

    overrides = parse_camera_topic_overrides(args.camera_topic)
    latest_sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    class NodeImpl(Node):
        def __init__(self):
            super().__init__("task2_policy_ros_adapter")

            self.lock = threading.Lock()
            self.cond = threading.Condition(self.lock)

            self.sim_time = None
            self.last_clock = None
            self.last_clock_wall = time.monotonic()
            self.joints = {}
            self.wrenches = {"left": None, "right": None}
            self.real_state_arrival = {}
            self.odom = None
            self.ee = {"left": None, "right": None}

            self.required_image_keys = []
            self.feature_to_logical = {}
            self.camera_topics = {}
            self.camera_frames = {}
            self.camera_subscriptions = []

            self.reset_pending = True
            self.awaiting_initial_scene_reset = bool(args.reset_scene_on_start and not real)
            self.initial_scene_reset_requested = False
            self.stop = False
            self.last_wait_log_wall = 0.0
            self.last_safety_log_wall = 0.0
            self.last_dry_run_log_wall = 0.0
            self.action_safety = Task2ActionSafetyFilter(
                spine_min=(args.real_spine_height if real else args.spine_min),
                spine_max=(args.real_spine_height if real else args.spine_max),
                step_scale=args.action_safety_step_scale,
                contract=("real42_17" if real else "isaac37_20"),
            )
            self.pad_baseline_median_z = None
            self.pad_lift_started_sim = None
            self.pad_max_lift_m = 0.0
            self.grasp_detected = False
            self.rollout_index = args.rollout_index_start
            self.rollout_seed = args.rollout_seed_start + self.rollout_index
            self.rollout_start_sim = None
            self.rollout_results = []

            self.pubs = {}
            if not args.dry_run:
                self.pubs = {
                    "la": self.create_publisher(JointState, command_topics["left"], 10),
                    "ra": self.create_publisher(JointState, command_topics["right"], 10),
                    "lg": self.create_publisher(
                        Float32 if real else JointState, command_topics["left_gripper"], 10
                    ),
                    "rg": self.create_publisher(
                        Float32 if real else JointState, command_topics["right_gripper"], 10
                    ),
                    "sp": self.create_publisher(
                        Float32 if real else JointState, command_topics["spine"], 10
                    ),
                    "scene_reset": self.create_publisher(String, SCENE_RESET_REQUEST_TOPIC, 10),
                }

            if not real:
                self.create_subscription(Clock, CLOCK_TOPIC, self.cb_clock, 10)
            if real:
                for side, topic in (("left", state_topics["left"]), ("right", state_topics["right"]),
                                    ("left_gripper", state_topics["left_gripper"]),
                                    ("right_gripper", state_topics["right_gripper"])):
                    self.create_subscription(JointState, topic,
                                             lambda m, side=side: self.cb_real_joints(side, m), latest_sensor_qos)
                self.create_subscription(
                    WrenchStamped, state_topics["left_wrench"],
                    lambda m: self.cb_real_wrench("left", m), latest_sensor_qos,
                )
                self.create_subscription(
                    WrenchStamped, state_topics["right_wrench"],
                    lambda m: self.cb_real_wrench("right", m), latest_sensor_qos,
                )
            else:
                self.create_subscription(JointState, FULL_STATES_TOPIC, self.cb_joints, 10)
            if not real:
                self.create_subscription(Odometry, ODOM_TOPIC, self.cb_odom, 10)
            if not real:
                self.create_subscription(
                    PoseStamped, LEFT_EE_TOPIC, lambda m: self.cb_ee("left", m), 10
                )
                self.create_subscription(
                    PoseStamped, RIGHT_EE_TOPIC, lambda m: self.cb_ee("right", m), 10
                )
                self.create_subscription(String, SCENE_RESET_TOPIC, self.cb_reset, 10)
                self.create_subscription(Float32MultiArray, PAD_POINTS_TOPIC, self.cb_pad_points, 10)

            # Connect before starting the control thread so the checkpoint can
            # tell us which camera(s) it needs.
            sock, hello = self.connect()
            if sock is None:
                raise RuntimeError("rclpy stopped before inference connection")

            self.configure_checkpoint_cameras(hello)
            self.server_n_action_steps = int(hello.get("n_action_steps") or 0)
            self.state_shape = tuple(hello.get("state_shape") or ())
            self.action_shape = tuple(hello.get("action_shape") or ())
            self.server_replan_interval = int(hello.get("replan_interval") or 0)
            self.server_queue_remaining = 0
            self.server_inference_busy = False
            self.server_replan_due = True
            self.last_chunk_id = 0
            self.last_action = None
            self.action_hold_count = 0
            self.last_latency_warn_wall = 0.0

            self.worker = threading.Thread(
                target=self.loop,
                args=(sock,),
                daemon=True,
            )
            self.worker.start()

            self.get_logger().info(
                f"adapter started, inference={args.host}:{args.port}"
            )
            if args.disable_action_safety:
                self.get_logger().warning(
                    "action safety is DISABLED; raw absolute policy targets will be published"
                )
            else:
                self.get_logger().info(
                    "action safety enabled: demonstrated arm ranges + P99.5 target slew limits; "
                    f"step_scale={args.action_safety_step_scale:g}"
                )
            if args.dry_run:
                self.get_logger().warning(
                    "DRY RUN enabled: inference runs, but no arm/gripper/spine publishers exist"
                )

        # -------------------------- callbacks --------------------------

        def clear_camera_frames_locked(self):
            for logical in self.camera_frames:
                self.camera_frames[logical] = None

        def cb_clock(self, m):
            t = m.clock.sec + m.clock.nanosec * 1e-9
            with self.cond:
                self.last_clock_wall = time.monotonic()
                if (
                    self.last_clock is not None
                    and t < self.last_clock - 0.1
                ):
                    self.reset_pending = True
                    self.action_safety.reset()
                    self.clear_camera_frames_locked()

                self.last_clock = t
                self.sim_time = t
                self.cond.notify_all()

        def cb_joints(self, m):
            with self.lock:
                for i, n in enumerate(m.name):
                    if i < len(m.position):
                        self.joints[n] = float(m.position[i])

        def cb_real_joints(self, side, m):
            """Accept official per-device JointState streams and normalize names."""
            with self.lock:
                if side == "left":
                    names = LEFT_JOINTS
                elif side == "right":
                    names = RIGHT_JOINTS
                elif side == "spine":
                    names = [SPINE_JOINT]
                elif side == "left_gripper":
                    names = [LEFT_GRIPPER_DRIVER]
                else:
                    names = [RIGHT_GRIPPER_DRIVER]
                assigned = 0
                # Prefer semantic name matching.  The official broadcaster
                # names are stable, while message ordering is not guaranteed.
                for incoming, value in zip(m.name, m.position):
                    low = incoming.lower()
                    match = re.search(r"(?:joint|j)[_\- ]?([1-7])(?:$|[^0-9])", low)
                    if side in ("left", "right") and match:
                        idx = int(match.group(1)) - 1
                    else:
                        idx = 0 if assigned == 0 else None
                    if idx is not None and idx < len(names):
                        self.joints[names[idx]] = float(value)
                        assigned += 1
                if assigned == 0:
                    for i, value in enumerate(m.position[:len(names)]):
                        self.joints[names[i]] = float(value)
                if real:
                    self.real_state_arrival[side] = time.monotonic()
                    self.sim_time = time.monotonic()
                    self.last_clock_wall = self.sim_time

        def cb_real_wrench(self, side, m):
            w = m.wrench
            values = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
            with self.lock:
                self.wrenches[side] = [float(value) for value in values]
                self.real_state_arrival[f"{side}_wrench"] = time.monotonic()
                self.sim_time = time.monotonic()
                self.last_clock_wall = self.sim_time

        def cb_odom(self, m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            v = m.twist.twist.linear
            w = m.twist.twist.angular
            with self.lock:
                self.odom = (
                    p.x,
                    p.y,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                    v.x,
                    v.y,
                    w.z,
                )
                if real:
                    self.sim_time = time.monotonic()
                    self.last_clock_wall = time.monotonic()

        def cb_ee(self, side, m):
            p = m.pose.position
            q = m.pose.orientation
            with self.lock:
                self.ee[side] = [
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                ]

        def cb_camera(self, logical, m):
            # Arrival sim time is retained for freshness. For cross-camera sync
            # prefer Image.header.stamp when available: large RGB callbacks can
            # arrive on different sim ticks even when captured together.
            stamp_sim = None
            try:
                stamp_sim = float(m.header.stamp.sec) + float(m.header.stamp.nanosec) * 1e-9
                if stamp_sim <= 0.0:
                    stamp_sim = None
            except Exception:
                stamp_sim = None
            with self.lock:
                if real:
                    # ROS header time and Python monotonic time are unrelated.
                    # Use local arrival time for real-robot freshness checks.
                    arrival = time.monotonic()
                    self.sim_time = arrival
                    stamp_sim = None
                else:
                    arrival = self.sim_time
                self.camera_frames[logical] = (m, arrival, stamp_sim)

        def cb_reset(self, _m):
            with self.cond:
                self.awaiting_initial_scene_reset = False
                self.reset_pending = True
                self.action_safety.reset()
                self.clear_camera_frames_locked()
                self.pad_baseline_median_z = None
                self.pad_lift_started_sim = None
                self.pad_max_lift_m = 0.0
                self.grasp_detected = False
                self.cond.notify_all()

        def cb_pad_points(self, m):
            data = m.data
            if len(data) < 5:
                return
            sample_sim = float(data[0])
            count = int(round(float(data[1])))
            if count <= 0 or len(data) != 2 + 3 * count:
                return
            z = [float(data[4 + 3 * i]) for i in range(count)]
            z.sort()
            median_z = z[len(z) // 2]

            with self.lock:
                if self.pad_baseline_median_z is None:
                    self.pad_baseline_median_z = median_z
                    return
                lift = median_z - self.pad_baseline_median_z
                self.pad_max_lift_m = max(self.pad_max_lift_m, lift)
                # This detector is telemetry only.  It must never alter the
                # continuous policy rollout or terminate the ROS node.
                if lift >= args.grasp_lift_threshold_m:
                    if self.pad_lift_started_sim is None:
                        self.pad_lift_started_sim = sample_sim
                    sustained = sample_sim - self.pad_lift_started_sim
                    if not self.grasp_detected and sustained >= args.grasp_lift_sustain_sim:
                        self.grasp_detected = True
                        self.get_logger().info(
                            "GRASP DETECTED: thermal-pad median height sustained "
                            f"{lift:.3f}m above reset baseline for {sustained:.2f}s sim"
                        )
                else:
                    self.pad_lift_started_sim = None

        def write_rollout_result(self, result):
            self.rollout_results.append(result)
            if not args.result_jsonl:
                return
            output = Path(args.result_jsonl)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")

        def finish_rollout(self, sim, reason):
            elapsed = (
                None
                if self.rollout_start_sim is None
                else max(0.0, float(sim) - self.rollout_start_sim)
            )
            result = {
                "rollout_index": self.rollout_index,
                "seed": self.rollout_seed,
                "success": bool(self.grasp_detected),
                "reason": reason,
                "elapsed_sim_s": elapsed,
                "pad_max_lift_m": self.pad_max_lift_m,
                "action_hold_count": self.action_hold_count,
                "completed_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            self.write_rollout_result(result)
            self.get_logger().info("ROLLOUT RESULT " + json.dumps(result, sort_keys=True))

            self.rollout_index += 1
            if self.rollout_index >= args.rollout_count:
                self.stop = True
                self.get_logger().info(
                    f"rollout batch complete: {sum(r['success'] for r in self.rollout_results)}"
                    f"/{len(self.rollout_results)} grasp successes"
                )
                rclpy.shutdown()
                return

            self.rollout_seed = args.rollout_seed_start + self.rollout_index
            self.rollout_start_sim = None
            self.awaiting_initial_scene_reset = True
            self.initial_scene_reset_requested = False

        # --------------------- checkpoint handshake --------------------

        def connect(self):
            while rclpy.ok() and not self.stop:
                try:
                    s = socket.create_connection(
                        (args.host, args.port),
                        timeout=2,
                    )
                    s.settimeout(None)
                    s.setsockopt(
                        socket.IPPROTO_TCP,
                        socket.TCP_NODELAY,
                        1,
                    )

                    hello = recv_obj(s)
                    if (
                        not hello.get("ok")
                        or hello.get("op") != "hello"
                    ):
                        raise RuntimeError(
                            f"bad inference handshake: {hello}"
                        )

                    if hello.get("policy") != "pi05":
                        raise RuntimeError(
                            f"expected PI0.5 inference server, got policy={hello.get('policy')!r}"
                        )
                    expected_state = (REAL_STATE_DIM,) if real else (ISAAC_STATE_DIM,)
                    expected_action = (REAL_ACTION_DIM,) if real else (ISAAC_ACTION_DIM,)
                    if tuple(hello.get("state_shape") or ()) != expected_state:
                        raise RuntimeError(
                            f"PI0.5 server state contract mismatch: {hello.get('state_shape')}"
                        )
                    if tuple(hello.get("action_shape") or ()) != expected_action:
                        raise RuntimeError(
                            f"PI0.5 server action contract mismatch: {hello.get('action_shape')}"
                        )

                    image_keys = list(hello.get("image_keys") or [])
                    if not image_keys:
                        raise RuntimeError(
                            "inference checkpoint reported no image keys"
                        )

                    self.get_logger().info(
                        "connected to PI0.5 inference; "
                        f"peft={hello.get('peft')} n_action_steps={hello.get('n_action_steps')} "
                        f"async={hello.get('async_execution')} "
                        f"replan_interval={hello.get('replan_interval')} "
                        f"task={hello.get('task')!r} cameras={image_keys}"
                    )
                    return s, hello

                except (
                    OSError,
                    ConnectionError,
                    EOFError,
                    RuntimeError,
                ) as exc:
                    self.get_logger().warning(
                        f"inference unavailable: {exc}"
                    )
                    time.sleep(1)

            return None, None

        def configure_checkpoint_cameras(self, hello):
            image_keys = list(hello["image_keys"])
            feature_to_logical = {
                key: camera_logical_from_feature(key)
                for key in image_keys
            }

            logicals = sorted(set(feature_to_logical.values()))
            topics = discover_camera_topics(
                self,
                logicals,
                overrides,
                args.camera_discovery_timeout,
            )

            self.required_image_keys = image_keys
            self.feature_to_logical = feature_to_logical
            self.camera_topics = topics
            self.camera_frames = {
                logical: None
                for logical in logicals
            }

            for logical in logicals:
                topic = topics[logical]
                sub = self.create_subscription(
                    Image,
                    topic,
                    lambda m, logical=logical: self.cb_camera(
                        logical, m
                    ),
                    latest_sensor_qos,
                )
                self.camera_subscriptions.append(sub)

            self.get_logger().info(
                "checkpoint camera wiring:"
            )
            for key in image_keys:
                logical = feature_to_logical[key]
                self.get_logger().info(
                    f"  {key} -> {topics[logical]}"
                )

        def validate_reconnect_handshake(self, hello):
            if hello.get("policy") != "pi05":
                raise RuntimeError(f"reconnected server is not PI0.5: {hello}")
            if (
                tuple(hello.get("state_shape") or ()) != self.state_shape
                or tuple(hello.get("action_shape") or ()) != self.action_shape
            ):
                raise RuntimeError(f"reconnected PI0.5 server changed Task2 state/action contract: {hello}")
            if not hello.get("async_execution"):
                raise RuntimeError("PI0.5 inference server does not support async chunk execution")
            new_keys = list(hello.get("image_keys") or [])
            if new_keys != self.required_image_keys:
                raise RuntimeError(
                    "inference server restarted with a different camera "
                    f"set: old={self.required_image_keys}, new={new_keys}. "
                    "Restart the ROS adapter so subscriptions are rebuilt "
                    "from the new checkpoint."
                )
            new_n = int(hello.get("n_action_steps") or 0)
            if self.server_n_action_steps and new_n != self.server_n_action_steps:
                raise RuntimeError(
                    f"inference server changed n_action_steps: old={self.server_n_action_steps}, new={new_n}; "
                    "restart the ROS adapter"
                )
            new_interval = int(hello.get("replan_interval") or 0)
            if new_interval != self.server_replan_interval:
                raise RuntimeError(
                    f"inference server changed replan interval: old={self.server_replan_interval}, "
                    f"new={new_interval}; restart the ROS adapter"
                )

        # -------------------------- snapshot ---------------------------

        @staticmethod
        def image_packet(m):
            return {
                "height": int(m.height),
                "width": int(m.width),
                "step": int(m.step),
                "encoding": str(m.encoding),
                "data": bytes(m.data),
            }

        def maybe_log_wait(self, reason):
            now = time.monotonic()
            if now - self.last_wait_log_wall >= 2.0:
                self.get_logger().warning(reason)
                self.last_wait_log_wall = now

        def snap(self):
            with self.lock:
                if self.sim_time is None:
                    return None

                if real:
                    required_sources = ("left", "right", "right_gripper", "left_wrench", "right_wrench")
                    missing_sources = [
                        source for source in required_sources
                        if source not in self.real_state_arrival
                    ]
                    stale_sources = [
                        source for source in required_sources
                        if source in self.real_state_arrival
                        and time.monotonic() - self.real_state_arrival[source] > args.real_state_max_age
                    ]
                    if missing_sources or stale_sources:
                        self.maybe_log_wait(
                            f"waiting for fresh real state; missing={missing_sources} stale={stale_sources}"
                        )
                        return None
                    s = build_real_state(dict(self.joints), dict(self.wrenches))
                else:
                    s = build_state(
                        dict(self.joints),
                        {k: (None if v is None else list(v)) for k, v in self.ee.items()},
                        self.odom,
                    )
                if s is None:
                    self.maybe_log_wait(
                        f"waiting for complete {self.state_shape[0]}D robot state"
                    )
                    return None

                packets = {}
                capture_times = []

                for key in self.required_image_keys:
                    logical = self.feature_to_logical[key]
                    item = self.camera_frames.get(logical)
                    if item is None:
                        self.maybe_log_wait(
                            f"waiting for camera {key} on "
                            f"{self.camera_topics[logical]}"
                        )
                        return None

                    msg, arrival_sim, header_sim = item
                    if arrival_sim is None:
                        return None

                    # Image.header.stamp is the render/capture time.  Callback
                    # arrival time can make an old DDS sample look fresh when
                    # the ROS executor is busy decoding multi-megabyte images.
                    # Fall back only for publishers that do not stamp images.
                    capture_sim = (
                        float(header_sim)
                        if header_sim is not None
                        else float(arrival_sim)
                    )
                    age = self.sim_time - capture_sim
                    if age > args.camera_max_age_sim:
                        self.maybe_log_wait(
                            f"camera {key} stale by {age:.3f}s sim"
                        )
                        return None

                    packets[key] = self.image_packet(msg)
                    capture_times.append(capture_sim)

                if (
                    len(capture_times) > 1
                    and max(capture_times) - min(capture_times)
                    > args.camera_sync_tolerance_sim
                ):
                    skew = max(capture_times) - min(capture_times)
                    self.maybe_log_wait(
                        f"waiting for synchronized cameras; "
                        f"latest capture skew={skew:.3f}s sim"
                    )
                    return None

                return {
                    "sim_time": self.sim_time,
                    "state": s,
                    "images": packets,
                }

        # ----------------------- command publish -----------------------

        def pub(self, pub, names, pos):
            m = JointState()
            m.header.stamp = self.get_clock().now().to_msg()
            m.name = list(names)
            m.position = [float(x) for x in pos]
            pub.publish(m)

        def apply(self, a):
            expected_dim = REAL_ACTION_DIM if real else ISAAC_ACTION_DIM
            if len(a) != expected_dim or not all(math.isfinite(float(x)) for x in a):
                raise ValueError(f"invalid {expected_dim}D action")
            if not real and max(abs(float(x)) for x in a[0:3]) > args.base_warn_threshold:
                self.get_logger().warning(f"nonzero base action {a[0:3]} ignored for fixpos")

            if not args.disable_action_safety:
                with self.lock:
                    a, safety = self.action_safety.filter(list(a), dict(self.joints))
                if safety["limited"]:
                    now = time.monotonic()
                    if now - self.last_safety_log_wall >= 1.0:
                        self.get_logger().warning(
                            "action safety limited policy target: "
                            f"raw_arm_delta={safety['raw_max_arm_delta']:.3f}rad "
                            f"safe_arm_delta={safety['safe_max_arm_delta']:.3f}rad "
                            f"range_dims={safety['range_limited']} "
                            f"slew_dims={safety['rate_limited']} "
                            f"limited_steps={safety['total_limited_steps']}"
                        )
                        self.last_safety_log_wall = now

            parts = split_task2_action(
                a,
                real=real,
                spine_min=args.spine_min,
                spine_max=args.spine_max,
                real_spine_height=args.real_spine_height,
            )
            left_arm = parts["left_arm"]
            right_arm = parts["right_arm"]
            lo = parts["left_gripper"]
            ro = parts["right_gripper"]
            spine = parts["spine"]
            command_names = REAL_ARM_COMMAND_NAMES if real else LEFT_JOINTS

            if args.dry_run:
                now = time.monotonic()
                if now - self.last_dry_run_log_wall >= 1.0:
                    self.get_logger().info(
                        "DRY RUN action: "
                        f"left={[round(float(x), 3) for x in left_arm]} left_gripper={lo:.3f} "
                        f"right={[round(float(x), 3) for x in right_arm]} right_gripper={ro:.3f} "
                        f"spine={spine:.3f}"
                    )
                    self.last_dry_run_log_wall = now
                return

            self.pub(self.pubs["la"], command_names, left_arm)
            self.pub(
                self.pubs["ra"],
                REAL_ARM_COMMAND_NAMES if real else RIGHT_JOINTS,
                right_arm,
            )

            if real:
                self.pubs["lg"].publish(Float32(data=lo))
                self.pubs["rg"].publish(Float32(data=ro))
            else:
                self.pub(self.pubs["lg"], [LEFT_GRIPPER_DRIVER], [(1.0 - lo) * GRIPPER_CLOSED_RAD])
                self.pub(self.pubs["rg"], [RIGHT_GRIPPER_DRIVER], [(1.0 - ro) * GRIPPER_CLOSED_RAD])

            if real:
                self.pubs["sp"].publish(Float32(data=spine))
            else:
                self.pub(self.pubs["sp"], [SPINE_JOINT], [spine])

        # -------------------------- control ----------------------------

        def loop(self, sock):
            period = 1.0 / args.fps
            next_sim = None

            while rclpy.ok() and not self.stop:
                if sock is None:
                    sock, hello = self.connect()
                    next_sim = None
                    if sock is None:
                        return
                    try:
                        self.validate_reconnect_handshake(hello)
                        self.server_queue_remaining = 0
                        self.server_inference_busy = False
                        self.server_replan_due = True
                        self.last_action = None
                    except Exception as exc:
                        self.get_logger().error(str(exc))
                        try:
                            sock.close()
                        except Exception:
                            pass
                        return
                    self.reset_pending = True

                try:
                    with self.cond:
                        self.cond.wait(timeout=0.05)
                        sim = self.sim_time
                        reset = self.reset_pending
                        self.reset_pending = False

                    if (
                        args.clock_stall_wall > 0
                        and time.monotonic() - self.last_clock_wall >= args.clock_stall_wall
                    ):
                        self.get_logger().error(
                            f"no {'real state' if real else '/isaac/clock'} update for "
                            f"{args.clock_stall_wall:.1f}s wall; "
                            "aborting without recording the in-progress rollout"
                        )
                        self.stop = True
                        rclpy.shutdown()
                        return

                    if sim is None:
                        continue

                    if self.awaiting_initial_scene_reset:
                        if (
                            not self.initial_scene_reset_requested
                            and self.pubs["scene_reset"].get_subscription_count() > 0
                        ):
                            msg = String()
                            msg.data = "policy_start"
                            self.pubs["scene_reset"].publish(msg)
                            self.initial_scene_reset_requested = True
                            self.get_logger().info(
                                "requested deterministic Task2 scene reset; "
                                "waiting for reset acknowledgement"
                            )
                        else:
                            self.maybe_log_wait(
                                "waiting for Task2 scene-reset service/acknowledgement"
                            )
                        continue

                    if reset:
                        with self.lock:
                            self.action_safety.reset()
                        send_obj(sock, {"op": "reset", "seed": self.rollout_seed})
                        rep = recv_obj(sock)
                        if not rep.get("ok"):
                            raise RuntimeError(rep.get("error"))

                        self.server_queue_remaining = 0
                        self.server_inference_busy = False
                        self.server_replan_due = True
                        self.last_chunk_id = 0
                        self.last_action = None
                        self.action_hold_count = 0
                        next_sim = sim + args.start_delay_sim
                        self.rollout_start_sim = next_sim
                        self.get_logger().info(
                            f"policy reset; rollout={self.rollout_index + 1}/{args.rollout_count} "
                            f"seed={self.rollout_seed} start @ sim {next_sim:.3f}"
                        )
                        continue

                    if next_sim is None:
                        next_sim = sim + args.start_delay_sim
                        continue

                    if sim + 1e-9 < next_sim:
                        continue

                    if (
                        args.rollout_max_sim > 0
                        and self.rollout_start_sim is not None
                        and sim - self.rollout_start_sim >= args.rollout_max_sim
                    ):
                        self.finish_rollout(sim, "timeout")
                        next_sim = None
                        continue

                    t0 = time.perf_counter()
                    should_prefetch = not self.server_inference_busy and bool(
                        self.server_replan_due
                    )
                    if should_prefetch:
                        snap = self.snap()
                        if snap is None:
                            # Keep executing an existing queue even when a fresh
                            # multi-camera snapshot is temporarily unavailable.
                            if self.server_queue_remaining > 0:
                                send_obj(sock, {"op": "cached"})
                            else:
                                continue
                        else:
                            send_obj(
                                sock,
                                {
                                    "op": "infer",
                                    "sim_time": snap["sim_time"],
                                    "state": snap["state"],
                                    "images": snap["images"],
                                },
                            )
                    else:
                        send_obj(sock, {"op": "cached"})

                    rep = recv_obj(sock)
                    if not rep.get("ok"):
                        raise RuntimeError(rep.get("error"))

                    self.server_queue_remaining = int(rep.get("queue_remaining") or 0)
                    self.server_inference_busy = bool(rep.get("inference_busy"))
                    self.server_replan_due = bool(rep.get("replan_due"))

                    action = rep.get("action")
                    if action is None:
                        # Expected only during initial asynchronous warmup. Do
                        # not advance the control schedule until action 0 exists.
                        if not rep.get("need_action"):
                            raise RuntimeError("async server returned no action without need_action")
                        next_sim += period
                        with self.lock:
                            cur = self.sim_time
                        if cur is not None and cur - next_sim > 2 * period:
                            next_sim = cur + period
                        continue

                    self.apply(action)
                    self.last_action = action
                    if rep.get("held"):
                        self.action_hold_count += 1

                    chunk_id = int(rep.get("chunk_id") or 0)
                    if chunk_id != self.last_chunk_id:
                        self.last_chunk_id = chunk_id
                        obs_sim = rep.get("observation_sim_time")
                        obs_age_ms = None
                        if obs_sim is not None:
                            obs_age_ms = max(0.0, (sim - float(obs_sim)) * 1000.0)
                        self.get_logger().info(
                            f"async chunk={chunk_id} inference={float(rep.get('inference_ms') or 0):.1f}ms "
                            f"delay_drop={int(rep.get('inference_delay_steps') or 0)} "
                            f"queue={self.server_queue_remaining} "
                            + (f"obs_age={obs_age_ms:.1f}ms" if obs_age_ms is not None else "obs_age=n/a")
                        )

                    if rep.get("held"):
                        now_wall = time.monotonic()
                        if now_wall - self.last_latency_warn_wall >= 2.0:
                            self.get_logger().warning(
                                "PI0.5 async action queue exhausted; repeating the last target "
                                f"(hold_count={self.action_hold_count}, inference_busy={self.server_inference_busy})"
                            )
                            self.last_latency_warn_wall = now_wall

                    next_sim += period

                    with self.lock:
                        cur = self.sim_time

                    if cur is not None and cur - next_sim > 2 * period:
                        next_sim = cur + period

                    if int(sim * 2) != int((sim - period) * 2):
                        self.get_logger().info(
                            f"request={rep.get('request_ms', 0):.1f}ms "
                            f"queue={self.server_queue_remaining} "
                            f"busy={self.server_inference_busy} "
                            f"since_replan={int(rep.get('steps_since_replan') or 0)} "
                            f"holds={self.action_hold_count} "
                            f"pad_lift={self.pad_max_lift_m:.3f}m "
                            f"grasp={self.grasp_detected} "
                            f"cams={len(self.required_image_keys)} "
                            f"spine={action[16 if real else 19]:.3f}"
                        )

                except (
                    ConnectionError,
                    EOFError,
                    BrokenPipeError,
                    ConnectionResetError,
                    OSError,
                ) as exc:
                    self.get_logger().error(
                        f"connection lost: {exc}"
                    )
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = None
                    next_sim = None

                except Exception as exc:
                    self.get_logger().error(
                        f"control error: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    time.sleep(0.1)

        def destroy_node(self):
            self.stop = True
            with self.cond:
                self.cond.notify_all()

            if self.worker.is_alive():
                self.worker.join(timeout=2)

            return super().destroy_node()

    rclpy.init()
    node = NodeImpl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_async():
    p = argparse.ArgumentParser(
        description=(
            "EBiM Task2 PI0.5 inference server + ROS2 Jazzy adapter. "
            "Inference mode loads LeRobot full or PEFT/LoRA PI0.5 checkpoints; "
            "ROS mode supports both the Isaac 37D/20D contract and the official "
            "real-robot 42D/17D contract."
        )
    )

    p.add_argument("--mode", choices=("inference", "ros"), required=True)
    p.add_argument("--ros-profile", choices=("isaac", "real"), default="isaac",
                   help="ROS wiring profile; real uses official EBiM robot topics")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)

    # Inference-side arguments (host lerobot conda environment).
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--n-action-steps-override",
        type=int,
        default=None,
        help=(
            "inference-only PI0.5 closed-loop replanning interval override; "
            "must be <= chunk_size. Defaults to the checkpoint's n_action_steps."
        ),
    )
    p.add_argument(
        "--debug-actions",
        action="store_true",
        help="print the first postprocessed action of every newly planned chunk",
    )
    p.add_argument(
        "--action-ensemble-samples",
        type=int,
        default=1,
        help=(
            "inference-only PI0.5 action-chunk samples to average per observation; "
            "1 preserves the single-sample path"
        ),
    )
    p.add_argument(
        "--no-reseed-on-reset",
        action="store_true",
        help=(
            "Do not restore --seed on scene reset. By default each Task2 rollout starts "
            "from the same PI0.5 flow-matching RNG state for reproducible checkpoint comparisons."
        ),
    )

    # ROS-side arguments (Jazzy container; no LeRobot dependency).
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--start-delay-sim", type=float, default=1.0)
    p.add_argument(
        "--reset-scene-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "request and await a deterministic Task2 scene reset before the first policy action"
        ),
    )
    p.add_argument("--spine-min", type=float, default=0.0)
    p.add_argument("--spine-max", type=float, default=0.54)
    p.add_argument("--base-warn-threshold", type=float, default=0.02)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run observation and inference paths without creating command publishers",
    )
    p.add_argument(
        "--disable-action-safety",
        action="store_true",
        help="disable demonstrated-range and target-slew protection (diagnostics only)",
    )
    p.add_argument(
        "--action-safety-step-scale",
        type=float,
        default=1.0,
        help="scale the demonstration-derived arm/spine per-command slew limits",
    )
    p.add_argument(
        "--grasp-lift-threshold-m",
        type=float,
        default=0.025,
        help="pad median-height increase required for preliminary grasp detection",
    )
    p.add_argument(
        "--grasp-lift-sustain-sim",
        type=float,
        default=0.30,
        help="simulation seconds the pad lift must persist before logging GRASP DETECTED",
    )
    p.add_argument(
        "--rollout-count",
        type=int,
        default=1,
        help="number of automatically reset rollouts to run in ROS mode",
    )
    p.add_argument(
        "--rollout-seed-start",
        type=int,
        default=1000,
        help="model RNG seed for rollout 0; subsequent rollouts increment it by one",
    )
    p.add_argument(
        "--rollout-index-start",
        type=int,
        default=0,
        help="resume at this zero-based rollout index while keeping --rollout-count as the total",
    )
    p.add_argument(
        "--rollout-max-sim",
        type=float,
        default=0.0,
        help="finish each rollout after this many simulation seconds; 0 disables timeout",
    )
    p.add_argument(
        "--result-jsonl",
        help="append one structured ROS rollout result per line to this path",
    )
    p.add_argument(
        "--clock-stall-wall",
        type=float,
        default=30.0,
        help="abort if /isaac/clock has no update for this many wall seconds; 0 disables",
    )

    p.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        metavar="KEY=/TOPIC",
        help=(
            "Optional camera topic override. May be repeated, e.g. "
            "--camera-topic wrist_left=/isaac/wrist_left_camera/image_raw. "
            "Normally unnecessary because the required cameras come from the checkpoint "
            "and topics are discovered from the live ROS graph."
        ),
    )
    p.add_argument(
        "--camera-discovery-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for required sensor_msgs/msg/Image topics",
    )
    p.add_argument(
        "--camera-max-age-sim",
        type=float,
        default=0.50,
        help=(
            "maximum image capture age in simulation seconds (default 0.50; "
            "the four high-resolution Isaac cameras update asynchronously)"
        ),
    )
    p.add_argument(
        "--camera-sync-tolerance-sim",
        type=float,
        default=0.50,
        help=(
            "maximum capture-time skew between required cameras in sim seconds; "
            "uses Image header stamps when available, otherwise callback-arrival sim time"
        ),
    )
    p.add_argument("--real-left-joint-states", dest="real_left_joint_states",
                   default="/left/franka_robot_state_broadcaster/measured_joint_states")
    p.add_argument("--real-right-joint-states", dest="real_right_joint_states",
                   default="/right/franka_robot_state_broadcaster/measured_joint_states")
    p.add_argument("--real-left-wrench", default="/left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame")
    p.add_argument("--real-right-wrench", default="/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame")
    p.add_argument("--real-left-gripper-states", dest="real_left_gripper_states", default="/left/gripper/joint_states")
    p.add_argument("--real-right-gripper-states", dest="real_right_gripper_states", default="/right/gripper/joint_states")
    p.add_argument("--real-left-arm-command", dest="real_left_arm_command", default="/left/gello/joint_states")
    p.add_argument("--real-right-arm-command", dest="real_right_arm_command", default="/right/gello/joint_states")
    p.add_argument("--real-left-gripper-command", dest="real_left_gripper_command", default="/left/gripper/gripper_client/target_gripper_width_percent")
    p.add_argument("--real-right-gripper-command", dest="real_right_gripper_command", default="/right/gripper/gripper_client/target_gripper_width_percent")
    p.add_argument("--real-spine-command", dest="real_spine_command", default="/spine/target_height")
    p.add_argument("--real-spine-height", type=float, default=REAL_DEMO_SPINE_HEIGHT)
    p.add_argument("--real-state-max-age", type=float, default=0.5)

    args = p.parse_args()

    if not (1 <= args.port <= 65535):
        p.error("--port must be in 1..65535")
    if args.fps <= 0:
        p.error("--fps must be > 0")
    if args.start_delay_sim < 0:
        p.error("--start-delay-sim must be >= 0")
    if args.spine_min > args.spine_max:
        p.error("--spine-min must be <= --spine-max")
    if args.action_safety_step_scale <= 0:
        p.error("--action-safety-step-scale must be > 0")
    if args.real_state_max_age <= 0:
        p.error("--real-state-max-age must be > 0")
    if args.dry_run and args.ros_profile != "real":
        p.error("--dry-run is currently supported only with --ros-profile real")
    if args.grasp_lift_threshold_m <= 0:
        p.error("--grasp-lift-threshold-m must be > 0")
    if args.grasp_lift_sustain_sim < 0:
        p.error("--grasp-lift-sustain-sim must be >= 0")
    if args.rollout_count <= 0:
        p.error("--rollout-count must be > 0")
    if not (0 <= args.rollout_index_start < args.rollout_count):
        p.error("--rollout-index-start must be in [0, --rollout-count)")
    if args.rollout_max_sim < 0:
        p.error("--rollout-max-sim must be >= 0")
    if args.rollout_count > 1 and args.rollout_max_sim <= 0:
        p.error("--rollout-max-sim must be > 0 when --rollout-count > 1")
    if args.clock_stall_wall < 0:
        p.error("--clock-stall-wall must be >= 0")
    if args.camera_discovery_timeout <= 0:
        p.error("--camera-discovery-timeout must be > 0")
    if args.camera_max_age_sim <= 0:
        p.error("--camera-max-age-sim must be > 0")
    if args.camera_sync_tolerance_sim < 0:
        p.error("--camera-sync-tolerance-sim must be >= 0")
    if args.n_action_steps_override is not None and args.n_action_steps_override <= 0:
        p.error("--n-action-steps-override must be > 0")
    if args.action_ensemble_samples <= 0:
        p.error("--action-ensemble-samples must be > 0")

    if args.mode == "inference":
        if not args.checkpoint:
            p.error("--checkpoint is required for --mode inference")
        run_inference(args)
    else:
        run_ros(args)


def main():
    """Use the pre-async v2 deployment path by default.

    The async implementation remains in this module for comparison, but the
    legacy policy-owned action queue avoids hard replacement of partial chunks.
    """
    if os.environ.get("TASK2_PI05_USE_ASYNC") == "1":
        return main_async()

    from task2_pi05_eval_node_v2 import main as legacy_main

    return legacy_main()


if __name__ == "__main__":
    main()
