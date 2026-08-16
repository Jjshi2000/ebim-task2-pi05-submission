#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import pickle
import random
import socket
import struct
import threading
import time


HOST = "127.0.0.1"
PORT = 8765

DEFAULT_TASK = "Pick up the thermal pad and place it on the target RAM board."
DEFAULT_ROBOT_TYPE = "fr3duo_mobile_task2"

GRIPPER_CLOSED_RAD = 0.8

LEFT_JOINTS = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
SPINE_JOINT = "franka_spine_vertical_joint"
LEFT_GRIPPER_DRIVER = "left_right_finger_joint"
RIGHT_GRIPPER_DRIVER = "right_right_finger_joint"

CLOCK_TOPIC = "/isaac/clock"
FULL_STATES_TOPIC = "/isaac/joint_states_full"
ODOM_TOPIC = "/isaac/odom"
LEFT_EE_TOPIC = "/isaac/left_ee_pose"
RIGHT_EE_TOPIC = "/isaac/right_ee_pose"
SCENE_RESET_TOPIC = "/isaac/task2/scene_reset"

LEFT_ARM_CMD_TOPIC = "/isaac/left_joint_commands"
RIGHT_ARM_CMD_TOPIC = "/isaac/right_joint_commands"
LEFT_GRIPPER_CMD_TOPIC = "/isaac/left_robotiq_joint_commands"
RIGHT_GRIPPER_CMD_TOPIC = "/isaac/right_robotiq_joint_commands"
SPINE_CMD_TOPIC = "/isaac/spine_joint_commands"


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
    ),
    "wrist_right": (
        "/isaac/wrist_right_camera/image_raw",
        "/isaac/right_wrist_camera/image_raw",
        "/isaac/wrist_right/image_raw",
        "/isaac/right_wrist/image_raw",
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
# envelope.  The arm limits include 0.03 rad beyond the observed extrema.  The
# per-step slew limits are the rounded 99.5th percentiles of within-episode
# absolute action deltas, so isolated model jumps cannot become instantaneous
# joint-target jumps while normal demonstrated motion remains unchanged.
DEMO_LEFT_ARM_MIN = (-1.242, -1.435, -0.030, -2.734, -0.030, 1.402, 0.713)
DEMO_LEFT_ARM_MAX = (0.076, -0.203, 1.775, -1.362, 1.306, 2.535, 1.841)
DEMO_RIGHT_ARM_MIN = (-0.607, -1.857, -2.769, -3.107, -2.117, 1.540, -2.034)
DEMO_RIGHT_ARM_MAX = (2.876, 0.087, 0.081, -0.101, 1.943, 4.641, 0.815)

DEMO_LEFT_ARM_MAX_STEP = (0.010, 0.012, 0.016, 0.010, 0.010, 0.012, 0.010)
DEMO_RIGHT_ARM_MAX_STEP = (0.050, 0.060, 0.090, 0.045, 0.065, 0.070, 0.060)
DEMO_SPINE_MAX_STEP = 0.020


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
    ):
        if step_scale <= 0:
            raise ValueError("step_scale must be > 0")
        self.spine_min = float(spine_min)
        self.spine_max = float(spine_max)
        self.step_scale = float(step_scale)
        self.last_action = None
        self.total_limited_steps = 0

    def reset(self) -> None:
        self.last_action = None

    @staticmethod
    def _measured_action(joints: dict[str, float]) -> list[float]:
        required = (
            LEFT_JOINTS
            + RIGHT_JOINTS
            + [LEFT_GRIPPER_DRIVER, RIGHT_GRIPPER_DRIVER, SPINE_JOINT]
        )
        missing = [name for name in required if name not in joints]
        if missing:
            raise RuntimeError(
                "cannot initialize action safety without measured joints: "
                + ", ".join(missing)
            )

        left_open = _clip(
            1.0 - float(joints[LEFT_GRIPPER_DRIVER]) / GRIPPER_CLOSED_RAD,
            0.0,
            1.0,
        )
        right_open = _clip(
            1.0 - float(joints[RIGHT_GRIPPER_DRIVER]) / GRIPPER_CLOSED_RAD,
            0.0,
            1.0,
        )
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
        if len(action) != 20 or not all(math.isfinite(float(x)) for x in action):
            raise ValueError("invalid 20D action")
        if self.last_action is None:
            self.last_action = self._measured_action(measured_joints)

        raw = [float(x) for x in action]
        bounded = list(raw)
        bounded[0:3] = [0.0, 0.0, 0.0]

        arm_bounds = list(zip(DEMO_LEFT_ARM_MIN, DEMO_LEFT_ARM_MAX)) + list(
            zip(DEMO_RIGHT_ARM_MIN, DEMO_RIGHT_ARM_MAX)
        )
        range_limited = []
        for offset, (lower, upper) in enumerate(arm_bounds, start=3):
            bounded[offset] = _clip(raw[offset], lower, upper)
            if not math.isclose(bounded[offset], raw[offset], abs_tol=1.0e-12):
                range_limited.append(offset)

        bounded[17] = _clip(raw[17], 0.0, 1.0)
        bounded[18] = _clip(raw[18], 0.0, 1.0)
        bounded[19] = _clip(raw[19], self.spine_min, self.spine_max)
        for index in (17, 18, 19):
            if not math.isclose(bounded[index], raw[index], abs_tol=1.0e-12):
                range_limited.append(index)

        safe = list(bounded)
        max_steps = (
            list(DEMO_LEFT_ARM_MAX_STEP)
            + list(DEMO_RIGHT_ARM_MAX_STEP)
            + [DEMO_SPINE_MAX_STEP]
        )
        slew_indices = list(range(3, 17)) + [19]
        rate_limited = []
        for index, max_step in zip(slew_indices, max_steps):
            previous = float(self.last_action[index])
            limit = float(max_step) * self.step_scale
            safe[index] = _clip(bounded[index], previous - limit, previous + limit)
            if not math.isclose(safe[index], bounded[index], abs_tol=1.0e-12):
                rate_limited.append(index)

        raw_max_arm_delta = max(
            abs(raw[index] - float(self.last_action[index]))
            for index in range(3, 17)
        )
        safe_max_arm_delta = max(
            abs(safe[index] - float(self.last_action[index]))
            for index in range(3, 17)
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
    if state_shape != (37,):
        raise ValueError(f"Task2 requires checkpoint observation.state shape (37,), got {state_shape}")

    action_feature = getattr(config, "action_feature", None)
    action_shape = _feature_shape(action_feature)
    if action_shape != (20,):
        raise ValueError(f"Task2 requires checkpoint action shape (20,), got {action_shape}")

    return {
        "policy": policy,
        "config": config,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": torch.device(device),
        "image_keys": image_keys,
        "expected_shapes": expected_shapes,
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
    from lerobot.policies.utils import prepare_observation_for_inference

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

    # n_action_steps is an inference execution horizon only. PI0.5 is trained
    # against chunk_size actions; select_action() merely chooses how many of
    # that predicted chunk are queued for execution. Allow a deployment-time
    # override without mutating the checkpoint on disk.
    if args.n_action_steps_override is not None:
        n = int(args.n_action_steps_override)
        chunk_size = int(getattr(config, "chunk_size", 0))
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
    print(f"  num_inference_steps: {getattr(config, 'num_inference_steps', None)}")
    print(f"  task: {args.task}")
    print(f"  robot_type: {args.robot_type}")
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
                "state_shape": [37],
                "action_shape": [20],
                "task": args.task,
                "robot_type": args.robot_type,
                "chunk_size": int(getattr(config, "chunk_size", 0)),
                "n_action_steps": int(getattr(config, "n_action_steps", 0)),
            },
        )

        first_image_log = True

        try:
            while True:
                req = recv_obj(conn)
                op = req.get("op")

                if op == "reset":
                    reset_seed = int(req.get("seed", args.seed))
                    _reset_pipeline(
                        policy,
                        preprocessor,
                        postprocessor,
                        seed=reset_seed if not args.no_reseed_on_reset else None,
                    )
                    send_obj(conn, {"ok": True, "seed": reset_seed})
                    continue

                if op not in ("infer", "cached"):
                    raise ValueError(f"unknown request op={op!r}")

                t0 = time.perf_counter()
                queue_before = _pi05_action_queue_len(policy)

                # While PI0.5's internal action queue is non-empty, select_action()
                # ignores the observation and only pops the next cached action.
                # Do not resend/decode three RGB frames or gate 30 Hz execution on
                # camera synchronization for those cached steps.
                if op == "cached":
                    if queue_before is None or queue_before <= 0:
                        send_obj(
                            conn,
                            {
                                "ok": False,
                                "need_observation": True,
                                "error": "PI0.5 action queue is empty",
                            },
                        )
                        continue
                    with torch.inference_mode():
                        action = policy.select_action({})
                        action = postprocessor(action)
                    if not torch.is_tensor(action):
                        raise TypeError(
                            f"postprocessor returned {type(action).__name__}, expected torch.Tensor"
                        )
                    if action.ndim == 1:
                        action = action.unsqueeze(0)
                    if tuple(action.shape) != (1, 20):
                        raise ValueError(
                            f"expected postprocessed Task2 action [1,20], got {tuple(action.shape)}"
                        )
                    a = action[0].detach().float().cpu().numpy()
                    if not np.isfinite(a).all():
                        raise ValueError("non-finite PI0.5 action")
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    queue_after = _pi05_action_queue_len(policy)
                    send_obj(
                        conn,
                        {
                            "ok": True,
                            "action": a.tolist(),
                            "inference_ms": elapsed_ms,
                            "replanned": False,
                            "queue_before": queue_before,
                            "queue_remaining": queue_after,
                        },
                    )
                    continue

                state = np.asarray(req["state"], dtype=np.float32)
                if state.shape != (37,) or not np.isfinite(state).all():
                    raise ValueError(f"invalid Task2 state: shape={state.shape}, finite={np.isfinite(state).all()}")

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
                    # Keep HWC uint8 here. prepare_observation_for_inference performs
                    # exactly the LeRobot uint8->[0,1], HWC->CHW and batching path.
                    raw_observation[key] = image

                if first_image_log:
                    print("first live camera shapes (H,W,C):")
                    for key in image_keys:
                        print(f"  {key}: {shape_log[key]}")
                    first_image_log = False

                replanned = (queue_before == 0) if queue_before is not None else None

                autocast_ctx = (
                    torch.autocast(device_type=device.type)
                    if device.type == "cuda" and bool(getattr(config, "use_amp", False))
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    observation = prepare_observation_for_inference(
                        raw_observation,
                        device=device,
                        task=args.task,
                        robot_type=args.robot_type,
                    )
                    observation = preprocessor(observation)
                    action = policy.select_action(observation)
                    action = postprocessor(action)

                if not torch.is_tensor(action):
                    raise TypeError(f"postprocessor returned {type(action).__name__}, expected torch.Tensor")
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                if tuple(action.shape) != (1, 20):
                    raise ValueError(f"expected postprocessed Task2 action [1,20], got {tuple(action.shape)}")

                a = action[0].detach().float().cpu().numpy()
                if not np.isfinite(a).all():
                    raise ValueError("non-finite PI0.5 action")

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                queue_after = _pi05_action_queue_len(policy)
                if args.debug_actions and replanned:
                    print(
                        "REPLAN action0="
                        + np.array2string(a, precision=4, suppress_small=False),
                        flush=True,
                    )
                send_obj(
                    conn,
                    {
                        "ok": True,
                        "action": a.tolist(),
                        "inference_ms": elapsed_ms,
                        "replanned": replanned,
                        "queue_before": queue_before,
                        "queue_remaining": queue_after,
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
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    overrides = parse_camera_topic_overrides(args.camera_topic)
    image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )

    class NodeImpl(Node):
        def __init__(self):
            super().__init__("task2_policy_ros_adapter")

            self.lock = threading.Lock()
            self.cond = threading.Condition(self.lock)

            self.sim_time = None
            self.last_clock = None
            self.joints = {}
            self.odom = None
            self.ee = {"left": None, "right": None}

            self.required_image_keys = []
            self.feature_to_logical = {}
            self.camera_topics = {}
            self.camera_frames = {}
            self.camera_subscriptions = []

            self.reset_pending = True
            self.stop = False
            self.last_wait_log_wall = 0.0
            self.last_safety_log_wall = 0.0
            self.action_safety = Task2ActionSafetyFilter(
                spine_min=args.spine_min,
                spine_max=args.spine_max,
                step_scale=args.action_safety_step_scale,
            )

            self.pubs = {
                "la": self.create_publisher(
                    JointState, LEFT_ARM_CMD_TOPIC, 10
                ),
                "ra": self.create_publisher(
                    JointState, RIGHT_ARM_CMD_TOPIC, 10
                ),
                "lg": self.create_publisher(
                    JointState, LEFT_GRIPPER_CMD_TOPIC, 10
                ),
                "rg": self.create_publisher(
                    JointState, RIGHT_GRIPPER_CMD_TOPIC, 10
                ),
                "sp": self.create_publisher(
                    JointState, SPINE_CMD_TOPIC, 10
                ),
            }

            self.create_subscription(
                Clock, CLOCK_TOPIC, self.cb_clock, 10
            )
            self.create_subscription(
                JointState, FULL_STATES_TOPIC, self.cb_joints, 10
            )
            self.create_subscription(
                Odometry, ODOM_TOPIC, self.cb_odom, 10
            )
            self.create_subscription(
                PoseStamped,
                LEFT_EE_TOPIC,
                lambda m: self.cb_ee("left", m),
                10,
            )
            self.create_subscription(
                PoseStamped,
                RIGHT_EE_TOPIC,
                lambda m: self.cb_ee("right", m),
                10,
            )
            self.create_subscription(
                String, SCENE_RESET_TOPIC, self.cb_reset, 10
            )

            # Connect before starting the control thread so the checkpoint can
            # tell us which camera(s) it needs.
            sock, hello = self.connect()
            if sock is None:
                raise RuntimeError("rclpy stopped before inference connection")

            self.configure_checkpoint_cameras(hello)
            self.server_n_action_steps = int(hello.get("n_action_steps") or 0)
            self.server_queue_remaining = 0
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
                    "action safety enabled: demonstrated arm ranges + p99.5 target slew limits; "
                    f"step_scale={args.action_safety_step_scale:g}"
                )

        # -------------------------- callbacks --------------------------

        def clear_camera_frames_locked(self):
            for logical in self.camera_frames:
                self.camera_frames[logical] = None

        def cb_clock(self, m):
            t = m.clock.sec + m.clock.nanosec * 1e-9
            with self.cond:
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
                self.camera_frames[logical] = (m, self.sim_time, stamp_sim)

        def cb_reset(self, _m):
            with self.cond:
                self.reset_pending = True
                self.action_safety.reset()
                self.clear_camera_frames_locked()
                self.cond.notify_all()

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
                    if tuple(hello.get("state_shape") or ()) != (37,):
                        raise RuntimeError(
                            f"PI0.5 server state contract mismatch: {hello.get('state_shape')}"
                        )
                    if tuple(hello.get("action_shape") or ()) != (20,):
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
                    image_qos,
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
            if tuple(hello.get("state_shape") or ()) != (37,) or tuple(hello.get("action_shape") or ()) != (20,):
                raise RuntimeError(f"reconnected PI0.5 server changed Task2 state/action contract: {hello}")
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

                s = build_state(
                    dict(self.joints),
                    {
                        k: (
                            None if v is None else list(v)
                        )
                        for k, v in self.ee.items()
                    },
                    self.odom,
                )
                if s is None:
                    self.maybe_log_wait(
                        "waiting for complete 37D robot state"
                    )
                    return None

                packets = {}
                arrival_times = []
                header_times = []

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

                    age = self.sim_time - arrival_sim
                    if age > args.camera_max_age_sim:
                        self.maybe_log_wait(
                            f"camera {key} stale by {age:.3f}s sim"
                        )
                        return None

                    packets[key] = self.image_packet(msg)
                    arrival_times.append(arrival_sim)
                    header_times.append(header_sim)

                if all(t is not None for t in header_times):
                    sync_times = [float(t) for t in header_times]
                    sync_basis = "header"
                else:
                    sync_times = arrival_times
                    sync_basis = "arrival"

                if (
                    len(sync_times) > 1
                    and max(sync_times) - min(sync_times)
                    > args.camera_sync_tolerance_sim
                ):
                    skew = max(sync_times) - min(sync_times)
                    self.maybe_log_wait(
                        f"waiting for synchronized cameras; "
                        f"latest skew={skew:.3f}s sim ({sync_basis})"
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
            if (
                len(a) != 20
                or not all(math.isfinite(float(x)) for x in a)
            ):
                raise ValueError("invalid 20D action")

            if (
                max(abs(float(x)) for x in a[0:3])
                > args.base_warn_threshold
            ):
                self.get_logger().warning(
                    f"nonzero base action {a[0:3]} ignored for fixpos"
                )

            if not args.disable_action_safety:
                with self.lock:
                    a, safety = self.action_safety.filter(
                        list(a), dict(self.joints)
                    )
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

            self.pub(
                self.pubs["la"],
                LEFT_JOINTS,
                a[3:10],
            )
            self.pub(
                self.pubs["ra"],
                RIGHT_JOINTS,
                a[10:17],
            )

            lo = max(0.0, min(1.0, float(a[17])))
            ro = max(0.0, min(1.0, float(a[18])))

            self.pub(
                self.pubs["lg"],
                [LEFT_GRIPPER_DRIVER],
                [(1.0 - lo) * GRIPPER_CLOSED_RAD],
            )
            self.pub(
                self.pubs["rg"],
                [RIGHT_GRIPPER_DRIVER],
                [(1.0 - ro) * GRIPPER_CLOSED_RAD],
            )

            spine = max(
                args.spine_min,
                min(args.spine_max, float(a[19])),
            )
            self.pub(
                self.pubs["sp"],
                [SPINE_JOINT],
                [spine],
            )

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

                    if sim is None:
                        continue

                    if reset:
                        with self.lock:
                            self.action_safety.reset()
                        send_obj(sock, {"op": "reset", "seed": args.seed})
                        rep = recv_obj(sock)
                        if not rep.get("ok"):
                            raise RuntimeError(rep.get("error"))

                        self.server_queue_remaining = 0
                        next_sim = sim + args.start_delay_sim
                        self.get_logger().info(
                            f"policy reset; start @ sim {next_sim:.3f}"
                        )
                        continue

                    if next_sim is None:
                        next_sim = sim + args.start_delay_sim
                        continue

                    if sim + 1e-9 < next_sim:
                        continue

                    t0 = time.perf_counter()
                    if self.server_queue_remaining > 0:
                        send_obj(sock, {"op": "cached"})
                    else:
                        snap = self.snap()
                        if snap is None:
                            continue
                        send_obj(
                            sock,
                            {
                                "op": "infer",
                                "state": snap["state"],
                                "images": snap["images"],
                            },
                        )
                    rep = recv_obj(sock)
                    if not rep.get("ok"):
                        if rep.get("need_observation"):
                            self.server_queue_remaining = 0
                            continue
                        raise RuntimeError(rep.get("error"))

                    self.server_queue_remaining = int(rep.get("queue_remaining") or 0)
                    self.apply(rep["action"])

                    if rep.get("replanned") and self.server_n_action_steps > 0:
                        horizon_ms = 1000.0 * self.server_n_action_steps / args.fps
                        infer_ms = float(rep.get("inference_ms", 0.0))
                        now_wall = time.monotonic()
                        if infer_ms > horizon_ms and now_wall - self.last_latency_warn_wall >= 5.0:
                            self.get_logger().warning(
                                f"PI0.5 replan latency {infer_ms:.1f}ms exceeds action horizon "
                                f"{horizon_ms:.1f}ms ({self.server_n_action_steps} steps @ {args.fps:g}Hz). "
                                "Synchronous inference may miss real-time Task2 control deadlines; "
                                "consider RTC after validating checkpoint behavior."
                            )
                            self.last_latency_warn_wall = now_wall

                    next_sim += period

                    with self.lock:
                        cur = self.sim_time

                    if (
                        cur is not None
                        and cur - next_sim > 2 * period
                    ):
                        next_sim = cur + period

                    if int(sim * 2) != int((sim - period) * 2):
                        self.get_logger().info(
                            f"infer={rep.get('inference_ms', 0):.1f}ms "
                            f"roundtrip="
                            f"{(time.perf_counter() - t0) * 1000:.1f}ms "
                            f"cams={len(self.required_image_keys)} "
                            f"spine={rep['action'][19]:.3f}"
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

def main():
    p = argparse.ArgumentParser(
        description=(
            "EBiM Task2 PI0.5 inference server + ROS2 Jazzy adapter. "
            "Inference mode loads LeRobot full or PEFT/LoRA PI0.5 checkpoints; "
            "ROS mode bridges the verified Task2 37D observation / 20D action contract."
        )
    )

    p.add_argument("--mode", choices=("inference", "ros"), required=True)
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
            "inference-only PI0.5 execution horizon override; must be <= chunk_size. "
            "Useful when replanning is slower than n_action_steps/fps."
        ),
    )
    p.add_argument(
        "--debug-actions",
        action="store_true",
        help="print the first postprocessed 20D action of every newly planned chunk",
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
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start-delay-sim", type=float, default=1.0)
    p.add_argument("--spine-min", type=float, default=0.0)
    p.add_argument("--spine-max", type=float, default=0.54)
    p.add_argument("--base-warn-threshold", type=float, default=0.02)
    p.add_argument(
        "--disable-action-safety",
        action="store_true",
        help=(
            "disable demonstrated-range and target-slew protection (diagnostics only)"
        ),
    )
    p.add_argument(
        "--action-safety-step-scale",
        type=float,
        default=1.0,
        help=(
            "scale the demonstration-derived arm/spine per-command slew limits"
        ),
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
    if args.camera_discovery_timeout <= 0:
        p.error("--camera-discovery-timeout must be > 0")
    if args.camera_max_age_sim <= 0:
        p.error("--camera-max-age-sim must be > 0")
    if args.camera_sync_tolerance_sim < 0:
        p.error("--camera-sync-tolerance-sim must be >= 0")
    if args.n_action_steps_override is not None and args.n_action_steps_override <= 0:
        p.error("--n-action-steps-override must be > 0")

    if args.mode == "inference":
        if not args.checkpoint:
            p.error("--checkpoint is required for --mode inference")
        run_inference(args)
    else:
        run_ros(args)


if __name__ == "__main__":
    main()
