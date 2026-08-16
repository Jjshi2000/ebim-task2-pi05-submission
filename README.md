# EBiM Task 2 - PI0.5 Full Fine-Tune

This repository is the runnable Phase I submission for **Task 2 - Deformable
Material Handling (Thermal Pad Placement)**. It deploys a fully fine-tuned
LeRobot PI0.5 policy at training step 20,000 against the official Isaac Sim ROS
topic contract.

The model weights are supplementary material hosted separately in a private
Hugging Face model repository. This public repository contains the complete
container build recipe, policy runtime, ROS adapter, and integration guide.

## Policy contract

| Item | Value |
| --- | --- |
| Policy | LeRobot PI0.5, full fine-tune (`peft=false`) |
| Training checkpoint | 20,000 steps |
| Language task | `Pick up the thermal pad and place it on the target RAM board.` |
| State | 37 dimensions |
| Action | 20 dimensions |
| Action chunk | 50 steps |
| Deployment horizon | 50 steps with asynchronous prefetch |
| Cameras | head, left wrist, right wrist |
| Control rate | 30 Hz |

Expected image features and ROS topics:

| Checkpoint feature | Shape | Preferred ROS topic |
| --- | --- | --- |
| `observation.images.head` | `3 x 720 x 1280` | `/isaac/head_camera/image_raw` |
| `observation.images.wrist_left` | `3 x 480 x 848` | `/isaac/left_wrist_camera/image_raw` |
| `observation.images.wrist_right` | `3 x 480 x 848` | `/isaac/right_wrist_camera/image_raw` |

The adapter discovers compatible aliases from the ROS graph, but it refuses to
silently resize a camera whose geometry differs from the checkpoint contract.

## Requirements

- Linux x86_64
- NVIDIA GPU with sufficient memory for the 9.35 GB full checkpoint
- NVIDIA driver compatible with the PyTorch CUDA runtime
- Docker Engine and NVIDIA Container Toolkit
- Official EBiM Task 2 Isaac Sim scene publishing the ROS topics below
- Access to the private Hugging Face model repository, or a local checkpoint
  directory mounted into the container

## Build

```bash
docker build --pull -t ebim-task2-pi05:20k .
```

The Dockerfile starts from `ros:jazzy-ros-base` and installs LeRobot from the
exact training revision:

```text
22bd7a2f489b367d8df42de803b1e8c4ca63a3f9
```

No model weight, dataset, token, or machine-specific path is baked into the
image.

## Run with private Hugging Face weights

The evaluator's Hugging Face account must first be granted read access to the
private model repository. Pass an access token belonging to that authorized
account at runtime; never add a token to this repository or Docker image.

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e MODEL_REPO=YOUR_HF_ORG/ebim-task2-pi05-fullft-20k \
  -e HF_TOKEN \
  -v ebim-hf-cache:/cache/huggingface \
  -v ebim-models:/models \
  ebim-task2-pi05:20k
```

The entrypoint downloads the complete checkpoint to
`/models/pi05-task2-fullft-20k`, validates the required processor/tokenizer
artifacts, starts PI0.5 inference, waits for port `8765`, and then starts the
ROS adapter. The policy runs continuously. It does not stop on elapsed time or
preliminary grasp detection.

## Run with a pre-downloaded checkpoint

The mounted directory must contain the *contents* of LeRobot's
`pretrained_model` directory, including `config.json`, `model.safetensors`, both
processor JSON files, normalization statistics, and the `tokenizer/` folder.

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e MODEL_DIR=/models/pi05-task2-fullft-20k \
  -v /absolute/path/to/pretrained_model:/models/pi05-task2-fullft-20k:ro \
  ebim-task2-pi05:20k
```

The expected `model.safetensors` checksum is:

```text
661e4995cda5421ed23258dea6a51c751f23dce25095f0614e44af7fbd5d40a8
```

## Runtime configuration

The default container mode starts inference and ROS together. Supported
environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODE` | `all` | `all`, `inference`, or `ros` |
| `MODEL_REPO` | empty | Private Hugging Face model repository |
| `MODEL_DIR` | `/models/pi05-task2-fullft-20k` | Download or mounted checkpoint directory |
| `HF_TOKEN` | empty | Runtime-only token for an authorized HF account |
| `DEVICE` | `cuda` | PI0.5 inference device |
| `INFERENCE_HOST` | `127.0.0.1` | Policy socket host |
| `INFERENCE_PORT` | `8765` | Policy socket port |
| `N_ACTION_STEPS` | `50` | Executed action horizon and replan interval |
| `FPS` | `30` | ROS command loop rate |
| `TASK` | official Task 2 instruction | Language instruction |

For split deployment, start an inference container with `MODE=inference`, then
start a ROS container with `MODE=ros` and set `INFERENCE_HOST` to the reachable
inference host.

## ROS interface

The adapter consumes the official Task 2 observation streams, including:

- `/isaac/clock`
- `/isaac/joint_states_full`
- `/isaac/odom`
- `/isaac/left_ee_pose`
- `/isaac/right_ee_pose`
- `/isaac/task2/pad_points`
- the three RGB topics listed above

It publishes:

- `/isaac/left_joint_commands`
- `/isaac/right_joint_commands`
- `/isaac/left_robotiq_joint_commands`
- `/isaac/right_robotiq_joint_commands`
- `/isaac/spine_joint_commands`

At startup it requests a deterministic scene reset through
`/isaac/task2/scene_reset_request` and waits for acknowledgement. Action targets
are constrained to demonstrated joint ranges and per-step slew limits. Long
action chunks are prefetched before the queue is empty to hide inference
latency.

## Troubleshooting

- `checkpoint is incomplete`: mount or upload the entire `pretrained_model`
  directory, not only `model.safetensors`.
- `401` or `403` from Hugging Face: grant the evaluator account access and use
  that account's runtime token. Do not send personal credentials to organizers.
- missing camera topic: confirm the official Isaac scene was launched with the
  robot RGB cameras and ROS bridge enabled.
- camera shape mismatch: use the exact camera resolutions in the policy
  contract; the adapter intentionally does not silently resize inputs.
- action queue errors: confirm `TASK2_PI05_USE_ASYNC=1` is present in both
  inference and ROS processes. It is enabled by default in this image.

## Reproducibility

- EBiM benchmark base commit:
  `0004645a4b8843f0e04a5ca531fce0598e058910`
- LeRobot commit:
  `22bd7a2f489b367d8df42de803b1e8c4ca63a3f9`
- Full checkpoint manifest: [`model-manifest.json`](model-manifest.json)

The checkpoint stores its tokenizer and exact QUANTILES normalization and
unnormalization processors. Deployment loads those saved artifacts rather than
rebuilding statistics from an external dataset.

## License

See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

