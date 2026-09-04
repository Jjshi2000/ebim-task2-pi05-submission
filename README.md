# EBiM Phase II Task 2

This repository deploys the 30,000-step PI0.5 full fine-tune for the official
Task 2 real robot. It approaches the table using the mobile-base ROS 2 topics,
then runs the manipulation policy continuously without a rollout timeout or an
early success exit.

The checkpoint is stored separately in
`junjie-jjs/ebim-task2-pi05-fullft-30k`. It was trained on the released Munich
real-robot dataset and uses its exact 42D observation and 17D action layouts.

## Build

```bash
docker build --pull -t ebim-task2-pi05:30k .
```

The image is based on `ros:jazzy-ros-base`, installs the pinned LeRobot
revision, and contains no checkpoint, dataset, token, Isaac Sim installation,
or machine-specific path.

## Real-robot contract

The adapter uses the official ROS 2 interface directly:

| Signal | Topic |
| --- | --- |
| Left/right arm position | `/left/right/franka_robot_state_broadcaster/measured_joint_states` |
| Left/right external wrench | `/left/right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` |
| Left/right gripper state | `/left/right/gripper/joint_states` |
| Head RGB | `/head_camera/zed_node/rgb/color/rect/image` |
| Wrist RGB | `/wrist_camera_left/right/camera/color/image_raw` |
| Arm commands | `/left/right/gello/joint_states` |
| Gripper commands | `/left/right/gripper/gripper_client/target_gripper_width_percent` |
| Spine command | `/spine/target_height` |

The 42D state order is left arm 7, left gripper 1, left external joint torque
7, left wrench 6, right arm 7, right gripper 1, right external joint torque 7,
and right wrench 6. The released converted training data contains constant zero
left-gripper and external-joint-torque fields, so the adapter deliberately
reproduces those zeros. Live 6D wrenches come from the official
`WrenchStamped` topics.

The 17D action order is left arm 7, left gripper 1, right arm 7, right gripper
1, and spine 1. The recorded spine target is exactly `434.0`; deployment holds
that demonstrated value. Arm limits and per-command slew limits are derived
from the 238 released demonstrations at 20 Hz.

## Required preflight

Start all robot drivers, cameras, and controllers, then run the read-only
preflight. It creates no command publishers:

```bash
docker run --rm --network host \
  --entrypoint bash ebim-task2-pi05:30k \
  -lc 'source /opt/ros/jazzy/setup.bash && \
       python3 /app/task2_real_preflight.py --timeout 15'
```

Do not continue until it exits with status 0 and reports `"ok": true`. Resolve
every missing topic, unexpected type, image geometry mismatch, non-finite
sample, and missing controller subscriber first.

## Dry-run inference

Cache or mount the checkpoint before the test slot. For an access-controlled
repository, provide `HF_TOKEN` only through the runtime environment and never
store it in the image or repository. Run the complete observation and inference
path without navigation and without creating arm, gripper, or spine command
publishers:

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ROS_PROFILE=real \
  -e DRY_RUN=1 \
  -e HF_TOKEN \
  -v ebim-hf-cache:/cache/huggingface \
  -v ebim-models:/models \
  ebim-task2-pi05:30k
```

Confirm the log reports `42D/17D`, the three expected image geometries, fresh
real state, finite actions, and no inference queue exhaustion. `DRY_RUN=1`
also suppresses automatic base navigation.

## Navigation calibration

`NAV_FORWARD_DISTANCE` is relative to the startup odometry pose. The navigator
has odometry feedback but no obstacle avoidance. Clear the path, keep an
operator on the emergency stop, and test it separately at 5 cm first:

```bash
docker run --rm --network host \
  --entrypoint bash ebim-task2-pi05:30k \
  -lc 'source /opt/ros/jazzy/setup.bash && \
       python3 /app/task2_base_nav.py \
         --forward-distance 0.05 --max-linear 0.05 --max-angular 0.10'
```

Increase the distance only after verifying direction, odometry, braking, and
the measured table-front target. Do not assume the old example value `0.80 m`
is valid at a new site.

## Live rollout

Place both arms and the spine in the same initial configuration used by the
demonstrations. Keep one operator exclusively on the physical emergency stop.
After preflight, dry run, and navigation calibration pass, run with the measured
distance:

```bash
docker run --rm --gpus all --network host --ipc host \
  -e ROS_PROFILE=real \
  -e NAV_FORWARD_DISTANCE=MEASURED_DISTANCE_METRES \
  -e HF_TOKEN \
  -v ebim-hf-cache:/cache/huggingface \
  -v ebim-models:/models \
  ebim-task2-pi05:30k
```

The default controller rate is 20 Hz and the default PI0.5 replanning interval
is the checkpoint's `10` actions. The action safety filter remains enabled.

## Runtime variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `all` | `all`, `inference`, or `ros` |
| `ROS_PROFILE` | `real` | `real` or legacy `isaac` topic wiring |
| `MODEL_REPO` | `junjie-jjs/ebim-task2-pi05-fullft-30k` | Hugging Face model repository |
| `MODEL_REVISION` | `af799f06e59be97a7a1b2603d610d22669305cf3` | Pinned Hugging Face model commit |
| `MODEL_DIR` | `/models/pi05-task2-fullft-30k` | Mounted/downloaded checkpoint |
| `N_ACTION_STEPS` | `10` | PI0.5 closed-loop replanning interval |
| `FPS` | `20` | Policy command rate matching the dataset |
| `DRY_RUN` | `0` | Set to `1` to disable all policy command publishers and navigation |
| `NAV_FORWARD_DISTANCE` | `0` | Calibrated startup-relative approach distance in metres |
| `REAL_SPINE_HEIGHT` | `434.0` | Demonstrated `/spine/target_height` value |
| `HF_TOKEN` | empty | Runtime-only access token when the model repository requires it |

The downloader verifies that the checkpoint contains the model, config,
processor JSON files, normalization states, and tokenizer before inference.

## Reproducibility

- LeRobot commit: `22bd7a2f489b367d8df42de803b1e8c4ca63a3f9`
- PI0.5 contract: state `[42]`, action `[17]`, three RGB inputs, chunk size 50
- Training cadence: 20 Hz; deployment replanning interval: 10
- Model details: [`model-manifest.json`](model-manifest.json)

Do not commit model weights, local paths, credentials, or private logs.
