# EBiM Phase II Task 2

This is a public, reproducible Docker submission for Task 2 (thermal-pad
handling). It first approaches the table with the official mobile-base ROS 2
topics, then starts a PI0.5 full-fine-tune policy. The policy runs continuously;
there is no rollout timeout or early success exit.

The submitted checkpoint is the 30,000-step full fine-tune. Weights are kept
outside Git and are downloaded at runtime from the public Hugging Face model
repository configured with `MODEL_REPO`.

## Build

```bash
docker build --pull -t ebim-task2-pi05:30k .
```

The image is based on `ros:jazzy-ros-base`, installs the pinned LeRobot
revision, and contains no checkpoint, dataset, token, Isaac Sim installation,
or machine-specific path.

## Run on the real EBiM platform

Start the robot's ROS 2 drivers and cameras first. Set `MODEL_REPO` to the
public Hugging Face repository containing the complete LeRobot
`pretrained_model` directory, then run:

```bash
docker run --rm --gpus all --network host --ipc host \
  -e MODEL_REPO=<owner>/<public-30k-model> \
  -e NAV_FORWARD_DISTANCE=0.80 \
  -e ROS_PROFILE=real \
  -v ebim-hf-cache:/cache/huggingface \
  -v ebim-models:/models \
  ebim-task2-pi05:30k
```

`NAV_FORWARD_DISTANCE` is measured from the startup pose in the odometry frame.
For a calibrated fixed target, pass `--target-x/--target-y/--target-yaw` by
using `NAV_ARGS` in a small wrapper or run `/app/task2_base_nav.py` directly.
The navigator publishes `geometry_msgs/msg/TwistStamped` on
`/swerve_drive_controller/cmd_vel` and stops before policy control starts.

The official real-robot input topics are wired by default:

| Signal | Topic |
| --- | --- |
| Base odometry | `/swerve_drive_controller/odom` |
| Left/right arm state | `/left/right/franka_robot_state_broadcaster/measured_joint_states` |
| Spine state | `/spine/joint_states` |
| Gripper state | `/left/right/gripper/joint_states` |
| Head RGB | `/head_camera/zed_node/rgb/color/rect/image` |
| Wrist RGB | `/wrist_camera_left/right/camera/color/image_raw` |

The official Franka impedance controller consumes arm targets on
`/left/gello/joint_states` and `/right/gello/joint_states` (the same topics used
by the released teleoperation stack). These can be overridden with
`--real-left-arm-command` and `--real-right-arm-command` if the testbed uses a
different controller.
Gripper and spine commands use the official `std_msgs/msg/Float32` topics by
default. If the platform supplies calibrated EE pose topics, pass
`--real-left-ee-topic` and `--real-right-ee-topic`; otherwise the adapter keeps
the required 37D contract with explicit zero EE placeholders.

## Isaac Sim or split deployment

For the official Isaac scene use `-e ROS_PROFILE=isaac`. To run inference and
ROS separately, set `MODE=inference` in one container and `MODE=ros` in another,
sharing port `8765` over the host network. The Isaac profile retains the
official `/isaac/*` topics and deterministic scene-reset handshake.

## Runtime variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `all` | `all`, `inference`, or `ros` |
| `ROS_PROFILE` | `real` | `real` or `isaac` topic wiring |
| `MODEL_REPO` | empty | Public Hugging Face model repository |
| `MODEL_DIR` | `/models/pi05-task2-fullft-30k` | Mounted/downloaded checkpoint |
| `N_ACTION_STEPS` | `50` | PI0.5 replanning cadence |
| `FPS` | `30` | Policy command rate |
| `NAV_FORWARD_DISTANCE` | `0` | Real-mode startup-relative approach distance (m) |
| `HF_TOKEN` | empty | Optional token for gated repositories |

The checkpoint must contain `config.json`, `model.safetensors`, both saved
processor JSON files, normalization/unnormalization safetensors, and the
`tokenizer/` directory. The downloader validates all required files before
starting inference.

## Reproducibility

- LeRobot commit: `22bd7a2f489b367d8df42de803b1e8c4ca63a3f9`
- PI0.5 contract: state `[37]`, action `[20]`, three RGB inputs, chunk size 50
- Model details: [`model-manifest.json`](model-manifest.json)

Do not commit model weights, local paths, credentials, or private logs. The
public GitHub repository is the required code/Docker/README submission; model
weights are supplementary material.
