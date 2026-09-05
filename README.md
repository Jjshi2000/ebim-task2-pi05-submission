# EBiM Phase II Task 2

The submission runs bounded docking followed by continuous PI0.5 manipulation
with the released 42D/17D checkpoint contract.

## Deployment Targets

Desktop/server GPU with ROS 2 Jazzy:

```bash
docker build -t ebim-task2-pi05:30k .
```

With the configured local proxy:

```bash
docker build --network host \
  --build-arg http_proxy=http://127.0.0.1:12334 \
  --build-arg https_proxy=http://127.0.0.1:12334 \
  --build-arg HTTP_PROXY=http://127.0.0.1:12334 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:12334 \
  --build-arg no_proxy=localhost,127.0.0.0/8,::1 \
  --build-arg NO_PROXY=localhost,127.0.0.0/8,::1 \
  -t ebim-task2-pi05:30k .
```

Jetson AGX Orin / JetPack 6:

```bash
docker build --platform linux/arm64 -f Dockerfile.jetson -t ebim-task2-pi05:jetson .
```

## Start Commands

The complete checkpoint includes `model.safetensors`, `config.json`, both
processor JSON files, normalization states, and the `tokenizer/` directory.
Mount it at `/models/pi05-task2-fullft-30k`, or let the downloader fetch the
pinned model repository before any robot motion. An access token belongs only
in `HF_TOKEN` at runtime.
Startup checks every required checkpoint file, including processors and
tokenizer, for presence and nonzero size. Complete local checkpoints avoid
network access; incomplete writable caches trigger the pinned download.
The strict loader subsequently validates tensor keys and shapes.

Read-only preflight creates no command publishers:

```bash
docker run --rm --network host --ipc host -e MODE=preflight ebim-task2-pi05:30k
```

Navigation dry run subscribes and logs scores/proposed motion without creating
any arm, gripper, spine, or base command publishers. It does not require a GPU
or load the manipulation policy:

```bash
docker run --rm --network host --ipc host -e DRY_RUN=1 ebim-task2-pi05:30k
```

Navigation-only performs arm/spine preparation and docking, then stops:

```bash
docker run --rm --network host --ipc host -e MODE=navigation ebim-task2-pi05:30k
```

Full rollout (replace the checkpoint path with the local complete checkpoint):

```bash
docker run --rm --gpus all --network host --ipc host \
  -v /path/to/pretrained_model:/models/pi05-task2-fullft-30k:ro \
  -v /path/to/run-logs:/app/out \
  ebim-task2-pi05:30k
```

For Jetson, use `--runtime nvidia` and the Jetson image. Set `ROS_DOMAIN_ID` to
the robot's domain. The policy endpoint is a trusted localhost pickle protocol;
do not expose it to an untrusted network. Keep the physical emergency stop
available. A software zero-velocity command cannot replace hardware braking.

## Runtime Configuration

All new physical/control settings are in [config/task2.real.yaml](config/task2.real.yaml).
Mount a complete YAML file and set `REAL_CONFIG=/app/config/site.yaml` to
override it. Bank and log paths resolve relative to that YAML. Registration
settings must match the saved bank calibration; changing them requires a rebuild.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODE` | `all` | `all`, `inference`, `ros`, `navigation`, `preflight` |
| `DRY_RUN` | `0` | Read-only navigation observation |
| `REAL_CONFIG` | `/app/config/task2.real.yaml` | Real-rig and docking settings |
| `MODEL_DIR` | `/models/pi05-task2-fullft-30k` | Complete local checkpoint |
| `MODEL_REPO` | `junjie-jjs/ebim-task2-pi05-fullft-30k` | Download source |
| `MODEL_REVISION` | `af799f06e59be97a7a1b2603d610d22669305cf3` | Pinned checkpoint revision |
| `INFERENCE_HOST`, `INFERENCE_PORT` | `127.0.0.1`, `8765` | Local inference endpoint |
| `DEVICE` | `cuda` | Inference device |
| `HF_TOKEN` | unset | Runtime-only credential if required |
| `ROS_PROFILE` | `real` | `isaac` retains the separate legacy simulation adapter |

`NAV_FORWARD_DISTANCE` is retired and rejected for real runs. Real navigation
is enabled by default. Disabling `navigation.enabled` requests visual
verification at the current position; it does not bypass the handoff gate.
Real cadence/replan remain 20 Hz / 10 steps. `policy.max_episode_s: 0` means
continuous manipulation, without a timeout or a grasp-based success exit.

## Robot Interface

The runtime uses typed graph discovery, including issue #29's tested namespace
variants. Overrides in `rig.topics` must be unambiguous. During live startup,
a missing arm command topic can be provisionally inferred from the observed
gripper namespace or an override. Its typed controller subscription is required
after activation, before moving from the held measured pose. Read-only preflight
cannot create/configure controllers.

| Channel | Recognized interface |
| --- | --- |
| Arm state | `/{left,right}/franka_robot_state_broadcaster/measured_joint_states` |
| Wrench | `/{left,right}/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` |
| Arm commands | `/{left,right}/follower/gello/joint_states`, then `/{left,right}/gello/joint_states` |
| Gripper state/commands | Follower and ordinary gripper namespaces |
| Head RGB | `zed_node` / `zed` and rectified RGB topic variants |
| Wrists | Camera namespace variants under `/wrist_camera_{left,right}` |
| Base | `/swerve_drive_controller/odom` and `/swerve_drive_controller/cmd_vel` |
| LiDAR | `/lidar_front/scan`, `/lidar_rear/scan` |
| Spine | `/spine/joint_states`, `/spine/target_height` |

Base commands support `Twist` and `TwistStamped`, with a zero header stamp as
in HKUST. Arm commands retain measured joint names and extrapolate each arm's
host clock with a 0.1 s forward bias. Zero-stamp sensors use arrival freshness;
without arm clock samples, commands use local wall time as in HKUST.
Repeated/backward positive source stamps do not refresh observations.

An isolated 20 Hz process publishes commands through vision and inference.
It independently checks a 0.30 s parent heartbeat, 0.20 s base lease and 0.50 s
active-policy lease. Faults stop the base and freeze existing arm holds.
The bounded IPC queue never blocks the parent's watchdog.

Startup reads controller state/type, then reloads impedance controllers using
HKUST's deactivate/unload/load/configure sequence when a type is available.
It sends current joints for one second before activation and verifies every
state transition. Without a type it never unloads. Active follower controllers
are held; inactive/unconfigured ones are activated/configured. Ambiguous
controller choices stop startup. Manager paths/choices are configurable in
`rig.controller_managers` and `rig.arm_controller_overrides`.
Identified inactive base hardware is activated with lifecycle ACTIVE (3) and
checked; unknown-state/unrelated hardware is not activated. Missing optional
base hardware services are handled as in HKUST. Models load before ROS startup,
and controller holds start before model warmup.

State order is left q7, left gripper1, left external joint torque7, left
wrench6, right q7, right gripper1, right external joint torque7, right wrench6.
The left gripper and external joint torque fields stay zero as in training.
Right gripper state stays raw knuckle radians; commands are opening fractions.
Action order is left q7, left width1, right q7, right width1, spine1.
Spine target is 434 mm, with metre/millimetre feedback conversion and verification.

Camera row stride, RGB/BGR/BGRA/RGBA encodings, and same-aspect resizing are
handled explicitly. Checkpoint dimensions remain head 720x1280 and wrists
480x640. No HSV steering, color remapping, or fixed object-centroid target is used.

## Docking And Safety

`BOOT -> WAIT_FOR_SENSORS -> COARSE_APPROACH -> VISUAL_SEARCH_INIT ->
FINE_VISUAL_ALIGNMENT -> FINAL_SETTLE_AND_VERIFY -> READY_FOR_PI05`.
Any safety failure enters `SAFE_STOP` and returns nonzero.

Coarse motion is forward only, with odometry travel, time, and front LiDAR
bounds. The base stops during image matching. Several valid frames, within
1.5 times the calibrated start-distribution geometry limit, are needed
before fine alignment. SIFT/RANSAC registers whole images against precomputed
demo medoids; low consensus, concentrated inliers, implausible geometry, and
unreliable transforms are rejected. Geometry and image quality are compared
with robust cross-episode calibration statistics.

Fine alignment tests small positive and negative body-frame x/y motions; yaw
is optional and disabled by default. It keeps only measured score improvements,
returns to saved odometry targets after rejected probes, and shrinks steps.
The reference stays fixed during optimization so reference switching cannot
create a false improvement. There are no large recovery moves.

Default probe sizes are 2.5 cm / 2 cm; the fine envelope is 20 cm radius,
1.8 m cumulative travel, 24 iterations and 180 s. Coarse limits are 2.6 m and
100 s. These are configurable safety bounds, not site-calibrated guarantees.
Forward slowdown/hard-stop thresholds use **raw front LaserScan range**, as in
HKUST (our thresholds are 0.85/0.40 m; HKUST uses 0.60/0.30 m). Side/reverse
probes require 0.40 m free space beyond the configured 0.40 m footprint after
sensor transformation. TF uses the odometry child frame; optional measured
extrinsics live in `safety.lidar_extrinsics`. Horizontal inverted scanner mounts
are supported. The rear scan is required by default; missing TF/coverage stops
startup/motion.

Fresh sensors, valid registration, repeated ready evaluations, stopped
odometry, demonstrated arm/spine posture, and fresh post-inference verification
are required before manipulation. Policy targets use the existing demonstrated
range/slew filter, intersected with configured FR3 joint limits. Slow inference
uses HKUST's two-inference coverage calculation with a 10% margin and our
10-step replan setting. At minimum playback rate 0.25, measured latency above
about 4.45 s is unsustainable; 8 s is an absolute request ceiling, not a supported
steady rate. Unsustainable latency or exhausted chunks
stop the run. Logs record states, probes, registration statistics, safety
failures, and policy chunk timing in JSONL under `/app/out`.

## Offline Tools

Run from this repository using a Python environment with NumPy, OpenCV, PyAV,
PyArrow, PyYAML and the pinned LeRobot runtime for inference checks:

```bash
python scripts/build_demo_start_bank.py --dataset /path/to/task2_munich
python scripts/evaluate_demo_bank.py --dataset /path/to/task2_munich
python scripts/replay_visual_alignment.py
python -m unittest discover -s tests -v
```

The builder uses actual episode IDs (which contain gaps), video presentation
timestamps, two low-arm-motion frames from the first 0.75 s, deterministic
balanced medoids, and cross-episode calibration. Replay reports all rejected
starts, temporal frames, photometric/geometric perturbations and blank images.

With the inference server running locally and all three episode videos present:

```bash
python scripts/smoke_pi05.py --dataset /path/to/task2_munich --port 8765
```

ROS fake-rig tests must run on an isolated ROS domain, never alongside hardware:

```bash
ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1 python scripts/ros_smoke.py --layout follower --report out/ros_follower.json
ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1 python scripts/ros_smoke.py --layout plain --report out/ros_plain.json
```

LeRobot revision: `22bd7a2f489b367d8df42de803b1e8c4ca63a3f9`.
Dataset revision: `495ebb7b56fb9e2f3952398a63d86f08cacb9531`.
Interface comparison: [HKUST issue #29](https://github.com/EBiM-Benchmark/submissions/issues/29),
commit `5fdb9d062a17222cd0c8f67af98f90719a5674cf`.
