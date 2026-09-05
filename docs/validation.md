# Validation Record

Date: 2026-09-05. This is an offline software validation record, not a physical
robot acceptance test. No model weights were changed. No repository was pushed,
published, or made public. Large container builds were not run.

## Latest Deployment Audit (Supersedes Earlier Runtime Checks)

Reference: HKUST issue #29, commit
`5fdb9d062a17222cd0c8f67af98f90719a5674cf`. This pass changes startup and command
transport. Historical fake-rig/synthetic results below do not validate these
new paths. No motion simulator or fake ROS robot was run in this pass.

- Added controller state/type selection and service sequencing in
  `app/rig_lifecycle.py`: impedance reload, missing/unconfigured controller
  handling, one-second current-joint stream before activation, BEST_EFFORT /
  activate_asap, and post-service state verification. Missing type prevents
  unload; ambiguous choices stop. Arm topics absent before configuration may
  be provisionally resolved. Current-pose streaming begins before activation,
  as in HKUST; matching subscribers are required after activation before moving.
- Added scoped base hardware activation with ACTIVE (3), component identification
  using base command interfaces, and verification. Unknown states are rejected.
  Optional absent base hardware services retain HKUST-compatible behavior;
  the base controller is still checked when its list service exists.
- Moved all actuator publication into a separate process with depth-one ROS
  publishers, bounded nonblocking IPC, a parent heartbeat and independent
  base/policy deadlines. Controller holds now precede model warmup. Shutdown
  forwards SIGTERM to the ROS runner before stopping inference.
- Matched zero base headers, arm host-clock + 0.1 s timestamps and local clock
  fallback. Fresh zero-stamp sensors are accepted, but frozen/backward positive
  stamps cannot refresh the observation.
- Fixed chunk exhaustion under steady slow inference using HKUST's two-latency
  coverage arithmetic plus 10% margin. Preserved 50 steps / 20 Hz / replan 10.
  Minimum playback rate 0.25 bounds sustainable latency to about 4.45 s.
- Forward LiDAR thresholds now use the raw sensor distance convention of HKUST.
  The previous extra footprint subtraction could stop docking too early.
  Side/reverse probes keep transformed footprint clearance; the base frame comes
  from odometry, and inverted horizontal scanner mounts are supported.
- Jetson recipe uses the same NGC iGPU base and Cyclone DDS default as HKUST,
  preserves vendor torch/torchvision and NumPy 1.x, and checks ROS/PI05 imports
  at build time. It still uses our pinned training LeRobot revision and 42D/17D
  checkpoint. No new image was built, so native ARM64 compatibility is unproven.

Latest executed checks: **30 focused unit tests passed**, Python compilation and
shell syntax passed. Tests cover lifecycle decisions/order and response
verification, typed bootstrap, timestamps, independent packet deadlines,
consecutive slow inference with scheduling jitter, scan transforms, strict
weight failures, camera stride, and state/action contracts. They do not execute
ROS lifecycle services or the new command child against a robot.

```bash
PYTHONPATH=app:tests python -m unittest test_checkpoint_startup test_deployment_contract test_real_contract \
  test_strict_weights test_navigation.InterfaceTest test_navigation.ChunkTest -v
python -m compileall -q app scripts tests
bash -n entrypoint.sh
```

Submission packaging inspection found that the existing local
`ebim-task2-pi05:30k` image is x86_64 and lacks the new real runner, command
process, lifecycle helper, YAML and reference bank. This was a network-disabled
file check, without ROS initialization or robot simulation. That old image
must not be mistaken for the updated checkout. No replacement image was built.

The startup downloader now checks all required files rather than only the main
weight file, skips network access for a complete local checkpoint, and rejects
missing/empty processors or tokenizer files after download. Three additional
regression tests passed, and the local actual 30k checkpoint passed the file
completeness check. This is not a new inference or checkpoint checksum test.
Docker build context now excludes local models, run output and environment files.

## Updated runtime image check

The current checkout was overlaid onto the existing dependency image and built
as `ebim-task2-pi05:updated-runtime`. The overlay installed the missing
Cyclone DDS and controller-manager packages, then passed Python compilation and
shell syntax checks. Inside the image, ROS imports for `rclpy`, `tf2_ros`,
controller-manager services, lifecycle messages and `GetParameters` passed.
The image contained the new real runner, command stream, lifecycle helper,
configuration and reference bank.

The mounted complete 30k checkpoint loaded in that image's inference server.
One real recorded observation produced a finite `(50, 17)` action chunk in
**2.817 s**. The server reported the expected 42D/17D contract, 50-step chunk,
20 Hz deployment cadence and 10-step replanning. The first action had spine
434.0 and finite arm/gripper values.

This is an x86_64 runtime-image and model-interface check. It is not an ARM64
Jetson build, ROS graph check on the robot, or manipulation success test. The
canonical Dockerfile's first network build attempt timed out while cloning
GitHub; use the proxy build-argument command in the README. The local overlay
image was used solely to complete dependency and inference verification.

## Evidence And Scope

The referenced task history was read back to its first available turn. The
pasted navigation specification, official ROS topic document, current real
and legacy inference adapters, startup scripts, model manifest, tests, and
HKUST issue #29's pinned Phase II modules were inspected. HKUST's `layout`,
`node`, `policy8`, `stream`, `rigops`, `imageproc`, `dock`, `run`, configuration,
Jetson recipe and fake-rig tests informed the interface comparison. Unrelated
Phase I files were not used as evidence of real-robot behavior.

HKUST's policy has a right-arm 8D action contract; this submission retains its
own 42D/17D full fine-tune. Only verified robot interface semantics were adopted.
The coordinate search, image registration and reference bank are independently
implemented. HKUST's board-centroid controller is not included.

## Changes That Address Deployment Failures

- Strict complete-checkpoint loading replaces an upstream loader that could
  catch a weight-loading exception and return an uninitialized/partial policy.
  Meta-device initialization avoids allocating a full random model first;
  upstream key conversion, expected tensor shapes/dtypes, and deterministic
  Transformers buffers are checked before use.
- Local tokenizer and saved processor loading are explicit. The 42D state still
  contains the raw right knuckle angle, while measured gripper hold commands
  correctly use opening fraction. Finite state/action and camera shape checks
  remain mandatory.
- Typed discovery recognizes follower and ordinary command namespaces, both
  base message types and camera variants. Incomplete/duplicate arm states
  cannot be marked valid. Command names come from measured joint names.
- Sensor queues retain one latest sample, with reliability compatible with
  discovered publishers. Frozen/backward source timestamps cannot refresh
  sensor data. Relative source-clock delay also contributes to freshness.
- An isolated ROS command process maintains arm commands through inference and docking.
  Base and active-policy leases stop stalled control loops. Fault holds freeze
  once to fresh measured arm positions; repeated zeros are attempted even when
  fault-aware sleeps or individual publications fail.
- Model loading/warmup precedes robot movement. Spine and arms are prepared
  before navigation. The final visual gate, arm/spine posture and stationary
  base are checked before policy execution. Navigation failure cannot start PI0.5.
- Real startup uses the new runner by default; the legacy version is no longer
  silently selected by an unset async environment variable. A port-probe client
  disconnecting during hello no longer terminates the inference server.

## Dataset Replay

Input: all 238 released Munich episodes, 476 selected early frames. Only head
videos are required for bank construction. Selection uses low measured right
arm motion across two temporal bins in the first 0.75 s; it is a stability
proxy, not a proof of optical sharpness. Video PTS is used because the stream
cadence can differ from the 20 Hz table.

The initial 640x360 / unconstrained global clustering trial accepted 99/476
frames. Raising feature resolution to 960x540 and using balanced representative
groups increased acceptance to **416/476 (87.39%)**. All eight references have
cross-episode calibration support. Reference arrays plus metadata total about
1.5 MB. Calibration excludes the chosen reference's own episode.

| Test | Observed Result |
| --- | --- |
| A: selected starts | 416 accepted; 60 rejected |
| A: invalid best matches | 43 too few unique matches; 15 insufficient consensus |
| A: remaining rejections | 2 geometrically valid registrations failed readiness |
| B: 1.5 s after start | 203/238 accepted |
| B: 3 s after start | 206/238 accepted |
| B: 6 s after start | 207/238 accepted |
| C: 1%, 3%, 6%, 12% horizontal translation | Median scores -0.66, -2.15, -4.41, -8.92 |
| C: 12% translation | 0/8 ready |
| C: 25% scale increase / 12 degree rotation | 0/8 ready for each |
| C: three moderate photometric changes | 8/8 ready for each |
| C: JPEG quality 55 | 8/8 ready |
| D: blank images | 0/8 ready |

Temporal results are a limitation: the static scene can remain matchable after
manipulation starts. This score tests docking-view consistency and does not
identify the manipulation phase. The arm/spine start checks supply additional
evidence. No negative dataset of real robot navigation approaches is available;
false-positive rate at incorrect physical docking poses is unmeasured.

These numbers are in-sample consistency checks. They are not held-out accuracy,
proof that a homography implies a reachable grasp pose, or site transfer evidence.
Some 6% translations and 10% scale changes still lie inside the measured gate;
the gate cannot be interpreted as a millimetre positioning tolerance.

Commands (repository root):

```bash
python scripts/build_demo_start_bank.py --dataset /tmp/ebim-real-meta/task2_munich
python scripts/evaluate_demo_bank.py --dataset /tmp/ebim-real-meta/task2_munich
```

Detailed results: `out/demo_bank_evaluation.json`, including every false reject
and all temporal/perturbation rows. The absolute dataset path is local test
context only; it is not embedded in deployment configuration.

## Historical Motion And ROS Checks (Before Latest Audit Changes)

Earlier `python -m unittest discover -s tests -v`: **32 tests passed**. Coverage includes
both signs in rotated body frames, stale/missing/NaN data, odometry jumps,
blocked/unknown LiDAR sectors, primitive timeout, repeated stop on fault,
dry-run no motion, rejected-probe rollback, step shrink, iteration limits,
post-stop handoff recheck, fixed-reference coarse capture, rejection of distant
but valid registrations, interface parsing, chunk timing, strict checkpoint
failures, and legacy state/action contracts.

`python scripts/replay_visual_alignment.py`: **8/8 synthetic searches reached
the known acceptance region**, with different signed offsets, optional yaw and
noise. Simulated duration 34.7-72.0 s; path 0.387-1.020 m. This validates the
optimizer against a known smooth score; it does not establish smoothness of a
real camera's score along a physical robot trajectory.

ROS tests use the already available `ebim-task2-pi05:30k-clean` image, a mounted
checkout, domain 173 and container-local networking. Only the small missing
`controller-manager-msgs` package was installed. No physical robot was connected.
The independent fake rig exercises stamped and unstamped base message types,
follower/plain namespaces, raw BGRA head frames, all state/wrench streams,
TF-transformed scans, two inactive controller services, and a host clock offset
of 12,345 s. Tests check dry-run creates no publishers, preparation, bounded
x/y motion, arm keepalive while the main thread sleeps, camera failure and
expired base command lease. Final received base commands are zero.

Initial large-image ROS runs intermittently hit camera skew/staleness and
stopped safely. The final subscriber uses depth one and publisher-compatible
reliability; transport failures must still stop the robot. Logs from the final
scenarios are `out/ros_follower.json` and `out/ros_plain.json`.

A third scenario (`--policy`, `out/ros_policy.json`) exercised repeated visual
handoff verification and the real policy playback loop using a delayed mock
chunk client. It received 534 arm commands, maximum interval 55.3 ms, zero
joint-name/timestamp errors, two controller activations and final zero base
velocity. The actual model is checked separately below. The two-second duration
is a test-only override; the deployed default remains continuous manipulation.

## Real Checkpoint Inference

Hardware: local RTX 4060 Ti 16 GB, existing LeRobot conda environment, pinned
source revision, CUDA torch runtime. Input is episode 0's actual 42D state and
all three real camera frames. No ROS command publishers are involved.

```bash
python app/task2_pi05_eval_node.py --mode inference \
  --checkpoint /path/to/pretrained_model --host 127.0.0.1 --port 18765 --device cuda
python scripts/smoke_pi05.py --dataset /tmp/ebim-real-meta/task2_munich --port 18765
```

Three requests returned finite `50x17` arrays. Request durations were **2.650 s,
0.327 s and 0.319 s**. Predicted spine was 434.0; slightly above-one gripper
outputs are clamped by the existing action filter. Actual actions and timings
are recorded in `out/pi05_smoke.json`. These timings do not predict Jetson
latency or task success. Policy warmup runs before moving the robot.

## Remaining Physical Limits

No true starting-position-to-table trajectory is in the demonstration data.
Coarse navigation assumes the marked start faces the table along a clear
straight corridor. The bounded search can stop at a local optimum, lose
registration, or encounter a LiDAR limit before reaching the desired view.
Rear/side TF and scan coverage are required for probing; table surfaces and
obstacles outside the LiDAR scan plane are not covered by the 2D safety model.

The configured circular footprint, braking clearance, FR3 limits and startup
joint interpolation have not been measured on this installation. Startup arm
interpolation has no collision planner and requires a clear initial posture.
Joint-state and laser checks do not prove self-collision or table collision
freedom. No automatic large recovery motion is provided.

The separate Jetson recipe preserves HKUST's native CUDA/NumPy ABI strategy,
but has not been built or run on ARM64. The default Dockerfile targets desktop
CUDA. Tests use the existing image with current code mounted, so they do not
constitute validation of a fresh Docker build. Physical arrival and successful
manipulation cannot be guaranteed from this record.
