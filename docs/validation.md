# Validation and usage

The repository contains the real runner, navigation code, PI0.5 adapter,
reference bank and Docker recipes. The 30 focused tests cover checkpoint files,
controller startup, ROS message contracts, command leases, LiDAR transforms and
PI0.5 state/action dimensions.

The code has not been tested on the physical robot or Jetson. Offline visual
evaluation accepted 416 of 476 selected start frames (87.4%); this is an image
consistency result, not a navigation or manipulation success rate.

## Build

Desktop ROS 2 Jazzy image:

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

Jetson AGX Orin candidate image:

```bash
docker build --platform linux/arm64 -f Dockerfile.jetson \
  -t ebim-task2-pi05:jetson .
```

## Checkpoint

Mount the complete checkpoint directory at
`/models/pi05-task2-fullft-30k`. It must contain the model, saved processors,
normalization files and `tokenizer/`. The startup script checks all required
files and downloads the pinned Hugging Face revision when the directory is
incomplete.

## Run

Read-only preflight:

```bash
docker run --rm --network host --ipc host -e MODE=preflight ebim-task2-pi05:30k
```

Navigation only:

```bash
docker run --rm --network host --ipc host -e MODE=navigation ebim-task2-pi05:30k
```

Full rollout:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v /path/to/pretrained_model:/models/pi05-task2-fullft-30k:ro \
  -v /path/to/run-logs:/app/out \
  ebim-task2-pi05:30k
```

Set `ROS_DOMAIN_ID` to the robot's domain. For Jetson use `--runtime nvidia`
and the Jetson image. Keep the hardware emergency stop available.

The real runner uses the official 42D state and 17D action contract, three
cameras, 50-step chunks, 20 Hz execution and 10-step replanning. Navigation
must pass its visual handoff before PI0.5 manipulation starts.

## PI0.5 smoke check

With the inference server running and the released dataset available:

```bash
python scripts/smoke_pi05.py --dataset /path/to/task2_munich --port 8765 --chunks 1
```

The expected result is a finite `(50, 17)` action chunk.
