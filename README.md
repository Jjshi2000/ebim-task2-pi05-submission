# EBiM Phase II Task 2

Real robot submission: visual docking followed by PI0.5 manipulation.

## Build

Desktop/server (ROS 2 Jazzy):

```bash
docker build -t ebim-task2-pi05:30k .
```

Jetson AGX Orin (JetPack 6):

```bash
docker build --platform linux/arm64 -f Dockerfile.jetson -t ebim-task2-pi05:jetson .
```

## Checkpoint

Mount the complete checkpoint directory at `/models/pi05-task2-fullft-30k`.
It must include the model, processor files, normalization files and `tokenizer/`.

## Run

Preflight (no robot commands):

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

Set `ROS_DOMAIN_ID` to the robot's domain. On Jetson, use `--runtime nvidia`
and the Jetson image. Keep the hardware emergency stop available.

The runner uses the 42D state and 17D action contract, three cameras,
50-step action chunks, 20 Hz execution and 10-step replanning. PI0.5 starts
after navigation reaches its visual handoff state.

## Configuration

Copy [config/task2.real.yaml](config/task2.real.yaml) and set
`REAL_CONFIG=/app/config/site.yaml` when site-specific settings are needed.

## Smoke check

```bash
python scripts/smoke_pi05.py --dataset /path/to/task2_munich --port 8765 --chunks 1
```

The check should report a finite `(50, 17)` action chunk.
