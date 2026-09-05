# Validation and usage

## Build

```bash
docker build -t ebim-task2-pi05:30k .
docker build --platform linux/arm64 -f Dockerfile.jetson -t ebim-task2-pi05:jetson .
```

## Checkpoint

Mount the complete PI0.5 checkpoint at `/models/pi05-task2-fullft-30k`.
Required files include the model, processors, normalization files and `tokenizer/`.

## Run

```bash
docker run --rm --network host --ipc host -e MODE=preflight ebim-task2-pi05:30k
docker run --rm --network host --ipc host -e MODE=navigation ebim-task2-pi05:30k
docker run --rm --gpus all --network host --ipc host \
  -v /path/to/pretrained_model:/models/pi05-task2-fullft-30k:ro \
  -v /path/to/run-logs:/app/out \
  ebim-task2-pi05:30k
```

Set `ROS_DOMAIN_ID` for the robot. Use `--runtime nvidia` with the Jetson image.

The runtime follows the 42D state / 17D action interface with 50-step chunks,
20 Hz execution and 10-step replanning. Navigation completes before policy
control begins.

## Smoke check

```bash
python scripts/smoke_pi05.py --dataset /path/to/task2_munich --port 8765 --chunks 1
```

Expected output: a finite `(50, 17)` action chunk.
