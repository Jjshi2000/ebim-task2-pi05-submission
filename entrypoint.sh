#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u

MODE="${MODE:-all}"
MODEL_DIR="${MODEL_DIR:-/models/pi05-task2-fullft-20k}"
MODEL_REPO="${MODEL_REPO:-junjie-jjs/ebim-task2-pi05-fullft-20k}"
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-8765}"
DEVICE="${DEVICE:-cuda}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
FPS="${FPS:-30}"
TASK="${TASK:-Pick up the thermal pad and place it on the target RAM board.}"

download_checkpoint() {
    if [[ -f "${MODEL_DIR}/model.safetensors" ]]; then
        return
    fi
    if [[ -z "${MODEL_REPO}" ]]; then
        echo "ERROR: set MODEL_REPO or mount a complete checkpoint at ${MODEL_DIR}" >&2
        exit 2
    fi
    python3 /app/download_model.py \
        --repo-id "${MODEL_REPO}" \
        --local-dir "${MODEL_DIR}"
}

inference_command=(
    python3 /app/task2_pi05_eval_node.py
    --mode inference
    --checkpoint "${MODEL_DIR}"
    --device "${DEVICE}"
    --task "${TASK}"
    --host "${INFERENCE_HOST}"
    --port "${INFERENCE_PORT}"
    --n-action-steps-override "${N_ACTION_STEPS}"
)

ros_command=(
    python3 /app/task2_pi05_eval_node.py
    --mode ros
    --host "${INFERENCE_HOST}"
    --port "${INFERENCE_PORT}"
    --fps "${FPS}"
    --start-delay-sim 1.0
    --reset-scene-on-start
)

case "${MODE}" in
    inference)
        download_checkpoint
        exec "${inference_command[@]}"
        ;;
    ros)
        exec "${ros_command[@]}"
        ;;
    all)
        download_checkpoint
        "${inference_command[@]}" &
        inference_pid=$!
        cleanup() {
            kill "${inference_pid}" 2>/dev/null || true
            wait "${inference_pid}" 2>/dev/null || true
        }
        trap cleanup EXIT INT TERM
        python3 /app/wait_for_port.py \
            --host "${INFERENCE_HOST}" \
            --port "${INFERENCE_PORT}" \
            --process-pid "${inference_pid}" \
            --timeout 600
        "${ros_command[@]}"
        ;;
    *)
        echo "ERROR: MODE must be all, inference, or ros; got ${MODE}" >&2
        exit 2
        ;;
esac
