#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u

MODE="${MODE:-all}"
MODEL_DIR="${MODEL_DIR:-/models/pi05-task2-fullft-30k}"
MODEL_REPO="${MODEL_REPO:-junjie-jjs/ebim-task2-pi05-fullft-30k}"
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-8765}"
DEVICE="${DEVICE:-cuda}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
FPS="${FPS:-30}"
TASK="${TASK:-Pick up the thermal pad and place it on the target RAM board.}"
ROS_PROFILE="${ROS_PROFILE:-real}"
NAV_FORWARD_DISTANCE="${NAV_FORWARD_DISTANCE:-0}"
REAL_LEFT_ARM_COMMAND="${REAL_LEFT_ARM_COMMAND:-/left/gello/joint_states}"
REAL_RIGHT_ARM_COMMAND="${REAL_RIGHT_ARM_COMMAND:-/right/gello/joint_states}"

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
    --ros-profile "${ROS_PROFILE}"
    --real-left-arm-command "${REAL_LEFT_ARM_COMMAND}"
    --real-right-arm-command "${REAL_RIGHT_ARM_COMMAND}"
    --start-delay-sim 1.0
    --reset-scene-on-start
)

navigation_command=(
    python3 /app/task2_base_nav.py
    --forward-distance "${NAV_FORWARD_DISTANCE}"
)

case "${MODE}" in
    inference)
        download_checkpoint
        exec "${inference_command[@]}"
        ;;
    ros)
        if [[ "${ROS_PROFILE}" == "real" && "${NAV_FORWARD_DISTANCE}" != "0" ]]; then
            "${navigation_command[@]}"
        fi
        exec "${ros_command[@]}"
        ;;
    all)
        # On the real testbed, finish the table approach before loading or
        # starting policy inference so no policy node can contend for the base.
        if [[ "${ROS_PROFILE}" == "real" && "${NAV_FORWARD_DISTANCE}" != "0" ]]; then
            "${navigation_command[@]}"
        fi
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
