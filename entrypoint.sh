#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u

MODE="${MODE:-all}"
MODEL_DIR="${MODEL_DIR:-/models/pi05-task2-fullft-30k}"
INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
INFERENCE_PORT="${INFERENCE_PORT:-8765}"
REAL_CONFIG="${REAL_CONFIG:-/app/config/task2.real.yaml}"
ROS_PROFILE="${ROS_PROFILE:-real}"

download_checkpoint() {
    python3 /app/download_model.py --ensure-complete \
        --repo-id "${MODEL_REPO:-junjie-jjs/ebim-task2-pi05-fullft-30k}" \
        --local-dir "${MODEL_DIR}" \
        --revision "${MODEL_REVISION:-af799f06e59be97a7a1b2603d610d22669305cf3}"
}

inference_command=(python3 /app/task2_pi05_eval_node.py --mode inference
    --checkpoint "${MODEL_DIR}" --device "${DEVICE:-cuda}"
    --task "${TASK:-Pick up the thermal pad and place it on the target RAM board.}"
    --host "${INFERENCE_HOST}" --port "${INFERENCE_PORT}" --n-action-steps-override 10)
ros_command=(python3 /app/task2_real_runner.py --config "${REAL_CONFIG}"
    --host "${INFERENCE_HOST}" --port "${INFERENCE_PORT}")
if [[ "${ROS_PROFILE}" != real ]]; then
    ros_command=(python3 /app/task2_pi05_eval_node.py --mode ros --ros-profile isaac
        --host "${INFERENCE_HOST}" --port "${INFERENCE_PORT}" --fps "${FPS:-30}")
elif [[ "${NAV_FORWARD_DISTANCE:-0}" != 0 ]]; then
    echo "ERROR: NAV_FORWARD_DISTANCE is retired; configure bounded navigation in REAL_CONFIG." >&2
    exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then ros_command+=(--dry-run); fi

case "${MODE}" in
    preflight)
        exec python3 /app/task2_real_runner.py --config "${REAL_CONFIG}" --preflight
        ;;
    navigation)
        ros_command+=(--navigation-only)
        exec "${ros_command[@]}"
        ;;
    inference)
        download_checkpoint
        exec "${inference_command[@]}"
        ;;
    ros)
        exec "${ros_command[@]}"
        ;;
    all)
        if [[ "${DRY_RUN:-0}" == 1 && "${ROS_PROFILE}" == real ]]; then
            exec "${ros_command[@]}"
        fi
        download_checkpoint
        "${inference_command[@]}" &
        inference_pid=$!
        ros_pid=
        cleanup() {
            if [[ -n "${ros_pid}" ]]; then
                kill -TERM "${ros_pid}" 2>/dev/null || true
                wait "${ros_pid}" 2>/dev/null || true
            fi
            kill "${inference_pid}" 2>/dev/null || true
            wait "${inference_pid}" 2>/dev/null || true
        }
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        python3 /app/wait_for_port.py --host "${INFERENCE_HOST}" --port "${INFERENCE_PORT}" \
            --process-pid "${inference_pid}" --timeout 600
        "${ros_command[@]}" &
        ros_pid=$!
        wait "${ros_pid}"
        ;;
    *)
        echo "ERROR: MODE must be all, inference, ros, navigation, or preflight" >&2
        exit 2
        ;;
esac
