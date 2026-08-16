FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
ARG LEROBOT_COMMIT=22bd7a2f489b367d8df42de803b1e8c4ca63a3f9

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    TASK2_PI05_USE_ASYNC=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pin LeRobot to the exact revision used for training and deployment tests.
# Linux PyTorch wheels include their CUDA runtime; the host supplies the NVIDIA driver.
RUN python3 -m venv --system-site-packages /opt/venv \
    && python3 -m pip install --no-cache-dir --upgrade pip "setuptools<82" wheel \
    && python3 -m pip install --no-cache-dir \
        "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}" \
        "httpx[socks]>=0.27,<1"

WORKDIR /app
COPY app/ /app/
COPY entrypoint.sh /entrypoint.sh
COPY model-manifest.json /app/model-manifest.json

RUN chmod +x /entrypoint.sh \
    && python3 -m py_compile \
        /app/task2_pi05_eval_node.py \
        /app/task2_pi05_eval_node_v2.py \
        /app/download_model.py \
        /app/wait_for_port.py

VOLUME ["/cache/huggingface", "/models"]
ENTRYPOINT ["/entrypoint.sh"]
