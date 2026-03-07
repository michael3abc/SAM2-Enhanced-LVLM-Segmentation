ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}
ARG HF_HOME=/workspace/inits/huggingface
ARG FLASH_ATTN_WHEEL_URL=https://drive.google.com/file/d/1Y87ikQEC8yQOFUxCVoERGPuQd7MYNwfj/view?usp=sharing
ARG FLASH_ATTN_WHEEL_NAME=flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL_SHA256=91571f68f4113065dba4464a0d8d911f9e27969f139a6851c46668e06c2ac5f7

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_DIR=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git git-lfs ca-certificates curl \
    build-essential ninja-build pkg-config \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/dev-entrypoint.sh /usr/local/bin/dev-entrypoint.sh
RUN python3.10 -m venv "${VENV_DIR}" \
    && "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir gdown==5.2.0 \
    && "${VENV_DIR}/bin/pip" install torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
    && grep -Ev '^(torch==|torchvision==|triton==|flash_attn==|xtuner==)' /tmp/requirements.txt > /tmp/requirements.no_ext.txt \
    && "${VENV_DIR}/bin/pip" install -r /tmp/requirements.no_ext.txt \
    && mkdir -p /tmp/wheels \
    && python3.10 -m gdown --fuzzy "${FLASH_ATTN_WHEEL_URL}" -O "/tmp/wheels/${FLASH_ATTN_WHEEL_NAME}" \
    && echo "${FLASH_ATTN_WHEEL_SHA256}  /tmp/wheels/${FLASH_ATTN_WHEEL_NAME}" | sha256sum -c - \
    && "${VENV_DIR}/bin/pip" install "/tmp/wheels/${FLASH_ATTN_WHEEL_NAME}" \
    && "${VENV_DIR}/bin/pip" install --no-deps xtuner==0.2.0
RUN chmod +x /usr/local/bin/dev-entrypoint.sh

ENV PATH="${VENV_DIR}/bin:${PATH}" \
    HF_HOME=${HF_HOME} \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADIO_TEMP_DIR=/tmp/gradio

ENTRYPOINT ["/usr/local/bin/dev-entrypoint.sh"]
CMD ["bash"]
