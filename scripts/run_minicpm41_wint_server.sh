#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

QUANTIZATION="${QUANTIZATION:-}"
if [[ -z "${QUANTIZATION}" && $# -gt 0 ]]; then
    QUANTIZATION="$1"
    shift
fi
case "${QUANTIZATION}" in
    wint4 | wint8) ;;
    *)
        printf 'Quantization must be wint4 or wint8, got: %s\n' "${QUANTIZATION:-<empty>}" >&2
        printf 'Usage: %s {wint4|wint8} /path/to/MiniCPM4.1-8B [server options]\n' "$0" >&2
        exit 2
        ;;
esac

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" && $# -gt 0 && "${1}" != --* ]]; then
    MODEL_PATH="$1"
    shift
fi
if [[ -z "${MODEL_PATH}" ]]; then
    printf 'Model path is required.\n' >&2
    printf 'Usage: %s {wint4|wint8} /path/to/MiniCPM4.1-8B [server options]\n' "$0" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
PYTHON_ENV_ROOT="$(cd "${PYTHON_ENV_BIN}/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
TP_SIZE="${TP_SIZE:-${#GPU_LIST[@]}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
API_PORT="${FD_API_PORT:-8180}"
METRICS_PORT="${FD_METRICS_PORT:-8181}"
ENGINE_QUEUE_PORT="${FD_ENGINE_QUEUE_PORT:-8182}"
CACHE_QUEUE_PORT="${FD_CACHE_QUEUE_PORT:-8183}"
ATTENTION_BACKEND="${FD_ATTENTION_BACKEND:-FLASH_ATTN}"
MODEL_SOURCE="${FD_MODEL_SOURCE:-HUGGINGFACE}"
SERVER_WORKERS="${FD_SERVER_WORKERS:-1}"
SERVED_MODEL_NAME="${FD_SERVED_MODEL_NAME:-MiniCPM4.1-8B}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-128}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    printf 'Model directory not found: %s\n' "${MODEL_PATH}" >&2
    exit 2
fi

if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PYTHON_ENV_BIN}:${PATH}"
fi
PYTHON_NVIDIA_LIBRARY_PATH="$(
    find "${PYTHON_ENV_ROOT}/lib" \
        -type d -path '*/site-packages/nvidia/*/lib' -print | sort | paste -sd: -
)"
if [[ -n "${PYTHON_NVIDIA_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

PYTHONPATH_VALUE="${REPO_ROOT}"
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi

printf 'Starting MiniCPM4.1 %s online-quantized server\n' "${QUANTIZATION}"
printf '  model: %s\n' "${MODEL_PATH}"
printf '  GPUs: %s (TP=%s)\n' "${GPU_IDS}" "${TP_SIZE}"
printf '  API: http://127.0.0.1:%s/v1/chat/completions\n' "${API_PORT}"

exec env \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    FD_ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
    FD_MODEL_SOURCE="${MODEL_SOURCE}" \
    "${PYTHON_BIN}" -u -m fastdeploy.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${API_PORT}" \
    --metrics-port "${METRICS_PORT}" \
    --engine-worker-queue-port "${ENGINE_QUEUE_PORT}" \
    --cache-queue-port "${CACHE_QUEUE_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --workers "${SERVER_WORKERS}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --no-enable-prefix-caching \
    --quantization "${QUANTIZATION}" \
    "$@"
