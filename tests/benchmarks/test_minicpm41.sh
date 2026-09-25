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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
PYTHON_ENV_ROOT="$(cd "${PYTHON_ENV_BIN}/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MINICPM41_MODEL_PATH:-${MODEL_PATH:-}}"

case "${TARGET}" in
    all | build | unit | operators | e2e) ;;
    *)
        printf 'Usage: %s [all|build|unit|operators|e2e]\n' "$0" >&2
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi

export PATH="${CUDA_HOME}/bin:${PYTHON_ENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_NVIDIA_LIBRARY_PATH="$(
    find "${PYTHON_ENV_ROOT}/lib" \
        -type d -path '*/site-packages/nvidia/*/lib' -print | sort | paste -sd: -
)"
if [[ -n "${PYTHON_NVIDIA_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

require_model() {
    if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}" ]]; then
        printf 'Set MODEL_PATH to the local MiniCPM4.1-8B directory.\n' >&2
        return 2
    fi
}

verify_build() {
    local library
    local symbol
    library="$(find fastdeploy/model_executor/ops/gpu -type f -name 'fastdeploy_ops*.so' -print -quit)"
    if [[ -z "${library}" ]]; then
        printf 'fastdeploy_ops shared library was not generated.\n' >&2
        return 1
    fi
    for symbol in infllmv2_update_compressed_k infllmv2_select_blocks infllmv2_attention_forward; do
        strings "${library}" | grep "${symbol}" >/dev/null
    done
    printf 'PASS build: %s\n' "${library}"
}

run_build() {
    local cudnn_header
    local nvidia_ml_stub=""
    local candidate
    cudnn_header="$(
        find "${PYTHON_ENV_ROOT}/lib" \
            -type f -path '*/site-packages/nvidia/cudnn/include/cudnn.h' -print -quit
    )"
    if [[ -n "${cudnn_header}" ]]; then
        export CPATH="$(dirname "${cudnn_header}")${CPATH:+:${CPATH}}"
    fi
    for candidate in \
        "${CUDA_HOME}/targets/x86_64-linux/lib/stubs" \
        /usr/lib/x86_64-linux-gnu/stubs; do
        if [[ -f "${candidate}/libnvidia-ml.so" ]]; then
            nvidia_ml_stub="${candidate}"
            break
        fi
    done
    if [[ -n "${nvidia_ml_stub}" ]]; then
        export LIBRARY_PATH="${nvidia_ml_stub}:${PYTHON_NVIDIA_LIBRARY_PATH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi
    MAX_JOBS="${MAX_JOBS:-4}" \
        bash build.sh 0 "${PYTHON_BIN}" false "${FD_BUILDING_ARCS:-[86]}"
    verify_build
}

run_unit() {
    "${PYTHON_BIN}" -m pytest -q \
        tests/model_executor/test_minicpm41.py \
        tests/model_executor/test_thinking_budget.py \
        tests/model_executor/test_infllmv2_attention_backend.py \
        tests/quantization/test_minicpm41_int_quant.py \
        tests/quantization/test_minicpm41_quality_eval.py
}

run_operators() {
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m pytest -q \
        tests/operators/test_infllmv2_attention_forward.py
}

run_e2e_nodes() {
    "${PYTHON_BIN}" -m pytest -q \
        tests/e2e/test_minicpm41_serving.py::test_minicpm41_openai_chat_completion_e2e \
        tests/e2e/test_minicpm41_serving.py::test_minicpm41_chat_completion_with_history_e2e \
        tests/e2e/test_minicpm41_serving.py::test_minicpm41_forces_multitoken_thinking_end_e2e \
        tests/e2e/test_minicpm41_serving.py::test_minicpm41_disable_thinking_does_not_emit_think_block_e2e \
        tests/e2e/test_minicpm41_serving.py::test_minicpm41_mixed_thinking_modes_e2e
}

run_e2e() {
    require_model
    export MINICPM41_MODEL_PATH="${MODEL_PATH}"
    export MINICPM41_E2E_MAX_MODEL_LEN="${MINICPM41_E2E_MAX_MODEL_LEN:-1024}"
    export MINICPM41_E2E_MAX_NUM_SEQS="${MINICPM41_E2E_MAX_NUM_SEQS:-2}"
    export MINICPM41_E2E_EXTRA_ARGS="${MINICPM41_E2E_EXTRA_ARGS:---workers 1 --max-num-batched-tokens 256 --num-gpu-blocks-override 32 --no-enable-prefix-caching}"
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

    export FD_ATTENTION_BACKEND=FLASH_ATTN
    unset MINICPM41_E2E_QUANTIZATION
    run_e2e_nodes
    export MINICPM41_E2E_QUANTIZATION=wint4
    run_e2e_nodes
    export MINICPM41_E2E_QUANTIZATION=wint8
    run_e2e_nodes
    unset MINICPM41_E2E_QUANTIZATION
    export FD_ATTENTION_BACKEND=INFLLMV2_ATTN
    run_e2e_nodes
    printf 'PASS e2e: BF16, WINT4, WINT8, and InfLLM-V2\n'
}

if [[ "${TARGET}" == "all" || "${TARGET}" == "build" ]]; then
    run_build
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "unit" ]]; then
    run_unit
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "operators" ]]; then
    run_operators
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "e2e" ]]; then
    run_e2e
fi
