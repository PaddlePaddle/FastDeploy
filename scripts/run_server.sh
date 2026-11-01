#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/envSetup.sh"

MODEL="/mnt/moark-models/PaddleOCR-VL-1.5"
MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-0}" \
python3 -m fastdeploy.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.4 \
    --max-num-seqs 256 \
    --graph-optimization-config '{"use_cudagraph":true,"graph_opt_level":0}' \
    --workers 6 \
    --max-concurrency 4096 \
    --port 8300 \
    --metrics-port 8301 \
    --engine-worker-queue-port 8302
