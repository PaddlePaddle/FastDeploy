#!/bin/bash


set -euo pipefail

pkill -9 -f "ocr_benchmark.py" 2>/dev/null || true
sleep 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/envSetup.sh"

cd /data/FastDeploy/benchmarks/paddleocr_vl

mkdir -p output

python3 ocr_benchmark.py \
    --input_dirs /data/OmniDocBench_v1_5/images_128_pdf \
    --paddlex_config_path PaddleOCR-VL-1_5_fastdeploy.yaml \
    --device metax_gpu \
    --gpu_ids 0 \
    --process_per_gpu 5 \
    --batch_size 4 \
    -o output/benchmark_result.json
