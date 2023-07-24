#!/usr/bin/env bash
set -x

bash benchmark_x86.sh config/config.x86.ort.fp32.e2e.txt
bash benchmark_x86.sh config/config.x86.ort.fp32.txt

bash benchmark_x86.sh config/config.x86.ov.fp32.e2e.txt
bash benchmark_x86.sh config/config.x86.ov.fp32.txt

bash benchmark_x86.sh config/config.x86.paddle.fp32.e2e.txt
bash benchmark_x86.sh config/config.x86.paddle.fp32.txt

set +x
