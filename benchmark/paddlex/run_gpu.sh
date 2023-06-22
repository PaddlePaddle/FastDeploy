#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES='0'
bash benchmark_gpu.sh config/config.gpu.ort.fp32.e2e.txt
bash benchmark_gpu.sh config/config.gpu.ort.fp32.txt
bash benchmark_gpu.sh config/config.gpu.ort.fp32.h2d.txt

bash benchmark_gpu.sh config/config.gpu.paddle.fp32.e2e.txt
bash benchmark_gpu.sh config/config.gpu.paddle.fp32.txt
bash benchmark_gpu.sh config/config.gpu.paddle.fp32.h2d.txt

# rm all paddle_trt/trt cache
find . -name "trt_serialized*" | xargs rm -rf
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp16.e2e.txt
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp16.txt
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp16.h2d.txt
# rm all paddle_trt/trt cache
find . -name "trt_serialized*" | xargs rm -rf
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp32.e2e.txt
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp32.txt
bash benchmark_gpu_trt.sh config/config.gpu.paddle_trt.fp32.h2d.txt
# rm all paddle_trt/trt cache
find . -name "trt_serialized*" | xargs rm -rf
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp16.e2e.txt
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp16.txt
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp16.h2d.txt
# rm all paddle_trt/trt cache
find . -name "trt_serialized*" | xargs rm -rf
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp32.e2e.txt
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp32.txt
bash benchmark_gpu_trt.sh config/config.gpu.trt.fp32.h2d.txt
# rm all paddle_trt/trt cache
find . -name "trt_serialized*" | xargs rm -rf

set +x
