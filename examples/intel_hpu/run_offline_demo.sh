#!/bin/bash

export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# export HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HPU_VISIBLE_DEVICES=0
export HABANA_PROFILE=0
export PROFILE_START=1
export PROFILE_END=3
# export HABANA_LOGS=hpu_logs
# export LOG_LEVEL_ALL=0
# export FLAGS_intel_hpu_runtime_debug=1
# export FLAGS_intel_hpu_reciperunner_debug=1

rm -rf log
FD_ATTENTION_BACKEND=HPU_ATTN python offline_demo.py
