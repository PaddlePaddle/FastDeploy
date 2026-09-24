# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

export PRE_COMPILE_LOG_LEVEL="INFO"
export DG_CACHE_DIR=$(pwd)/deep_gemm_cache
export CUDA_VISIBLE_DEVICES=0

echo DeepGEMM Cache Dir: $DG_CACHE_DIR

MODEL_PATH=${1:-"/workspace3/chenjianye/models/eb5_A35B_260113_midtrain_ema_ckpts_step_14220_extracted"}
CHUNK_SIZE=${2:-"8192"}
TENSOR_PARALLEL_SIZE=${3:-"4"}
EXPERT_PARALLEL_SIZE=${4:-"16"}
HAS_SHARED_EXPERTS=${5:-"True"}
nproc=$(nproc)


python warmup_gemm_for_blackwell.py \
    --model $MODEL_PATH \
    --chunk-size $CHUNK_SIZE \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --expert-parallel-size $EXPERT_PARALLEL_SIZE \
    --has-shared-experts $HAS_SHARED_EXPERTS
