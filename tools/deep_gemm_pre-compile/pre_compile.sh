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

echo DeepGEMM Cache Dir: $DG_CACHE_DIR

MODEL_PATH=${1:-"/path/to/model"}
TENSOR_PARALLEL_SIZE=${2:-"1"}
EXPERT_PARALLEL_SIZE=${3:-"8"}
HAS_SHARED_EXPERTS=${4:-"False"}
OUTPUT_FILE=${5:-"./deep_gemm_pre_compile_config.jsonl"}
nproc=$(nproc)

python generate_config.py \
    --model $MODEL_PATH \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --expert-parallel-size $EXPERT_PARALLEL_SIZE \
    --has-shared-experts $HAS_SHARED_EXPERTS \
    --output $OUTPUT_FILE

python pre_compile.py \
    --config-file $OUTPUT_FILE \
    --expert-parallel-size $EXPERT_PARALLEL_SIZE \
    --num-threads $nproc
