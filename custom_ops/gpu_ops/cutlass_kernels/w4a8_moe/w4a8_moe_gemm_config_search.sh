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

up_gate_proj_n=7168
up_gate_proj_k=8192

down_proj_n=8192
down_proj_k=3584
rm -rf up_gate_proj_7168_8192.log
rm -rf down_proj_8192_3584.log
num_experts=8

for tokens_per_expert in 1 2 4 8 16 20 24 28 32 36 48 64 96 128 160 192 224 256 384 512 768 1024 2048 3072 4096 8192

do
wait
CUDA_VISIBLE_DEVICES=2 ./w4a8_moe_gemm_test ${num_experts} ${ffn1_n} ${ffn1_k} ${tokens_per_expert} 0 1 >> ffn1_${ffn1_n}_${ffn1_k}.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 ./w4a8_moe_gemm_test ${num_experts} ${ffn2_n} ${ffn2_k} ${tokens_per_expert} 0 1 >> ffn2_${ffn2_n}_${ffn2_k}.log 2>&1 &
done
wait
echo "#### finish ####"
