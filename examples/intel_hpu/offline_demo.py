"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "ERNIE-4.5-21B-A3B-Paddle"
# model_name_or_path = "ERNIE-4.5-300B-A47B-Paddle"

# Hyperparameter settings
input_bs = 1
input_seq = None  # 1000
max_out_tokens = 128
server_max_bs = 128
TP = 1
EP = True

# num_gpu_blocks_override = ceil((input_seq + max_out_tokens) / 128) * server_max_bs
num_gpu_blocks_override = 2000
sampling_params = SamplingParams(max_tokens=max_out_tokens)
graph_optimization_config = {"use_cudagraph": False}
llm = LLM(
    model=model_name_or_path,
    tensor_parallel_size=TP,
    enable_expert_parallel=EP,
    engine_worker_queue_port=8602,
    num_gpu_blocks_override=num_gpu_blocks_override,
    block_size=128,
    max_model_len=32768,
    max_num_seqs=server_max_bs,
    graph_optimization_config=graph_optimization_config,
    disable_sequence_parallel_moe=True,
)

if input_seq is None:
    prompt = "user: who are you?"
else:
    prompt = "hi " * input_seq
prompts = [prompt] * input_bs
for i in range(2):
    output = llm.generate(prompts=prompts, use_tqdm=True, sampling_params=sampling_params)

print(output)
