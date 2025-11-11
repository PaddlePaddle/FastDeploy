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

model_name_or_path = "/workspace/ERNIE-4.5-0.3B-Paddle"

# 超参设置
sampling_params = SamplingParams(temperature=0.1, max_tokens=30, prompt_logprobs=100)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, enable_prefix_caching=False)
output = llm.generate(prompts="who are you？", use_tqdm=True, sampling_params=sampling_params)

print(output)
