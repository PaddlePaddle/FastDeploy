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

model_name_or_path = "/home/zexuli/Models/Qwen3-0.6B"

# 超参设置
sampling_params = SamplingParams(temperature=0.1,top_p=0.7,top_k=10,min_p=0.1)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1,reasoning_parser="qwen3")
prompt = "北京天安门在哪里?"
messages = [{"role": "user", "content": prompt}]
output = llm.chat([messages],
                   sampling_params)

print(output)
