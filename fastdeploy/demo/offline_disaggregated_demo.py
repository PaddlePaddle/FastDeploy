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

import time
import os
import subprocess
import signal

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams


model_name_or_path = "./models/eb45t02/"



prefill_cmd = (f"FD_LOG_DIR=log_prefill CUDA_VISIBLE_DEVICES=0,1,2,3 python fastdeploy.entrypoints.openai.api_server.py"
    + f" --model {model_name_or_path} --port 9811"
    + f" --splitwise-role prefill --tensor-parallel-size 4"
    + f" --engine-worker-queue-port 6676 --cache-queue-port 55663")

prefill_instance = subprocess.Popen(
        prefill_cmd,
        stdout=subprocess.PIPE,
        shell=True,
        preexec_fn=os.setsid,
    )




# # 超参设置
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["FD_LOG_DIR"] = "log_decode"
sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
llm_decode = LLM(
    model=model_name_or_path, 
    tensor_parallel_size=4, 
    splitwise_role="decode",
    engine_worker_queue_port=6678, 
    innode_prefill_ports=[6676],
    cache_queue_port=55668
    )


output = llm_decode.generate(prompts=["who are you？", "what can you do？"], use_tqdm=True)
print(output)


os.killpg(prefill_instance.pid, signal.SIGTERM)