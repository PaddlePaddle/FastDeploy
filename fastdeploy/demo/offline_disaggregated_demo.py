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

import multiprocessing
import os
import time

from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "baidu/ERNIE-4.5-21B-A3B-Paddle"


def start_decode(model_name_or_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["FD_LOG_DIR"] = "log_decode"
    llm_decode = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        splitwise_role="decode",
        engine_worker_queue_port=6678,
        innode_prefill_ports=[6676],
        cache_queue_port=55668,
    )
    return llm_decode


def start_prefill(model_name_or_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["FD_LOG_DIR"] = "log_prefill"
    LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        splitwise_role="prefill",
        engine_worker_queue_port=6677,
        cache_queue_port=55667,
    )


def main():
    prefill = multiprocessing.Process(target=start_prefill, args=(model_name_or_path,)).start()
    time.sleep(10)
    llm_decode = start_decode(model_name_or_path)

    output = llm_decode.generate(prompts=["who are you？", "what can you do？"], use_tqdm=True)
    print(output)

    prefill.join()


if __name__ == "__main__":
    main()
