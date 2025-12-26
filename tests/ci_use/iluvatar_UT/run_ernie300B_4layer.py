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

import os
import sys

from fastdeploy import LLM, SamplingParams
from fastdeploy.utils import set_random_seed

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, tests_dir)

from ci_use.iluvatar_UT.utils import TIMEOUT_MSG, timeout


@timeout(80)
def offline_infer_check():
    set_random_seed(123)

    prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.00001, max_tokens=16)
    llm = LLM(
        model="/model_data/ERNIE_300B_4L",
        tensor_parallel_size=2,
        max_model_len=8192,
        quantization="wint8",
        block_size=16,
    )
    outputs = llm.generate(prompts, sampling_params)

    assert outputs[0].outputs.token_ids == [
        23768,
        97000,
        47814,
        59335,
        68170,
        183,
        97404,
        100088,
        36310,
        95633,
        95913,
        41459,
        95049,
        94970,
        96840,
        2,
    ], f"{outputs[0].outputs.token_ids}"
    print("PASSED")


if __name__ == "__main__":
    try:
        result = offline_infer_check()
        sys.exit(0)
    except TimeoutError:
        print(TIMEOUT_MSG)
        sys.exit(124)
    except Exception:
        sys.exit(1)
