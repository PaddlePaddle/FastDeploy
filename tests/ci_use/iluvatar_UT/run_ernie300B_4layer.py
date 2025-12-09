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

import functools
import sys
import threading

from fastdeploy import LLM, SamplingParams
from fastdeploy.utils import set_random_seed


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


@timeout(80)
def offline_infer_check():
    set_random_seed(123)

    prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.00001, max_tokens=16)
    llm = LLM(
        model="/data1/fastdeploy/ERNIE_300B_4L",
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
        49080,
        94717,
        82966,
        99140,
        31615,
        51497,
        94851,
        60764,
        10889,
        2,
    ], f"{outputs[0].outputs.token_ids}"
    print("PASSED")


if __name__ == "__main__":
    try:
        result = offline_infer_check()
        sys.exit(0)
    except TimeoutError:
        print(
            "The timeout exit may be due to multiple processes sharing the "
            "same gpu card. You can check this using ixsmi on the device."
        )
        sys.exit(124)
    except Exception:
        sys.exit(1)
