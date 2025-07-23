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

import os


def check_safetensors_model(model_dir: str):
    """
    model_dir : the directory of the model
    Check whther the model is safetensors format
    """
    model_files = list()
    all_files = os.listdir(model_dir)
    for x in all_files:
        if x.startswith("model") and x.endswith(".safetensors"):
            model_files.append(x)

    is_safetensors = len(model_files) > 0
    if not is_safetensors:
        return False

    if len(model_files) == 1 and model_files[0] == "model.safetensors":
        return True
    try:
        # check all the file exists
        safetensors_num = int(model_files[0].strip(".safetensors").split("-")[-1])
        flags = [0] * safetensors_num
        for x in model_files:
            current_index = int(x.strip(".safetensors").split("-")[1])
            flags[current_index - 1] = 1
        assert (
            sum(flags) == safetensors_num
        ), f"Number of safetensor files should be {len(model_files)}, but now it's {sum(flags)}"
    except Exception as e:
        raise Exception(f"Failed to check unified checkpoint, details: {e}.")
    return is_safetensors
