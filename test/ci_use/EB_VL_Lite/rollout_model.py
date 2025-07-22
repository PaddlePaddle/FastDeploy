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

import argparse
import difflib

from paddleformers.trl.llm_utils import init_dist_env

from fastdeploy.rl.rollout_config import RolloutModelConfig
from fastdeploy.rl.rollout_model import RolloutModel

_, ranks = init_dist_env()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
args = parser.parse_args()

# base result
model_path = args.model_path

# Usage example:
init_kwargs = {
    "model_name_or_path": model_path,
    "max_model_len": 32768,
    "tensor_parallel_size": ranks,
    "dynamic_load_weight": True,
    "load_strategy": "ipc_snapshot",
    "enable_mm": True,
    "quantization": "wint8",
}

rollout_config = RolloutModelConfig(**init_kwargs)
actor_eval_model = RolloutModel(rollout_config)

content = ""
for k, v in actor_eval_model.state_dict().items():
    content += f"{k}\n"
for k, v in actor_eval_model.get_name_mappings_to_training().items():
    content += f"{k}:{v}\n"


def compare_strings(a: str, b: str) -> bool:
    if a == b:
        print("✅ 两个字符串完全一致")
        return True

    print("❌ 字符串不一致，差异如下（上下文差异显示）：")
    diff = difflib.ndiff(a.splitlines(), b.splitlines())
    for line in diff:
        if line.startswith("- ") or line.startswith("+ "):
            print(line)

    return False


with open("baseline.txt", "r", encoding="utf-8") as f:
    baseline = f.read()
    assert compare_strings(baseline, content), (
        "In the unittest of RL scenario, your modification "
        "caused inconsistency in the content before and after. Please fix it. "
        "Can request assistance from yuanlehome or gzy19990617 (github id)."
    )
