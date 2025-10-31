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

from paddleformers.trl.llm_utils import init_dist_env

from fastdeploy.rl.rollout_config import RolloutModelConfig
from fastdeploy.rl.rollout_model import RolloutModel

_, ranks = init_dist_env()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
parser.add_argument("--baseline_path", type=str, required=True, help="Path to the baseline path")
parser.add_argument("--quantization", type=str, default=None, help="Quantization")
parser.add_argument("--enable_mm", action="store_true", required=False, help="Flags to enable multi-modal model")
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
    "quantization": args.quantization,
}
if args.enable_mm:
    init_kwargs["enable_mm"] = True


rollout_config = RolloutModelConfig(**init_kwargs)
actor_eval_model = RolloutModel(rollout_config)

content = "".join(
    sorted(
        [f"{k}\n" for k, v in actor_eval_model.state_dict().items()]
        + [f"{k}:{v}\n" for k, v in actor_eval_model.get_name_mappings_to_training().items()]
    )
)


def compare_strings_line_by_line(a: str, b: str) -> bool:
    """
    Compare two multiline strings line by line.

    Returns:
        True if all lines match exactly in order and content.
        False if any line differs or the number of lines is not equal.
    """
    a_lines = a.splitlines()
    b_lines = b.splitlines()

    if len(a_lines) != len(b_lines):
        print(f"❌ Mismatch in number of lines: expected {len(a_lines)}, but got {len(b_lines)}.")
        return False

    for i, (line_a, line_b) in enumerate(zip(a_lines, b_lines)):
        if line_a != line_b:
            print(f"❌ Difference found on line {i + 1}:")
            print(f"  Expected: {repr(line_a)}")
            print(f"  Actual  : {repr(line_b)}")
            return False

    print("✅ All lines match exactly.")
    return True


with open(args.baseline_path, "r", encoding="utf-8") as f:
    baseline = f.read()
    assert compare_strings_line_by_line(baseline, content), (
        "In the unittest of RL scenario, your modification "
        "caused inconsistency in the content before and after. Please fix it. "
        "Can request assistance from yuanlehome or gzy19990617 (github id)."
    )
