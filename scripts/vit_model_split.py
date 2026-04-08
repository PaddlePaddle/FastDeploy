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

import argparse
import os

import paddle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="./",
    type=str,
    required=True,
    help="The directory of model.",
)
parser.add_argument(
    "--output_path",
    default="./",
    type=str,
    help="The directory of splited model",
)
parser.add_argument("--model_degree", default=4, type=int, help="Input model mp degree.")
args = parser.parse_args()

hidden_size = 1280
kv_num_heads = 16
head_dim = 80

input_model_state_dict = paddle.load(os.path.join(args.model_path, "model_state.pdparams"))

for i in range(args.model_degree):
    static_dict = {}
    for k, v in input_model_state_dict.items():
        if "qkv.weight" in k:
            static_dict[k] = (
                input_model_state_dict[k]
                .reshape([hidden_size, 3, kv_num_heads, head_dim])
                .split(args.model_degree, axis=-2)[i]
                .reshape([hidden_size, -1])
            )
        elif "qkv.bias" in k:
            static_dict[k] = (
                input_model_state_dict[k]
                .reshape([3, kv_num_heads, head_dim])
                .split(args.model_degree, axis=-2)[i]
                .reshape([-1])
            )
        elif "attn.proj.weight" in k:
            static_dict[k] = input_model_state_dict[k].split(args.model_degree, axis=-2)[i]
        elif "fc1.weight" in k:
            static_dict[k] = input_model_state_dict[k].split(args.model_degree, axis=-1)[i]
        elif "fc1.bias" in k:
            static_dict[k] = input_model_state_dict[k].split(args.model_degree, axis=-1)[i]
        elif "fc2.weight" in k:
            static_dict[k] = input_model_state_dict[k].split(args.model_degree, axis=-2)[i]
        else:
            static_dict[k] = v

    paddle.save(
        static_dict,
        os.path.join(args.model_path, f"model_state_tp0{i}.pdparams"),
    )
