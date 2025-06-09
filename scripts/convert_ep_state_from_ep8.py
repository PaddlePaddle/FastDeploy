"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""

import paddle
import paddle.distributed as dist
from glob import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
args = parser.parse_args()

rank = dist.get_rank()
ep_num = dist.get_world_size()
print("rank: ", rank)
# merge tpn -> tp1
model_dir = args.model_dir
save_merged_pp_dir = os.path.join(model_dir, "merged_tp1_state_split")
os.makedirs(save_merged_pp_dir, exist_ok=True)

model_path_pp = glob(os.path.join(model_dir, "shangxianv1_ep_hadamard_quantmodel_to_eval_pp*"))
for p in model_path_pp:
    model_path_ep = os.path.join(p, f"model_state.ep0{rank}.pdparams")
    print(p, model_path_ep)

    state_dicts = paddle.load(model_path_ep, return_numpy=True)

    print("merge ep")
    print("p: ", p)
    for k, v in state_dicts.items():
        v = paddle.to_tensor(v)
        if "mlp.experts" in k:
            k_list = k.split(".")
            export_id = rank * ep_num + int(k_list[5])
            k_list[5] = str(export_id)
            k = ".".join(k_list)
            print(f"key: {k}")
            save_split_path = os.path.join(save_merged_pp_dir, k)
            paddle.save(v, save_split_path)
        elif rank == 0:
            save_split_path = os.path.join(save_merged_pp_dir, k)
            paddle.save(paddle.to_tensor(v), save_split_path)
    print(f"merge {p} end")
print("merge end")
