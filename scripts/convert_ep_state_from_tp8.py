"""
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
"""

import paddle
import paddle.distributed as dist
import pdb
from glob import glob
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
args = parser.parse_args()

rank = dist.get_rank()
print("rank: ", rank)
# merge tpn -> tp1
model_dir = args.model_dir
model_path_pp = glob(os.path.join(model_dir, f"pp{rank}"))
model_path_pp_tp = []
for p in model_path_pp:
    model_path_tp = glob(os.path.join(p, "model_state*"))
    model_path_tp = sorted(model_path_tp)
    save_merged_pp_path = os.path.join(p, "merged_tp1_state.pdparams")
    save_merged_pp_dir = os.path.join(model_dir, "merged_tp1_state_split")
    os.makedirs(save_merged_pp_dir, exist_ok=True)
    print(p, model_path_tp)

    state_dicts = [paddle.load(path, return_numpy=True) for path in model_path_tp]
    state = state_dicts[0]

    print("merge tp")
    print("p: ", p)
    for k, v in state.items():
        save_split_path = os.path.join(save_merged_pp_dir, k)
        state_now = []
        for i in range(len(state_dicts)):
            state_now.append(state_dicts[i][k])
        print("k: ", k, ", v.shape: ", v.shape)
        if "qkv_proj" in k:
            """not need prmt"""
            # qkv not prmt
            ori_q = [s[:, :1024] for s in state_now]
            ori_k = [s[:, 1024:1152] for s in state_now]
            ori_v = [s[:, 1152:] for s in state_now]
            new_q = np.concatenate(ori_q, axis=1)
            new_k = np.concatenate(ori_k, axis=1)
            new_v = np.concatenate(ori_v, axis=1)
            print(new_q.shape)
            print(new_k.shape)
            print(new_v.shape)
            new_w = np.concatenate([new_q, new_k, new_v], axis=1)
            # new_w = np.concatenate(state_now, axis=1)
        elif "o_proj" in k or "down_proj" in k:
            new_w = np.concatenate(state_now, axis=0)
        elif "embed_tokens" in k:
            new_w = np.concatenate(state_now, axis=0)
        elif "up_gate_proj" in k:
            dim = state_now[0].shape[1]
            half_ffn1_1 = [s[:, :(dim // 2)] for s in state_now]
            half_ffn1_2 = [s[:, (dim // 2):] for s in state_now]
            new_ffn1_1 = np.concatenate(half_ffn1_1, axis=1)
            new_ffn1_2 = np.concatenate(half_ffn1_2, axis=1)
            new_w = np.concatenate([new_ffn1_1, new_ffn1_2], axis=1)
        elif "lm_head" in k or "mtp_linear_proj" in k:
            new_w = np.concatenate(state_now, axis=1)
        else:
            new_w = v
        print("merged_shape: ", new_w.shape)
        paddle.save(paddle.to_tensor(new_w), save_split_path)
    print("merge end")