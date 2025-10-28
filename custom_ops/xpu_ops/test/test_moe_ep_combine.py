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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import ep_moe_expert_combine

np.random.seed(2025)


def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def create_moe_index(token_num, moe_topk, expand_token_num):
    total_positions = token_num * moe_topk
    positions = np.random.choice(total_positions, size=expand_token_num, replace=False)
    rows = positions // moe_topk
    cols = positions % moe_topk
    values = np.random.permutation(expand_token_num)

    # moe_index is the output of moe_ep_dispatch
    # the val in moe_index is the row in ffn_out for corresponding token and expert, -1 means invalid
    moe_index = np.full((token_num, moe_topk), -1)
    for i in range(expand_token_num):
        moe_index[rows[i], cols[i]] = values[i]
    return moe_index


# 1) preparation
token_num = 10
moe_topk = 8
hidden_dim = 128
expand_token_num = 30

ffn_out = np.random.random((expand_token_num, hidden_dim))
moe_index = create_moe_index(token_num, moe_topk, expand_token_num)
moe_weights = np.random.random((token_num, moe_topk))
moe_weights = np_softmax(moe_weights)
moe_weights[moe_index == -1] = -1
print(f"ffn_out:\n{ffn_out}")
print(f"moe_index:\n{moe_index}")
print(f"moe_weights:\n{moe_weights}")

# 2) np calculation
combined_out_np = np.zeros((token_num, hidden_dim))
for token_idx, item in enumerate(moe_index):
    for topk_idx, ffn_out_row in enumerate(item):
        if ffn_out_row == -1:
            continue
        combined_out_np[token_idx] += ffn_out[ffn_out_row] * moe_weights[token_idx][topk_idx]
print(f"combined_out_np:\n{combined_out_np}")

# 3) xpu calculation
dtype = "bfloat16"
ffn_out_pd = paddle.to_tensor(ffn_out, dtype=dtype)
moe_index_pd = paddle.to_tensor(moe_index, dtype="int32")
moe_weights_pd = paddle.to_tensor(moe_weights, dtype=dtype)
combined_out_pd = ep_moe_expert_combine(
    ffn_out_pd,
    moe_index_pd,
    moe_weights_pd,
    moe_index_pd.shape[0],
    ffn_out_pd.shape[0],
    ffn_out_pd.shape[1],
    moe_index_pd.shape[1],
)

# comparison
# print("moe_index:\n", moe_index)
# print("moe_weights:\n", moe_weights)
# print("combined_out_np:\n", combined_out_np)
# print("combined_out_pd:\n", combined_out_pd)
combined_out_pd = combined_out_pd.astype("float32").numpy()
avg_diff = np.sum(np.abs(combined_out_pd - combined_out_np)) / combined_out_pd.size
assert (
    avg_diff < 2e-3
), f"avg_diff: {avg_diff}\n combined_out_np:\n{combined_out_np}\n combined_out_pd:\n{combined_out_pd}\n"
print(f"[Passed] avg_diff: {avg_diff}")
