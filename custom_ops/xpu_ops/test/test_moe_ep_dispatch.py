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

from fastdeploy.model_executor.ops.xpu import ep_moe_expert_dispatch

np.random.seed(2025)


def ep_moe_expert_dispatch_cpu(input, topk_ids, topk_weights, token_nums_per_expert, token_nums_this_rank):
    m, n = input.shape[0], input.shape[1]
    topk = topk_ids.shape[1]
    expert_num = len(token_nums_per_expert)
    expert_per_rank = expert_num

    permute_input = np.full((token_nums_this_rank, n), 0.0, dtype=np.float32)
    permute_indices_per_token = np.full((m, topk), -1, dtype=np.int32)
    recv_num_tokens_per_expert_list_cumsum = np.full(expert_num + 1, 0, dtype=np.int32)
    dst_indices = np.full((expert_num, m), -1, dtype=np.int32)
    cumsum_idx = np.full(expert_num, 0, dtype=np.int32)
    offset = 0
    for expert_id in range(expert_per_rank):
        for token_id in range(m):
            for k in range(topk):
                cur_index = topk_ids[token_id, k]
                if cur_index == expert_id:
                    permute_indices_per_token[token_id, k] = offset
                    permute_input[offset, :] = input[token_id, :]
                    offset += 1
        recv_num_tokens_per_expert_list_cumsum[expert_id + 1] = offset
    return (
        permute_input,
        permute_indices_per_token,
        recv_num_tokens_per_expert_list_cumsum,
        topk_weights,
        dst_indices,
        cumsum_idx,
    )


def create_moe_index(token_num, topk, expert_num):
    topk_ids = np.full((token_num, topk), -1, dtype=np.int32)
    token_nums_per_expert = np.full(expert_num_per_rank, 0, dtype=np.int32)
    token_all_num = 0
    for i in range(topk_ids.shape[0]):
        pos = np.random.choice(np.arange(topk), np.random.randint(1, topk + 1), replace=False)
        token_all_num += len(pos)
        for j in pos:
            topk_ids[i, j] = np.random.choice(expert_num, replace=False)
            token_nums_per_expert[topk_ids[i, j]] += 1
    return token_all_num, topk_ids, list(token_nums_per_expert)


# 1) preparation
token_num = 7
expert_num_per_rank = 4
topk = 8
hidden_dim = 8192

input = np.random.random((token_num, hidden_dim))
token_nums_this_rank, topk_ids, token_nums_per_expert = create_moe_index(token_num, topk, expert_num_per_rank)
topk_weights = np.random.random((token_num, topk))
print(f"input:\n{input}")
print(f"token_nums_this_rank:\n{token_nums_this_rank}")
print(f"topk_ids:\n{topk_ids}")
print(f"token_nums_per_expert:\n{token_nums_per_expert}")
print(f"topk_weights:\n{topk_weights}")

dtype = "bfloat16"
input_xpu = paddle.to_tensor(input, dtype=dtype)
topk_ids_xpu = paddle.to_tensor(topk_ids)
topk_weights_xpu = paddle.to_tensor(topk_weights)

# 2) cpu calculation
(
    permute_input,
    permute_indices_per_token,
    recv_num_tokens_per_expert_list_cumsum,
    dst_weights,
    dst_indices,
    cumsum_idx,
) = ep_moe_expert_dispatch_cpu(input, topk_ids, topk_weights, token_nums_per_expert, token_nums_this_rank)
print(f"permute_input:\n{permute_input}")
print(f"permute_indices_per_token:\n{permute_indices_per_token}")
print(f"recv_num_tokens_per_expert_list_cumsum:\n{recv_num_tokens_per_expert_list_cumsum}")
print(f"dst_weights:\n{dst_weights}")
print(f"dst_indices:\n{dst_indices}")
print(f"cumsum_idx:\n{cumsum_idx}")

# 3) xpu calculation
(
    permute_input_xpu,
    permute_indices_per_token_xpu,
    recv_num_tokens_per_expert_list_cumsum_xpu,
    dst_weights_xpu,
    expand_input_scales,
) = ep_moe_expert_dispatch(
    input_xpu,
    topk_ids_xpu,
    topk_weights_xpu,
    None,
    token_nums_per_expert,
    token_nums_this_rank,
    "weight_only_int8",
)

# comparison
permute_input_xpu = permute_input_xpu.astype("float32").numpy()
permute_indices_per_token_xpu = permute_indices_per_token_xpu.numpy()
recv_num_tokens_per_expert_list_cumsum_xpu = recv_num_tokens_per_expert_list_cumsum_xpu.numpy()

diff = np.sum(np.abs(permute_input - permute_input_xpu)) / permute_input.size
assert diff < 1e-2, f"diff: {diff}\n permute_input:\n {permute_input}\n permute_input_xpu:\n {permute_input_xpu}\n"

assert (
    permute_indices_per_token == permute_indices_per_token_xpu
).all(), f"permute_indices_per_token:\n {permute_indices_per_token}\n permute_indices_per_token_xpu:\n {permute_indices_per_token_xpu}\n"

assert (
    recv_num_tokens_per_expert_list_cumsum == recv_num_tokens_per_expert_list_cumsum_xpu
).all(), f"recv_num_tokens_per_expert_list_cumsum:\n {recv_num_tokens_per_expert_list_cumsum}\n recv_num_tokens_per_expert_list_cumsum_xpu:\n {recv_num_tokens_per_expert_list_cumsum_xpu}\n"

print("ep_moe_expert_dispatch test success!")
