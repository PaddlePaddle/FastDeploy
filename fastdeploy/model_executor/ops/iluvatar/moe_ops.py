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

from __future__ import annotations

import paddle

try:
    from paddle.nn.functional import swiglu
except ImportError:

    def swiglu(x):
        x, y = paddle.chunk(x, chunks=2, axis=-1)
        return paddle.nn.functional.silu(x) * y


def iluvatar_moe_expert_ffn(
    x,
    w1,
    w2,
    topk_idx,
    topk_weights,
    ep_size,
    ep_rank,
    tp_size,
    tp_rank,
    group,
    expert_num,
    hidden_size,
    inter_size,
    act_type="swiglu",
    weight_dtype="float16",
):
    """
    iluvatar moe expert ffn
    """
    # 1. 路由输入数据
    # 计算每个rank处理的experts数量
    num_local_experts = expert_num // ep_size
    # 计算当前rank处理的experts范围
    expert_start_idx = ep_rank * num_local_experts
    expert_end_idx = expert_start_idx + num_local_experts

    # 找出当前rank需要处理的token
    # mask [batch_size, top_k]
    mask = (topk_idx >= expert_start_idx) & (topk_idx < expert_end_idx)
    # [num_tokens] 哪些token被选中
    token_indices = paddle.nonzero(mask)
    if token_indices.shape[0] == 0:
        return paddle.zeros_like(x)

    # 获取选中token的原始索引和对应的expert索引
    batch_indices = token_indices[:, 0]
    k_indices = token_indices[:, 1]

    # 获取选中的experts id，并转为本地索引
    selected_experts = topk_idx[batch_indices, k_indices] - expert_start_idx
    # 获取对应的权重
    selected_weights = topk_weights[batch_indices, k_indices]

    # [num_tokens, hidden_size]
    selected_x = x.index_select(batch_indices, axis=0)

    # 2. 计算FFN
    # 准备输出tensor
    final_output = paddle.zeros_like(x)

    # 遍历本地每个expert进行计算
    for i in range(num_local_experts):
        # 找出当前expert负责的token
        expert_mask = selected_experts == i
        if not expert_mask.any():
            continue

        # [num_expert_tokens]
        expert_indices = paddle.nonzero(expert_mask).flatten()
        # [num_expert_tokens, hidden_size]
        expert_input = selected_x.index_select(expert_indices, axis=0)

        # 获取当前expert的权重
        # w1: [expert_num, inter_size * 2, hidden_size]
        # w2: [expert_num, hidden_size, inter_size]
        curr_w1 = w1[i].t()  # [hidden_size, inter_size * 2]
        curr_w2 = w2[i].t()  # [inter_size, hidden_size]

        # FFN计算
        # [num_expert_tokens, inter_size * 2]
        ffn1_output = paddle.matmul(expert_input, curr_w1)

        # 激活函数
        if act_type == "swiglu":
            act_out = swiglu(ffn1_output)
        else:
            # 默认gelu
            act_out = paddle.nn.functional.gelu(ffn1_output)

        # [num_expert_tokens, hidden_size]
        ffn2_output = paddle.matmul(act_out, curr_w2)

        # 加权
        # [num_expert_tokens, 1]
        expert_weights = selected_weights.index_select(expert_indices, axis=0).unsqueeze(-1)
        weighted_output = ffn2_output * expert_weights

        # 累加到最终结果
        # 注意：这里需要将结果scatter回原来的位置
        # 获取在原始batch中的索引
        original_indices = batch_indices.index_select(expert_indices, axis=0)
        final_output.index_add_(original_indices, weighted_output, axis=0)

    # 3. AllReduce聚合结果 (如果使用了EP)
    if ep_size > 1:
        paddle.distributed.all_reduce(final_output, group=group)

    return final_output
