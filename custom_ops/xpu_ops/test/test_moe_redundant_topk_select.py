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

from fastdeploy.model_executor.ops.xpu import moe_redundant_topk_select


def ref_moe_topk_select(gating_logits, bias, moe_topk, apply_norm_weight):
    assert apply_norm_weight is True

    def _softmax(x):
        axis = 1
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    softmax_logits = _softmax(gating_logits)
    softmax_logits_with_bias = np.copy(softmax_logits)
    if bias is not None:
        softmax_logits_with_bias += bias.reshape([1, -1])
    sorted_indices = np.argsort(softmax_logits_with_bias, axis=1, kind="stable")[:, ::-1]
    topk_ids = sorted_indices[:, :moe_topk]
    topk_weights = np.take_along_axis(softmax_logits, topk_ids, axis=1)
    topk_weights = topk_weights[:, :moe_topk]
    topk_weights /= np.sum(topk_weights, axis=1, keepdims=True)
    return topk_ids, topk_weights


def generate_expert_in_rank_num(num_values, extra_num):
    if num_values <= 0:
        return np.array([])
    # 一次性生成所有随机索引
    indices = np.random.randint(0, num_values, extra_num)
    # 使用 bincount 统计频率（向量化操作）
    bin_counts = np.bincount(indices, minlength=num_values)
    # 结果 = 基础值1 + 额外增加值
    return 1 + bin_counts


def generate_expert_id_to_ep_rank(expert_in_rank_num_list, num_rank, redundant_num_plus_one):
    num_expert = expert_in_rank_num_list.size
    redundant_num = redundant_num_plus_one - 1
    # 生成随机排名ID (一次性生成)
    rank_idx = np.random.randint(0, num_rank, num_expert)
    # 初始化结果矩阵 (-1 表示未分配)
    expert_id_to_rank_id = np.full((num_expert, redundant_num + 1), -1, dtype=int)
    # 初始分配 - 每个专家分配一个基础ID
    expert_ids = np.arange(num_expert)
    expert_id_to_rank_id[expert_ids, 0] = rank_idx
    if redundant_num > 0:
        positions = np.ones(num_expert, dtype=int)
        for expert_id in range(expert_in_rank_num_list.size):
            repeat_num = expert_in_rank_num_list[expert_id]
            while repeat_num > 1:
                rank_idx = np.random.randint(0, num_rank)
                expert_id_to_rank_id[expert_id][positions[expert_id]] = rank_idx
                positions[expert_id] += 1
                repeat_num -= 1
    return expert_id_to_rank_id


def generate_rank_to_id(id_to_rank, rank_num):
    max_rank = -1
    for ranks in id_to_rank:
        if ranks:
            current_max = max(ranks)
            if current_max > max_rank:
                max_rank = current_max
    if max_rank < 0 or max_rank >= rank_num:
        return []

    rank_to_id = [[] for _ in range(rank_num)]
    for id_val, ranks in enumerate(id_to_rank):
        for r in ranks:
            if r < 0:  # 忽略负数值
                continue
            if r < len(rank_to_id):  # 确保索引在有效范围内
                rank_to_id[r].append(id_val)
    return rank_to_id


def my_sort(key_arr, val_arr):
    if key_arr.shape != val_arr.shape:
        return None, None
    # 不转换整个数组，逐行处理
    sorted_keys = np.empty_like(key_arr)
    sorted_vals = np.empty_like(val_arr)

    for i in range(key_arr.shape[0]):
        keys = key_arr[i]
        vals = val_arr[i]
        idx = np.lexsort((keys, vals))
        sorted_keys[i] = keys[idx]
        sorted_vals[i] = vals[idx]

    return sorted_keys, sorted_vals


if __name__ == "__main__":
    seed = np.random.randint(1, 1e9)
    print(f"numpy random seed={seed}")
    np.random.seed(seed)

    rank_num = 8
    token_num = 1215
    expert_num = 256
    moe_topk = 8
    redundant_ep_rank_num_plus_one = 1  # no redundant experts
    apply_norm_weight = True
    enable_softmax_top_k_fused = True
    gating_logits = np.random.random([token_num, expert_num]).astype("float32")
    bias = np.random.random([expert_num]).astype("float32")
    expert_in_rank_num_list = generate_expert_in_rank_num(expert_num, redundant_ep_rank_num_plus_one - 1)
    print(f"expert_in_rank_num_list={expert_in_rank_num_list}")
    expert_id_to_ep_rank_array = generate_expert_id_to_ep_rank(
        expert_in_rank_num_list, rank_num, redundant_ep_rank_num_plus_one
    )
    tokens_per_expert_stats_list = np.random.randint(0, 20, size=(expert_num))
    print(f"expert_id_to_ep_rank_array={expert_id_to_ep_rank_array}")
    print(f"tokens_per_expert_stats_list={tokens_per_expert_stats_list}")

    #    ref_topk_ids, ref_topk_weights = ref_moe_topk_select(
    #        gating_logits, bias, moe_topk, apply_norm_weight
    #    )

    gating_logits = paddle.to_tensor(gating_logits).astype("float32")
    expert_id_to_ep_rank_array = paddle.to_tensor(expert_id_to_ep_rank_array).astype("int32")
    expert_in_rank_num_list = paddle.to_tensor(expert_in_rank_num_list).astype("int32")
    tokens_per_expert_stats_list = paddle.to_tensor(tokens_per_expert_stats_list).astype("int32")
    if bias is not None:
        bias = paddle.to_tensor(bias).astype("float32")

    gating_logits_ref = gating_logits.cpu()
    expert_id_to_ep_rank_array_ref = expert_id_to_ep_rank_array.cpu()
    expert_in_rank_num_list_ref = expert_in_rank_num_list.cpu()
    tokens_per_expert_stats_list_ref = tokens_per_expert_stats_list.cpu()
    bias_ref = None
    if bias is not None:
        bias_ref = bias.cpu()

    topk_ids, topk_weights = moe_redundant_topk_select(
        gating_logits,
        expert_id_to_ep_rank_array,
        expert_in_rank_num_list,
        tokens_per_expert_stats_list,
        bias,
        moe_topk,
        apply_norm_weight,
        enable_softmax_top_k_fused,
        redundant_ep_rank_num_plus_one,
    )
    topk_ids_ref, topk_weights_ref = moe_redundant_topk_select(
        gating_logits_ref,
        expert_id_to_ep_rank_array_ref,
        expert_in_rank_num_list_ref,
        tokens_per_expert_stats_list_ref,
        bias_ref,
        moe_topk,
        apply_norm_weight,
        enable_softmax_top_k_fused,
        redundant_ep_rank_num_plus_one,
    )

    topk_ids_np, topk_weights_np, tokens_per_expert_stats_list_np = (
        topk_ids.numpy(),
        topk_weights.numpy(),
        tokens_per_expert_stats_list.numpy(),
    )
    topk_ids_ref, topk_weights_ref, tokens_per_expert_stats_list_ref = (
        topk_ids_ref.numpy(),
        topk_weights_ref.numpy(),
        tokens_per_expert_stats_list_ref.numpy(),
    )
    sorted_topk_ids, sorted_topk_weights = my_sort(topk_ids_np, topk_weights_np)
    sorted_topk_ids_ref, sorted_topk_weights_ref = my_sort(topk_ids_ref, topk_weights_ref)

    assert np.array_equal(
        tokens_per_expert_stats_list_np, tokens_per_expert_stats_list_ref
    ), f"\ntokens_per_expert_stats_list:\n{tokens_per_expert_stats_list.numpy()}\ntokens_per_expert_stats_list_ref:\n{tokens_per_expert_stats_list_ref}"
    assert np.array_equal(
        sorted_topk_ids, sorted_topk_ids_ref
    ), f"\ntopk_ids:\n{topk_ids.numpy()}\ntopk_ids_ref:\n{topk_ids_ref}"
    assert np.allclose(
        sorted_topk_weights, sorted_topk_weights_ref
    ), f"\ntopk_weights:\n{topk_weights.numpy()}\ntopk_weights_ref:\n{topk_weights_ref}"

    print("Passed all tests.")
