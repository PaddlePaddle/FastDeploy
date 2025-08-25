# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.ops.xpu import speculate_get_token_penalty_multi_scores

paddle.seed(2023)


def allclose_any(a, b, rtol=1e-5, atol=1e-5, equal_nan=False):
    """检查两个数组是否满足任意一个容差条件"""
    condition = (np.abs(a - b) <= atol) | (np.abs(a - b) <= rtol * np.abs(b))  # 绝对误差条件  # 相对误差条件
    print(f"cond={condition}")
    # 处理 NaN（如果需要）
    if equal_nan:
        nan_mask = np.isnan(a) & np.isnan(b)
        condition = condition | nan_mask
    # 检查所有元素是否都满足条件
    return np.all(condition)


def find_max_diff(arr1, arr2):
    """找出两个数组元素差值的最大值及其索引
    返回:
        max_diff (float): 最大绝对值差
        index (tuple): 最大值的位置索引
        actual_diff (float): 实际差值（带符号）
    """
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    flat_idx = np.argmax(abs_diff)
    idx = np.unravel_index(flat_idx, arr1.shape)
    return abs_diff[idx], idx, diff[idx], arr1[idx], arr2[idx]


def test_main(
    pre_ids,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id,
    seq_len_this_time,
    output_padding_offset,
    output_cum_offsets,
    max_seq_len,
):
    pre_ids_ref = pre_ids.cpu()
    logits_ref = logits.cpu()
    penalty_scores_ref = penalty_scores.cpu()
    frequency_scores_ref = frequency_scores.cpu()
    presence_scores_ref = presence_scores.cpu()
    temperatures_ref = temperatures.cpu()
    bad_tokens_ref = bad_tokens.cpu()
    cur_len_ref = cur_len.cpu()
    min_len_ref = min_len.cpu()
    eos_token_id_ref = eos_token_id.cpu()
    seq_len_this_time_ref = seq_len_this_time.cpu()
    output_padding_offset_ref = output_padding_offset.cpu()
    output_cum_offsets_ref = output_cum_offsets.cpu()

    speculate_get_token_penalty_multi_scores(
        pre_ids,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
        seq_len_this_time,
        output_padding_offset,
        output_cum_offsets,
        max_seq_len,
    )
    speculate_get_token_penalty_multi_scores(
        pre_ids_ref,
        logits_ref,
        penalty_scores_ref,
        frequency_scores_ref,
        presence_scores_ref,
        temperatures_ref,
        bad_tokens_ref,
        cur_len_ref,
        min_len_ref,
        eos_token_id_ref,
        seq_len_this_time_ref,
        output_padding_offset_ref,
        output_cum_offsets_ref,
        max_seq_len,
    )
    logits_ref_np = logits_ref.astype("float32").numpy()
    logits_np = logits.astype("float32").numpy()
    np.set_printoptions(threshold=10000)
    # print(f"logits_ref={logits_ref_np[:50,:100]}")
    # print(f"logits={logits_np[:50,:100]}")

    diff_logits = np.sum(np.abs(logits_ref_np - logits_np))
    print("diff_logits\n", diff_logits)
    abs_diff, idx, diff, val1, val2 = find_max_diff(logits_ref_np, logits_np)
    print(f"abs_diff={abs_diff}, index={idx}, diff={diff}, {val1} vs {val2}")
    assert allclose_any(logits_ref_np, logits_np, 1e-5, 1e-5)
    # assert np.allclose(logits_ref_np, logits_np, 1e-5, 1e-5)


# gtest_speculate_token_penalty_multi_scores<float>(api::kXPU3, "GM", "GM", "GM", "GM", "GM", "GM", "GM", "GM", "GM", "GM", "GM", "GM",
#                                                   84, 100352, 12288, 1, 1, 54, 32768);


def miain():
    seed = np.random.randint(1, 1e9)
    print(f"random seed is {seed}")
    np.random.seed(seed)

    bs = 64
    max_seq_len = 32768  # 1024 #2048 #8192
    data_type = "float32"  # bfloat16 or float32

    # prepare output_padding_offset and output_cum_offsets
    tokens = [1] * bs
    token_num = np.sum(tokens)
    print(f"bs={bs}, tokens={tokens}, token_num={token_num}")
    output_padding_offset = []
    output_cum_offsets = [0]
    opo_offset = 0
    for bid in range(bs):
        ts = tokens[bid]
        for i in range(ts):
            output_padding_offset.append(opo_offset)
        opo_offset += max_seq_len - ts
        output_cum_offsets.append(opo_offset)
    output_cum_offsets = output_cum_offsets[:-1]
    # print(f"output_padding_offset={output_padding_offset}")
    # print(f"output_cum_offsets={output_cum_offsets}")
    output_padding_offset = paddle.to_tensor(output_padding_offset, "int32")
    output_cum_offsets = paddle.to_tensor(output_cum_offsets, "int32")

    # prepare pre_ids and logits
    pre_ids_len = 12288
    # pre_ids_len = np.random.randint(1, 512)
    logits_len = 100352
    # print(f"pre_ids_len={pre_ids_len}, logits_len={logits_len}")
    pre_ids = np.random.randint(1, logits_len, size=(bs, pre_ids_len))
    negative_start = np.random.randint(1, pre_ids_len + 1, size=(bs))
    print(negative_start)
    for i in range(bs):
        pre_ids[:, negative_start[i] :] = -1
    pre_ids = paddle.to_tensor(pre_ids).astype("int64")
    # logits = paddle.to_tensor(
    #     np.float32(np.random.random([token_num, logits_len]))
    # ).astype(data_type)
    logits = paddle.to_tensor(np.float32(np.zeros([token_num, logits_len]))).astype(data_type)
    # prepare other params
    penalty_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
    frequency_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
    presence_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
    temperatures = paddle.to_tensor(np.random.random([bs])).astype("float32")
    bad_tokens = paddle.to_tensor(np.random.randint(0, 101, size=(1))).astype("int64")
    cur_len = paddle.to_tensor(np.random.randint(1, 50, size=(bs))).astype("int64")
    min_len = paddle.to_tensor(np.random.randint(1, 50, size=(bs))).astype("int64")
    eos_token_id = paddle.to_tensor(np.random.randint(1, 101, size=(1))).astype("int64")
    seq_len_this_time = paddle.to_tensor(
        np.random.randint(0, 1, size=(bs)), "int32"
    )  # value of seq_len_this_time is useless

    # test
    test_main(
        pre_ids,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
        seq_len_this_time,
        output_padding_offset,
        output_cum_offsets,
        max_seq_len,
    )


if __name__ == "__main__":
    for i in range(10):
        miain()
