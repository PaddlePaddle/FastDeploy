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

from fastdeploy.model_executor.ops.xpu import draft_model_update


def run_paddle_test(device="cpu"):
    np.random.seed(42)
    paddle.seed(42)
    if device == "cpu":
        paddle.set_device(device)
    elif device == "xpu":
        paddle.set_device(device)
    else:
        raise ValueError(f"Invalid device: {device}")

    # 设置参数
    max_bsz = 128
    max_draft_token = 3
    pre_id_length = 3
    max_seq_len = 100
    max_base_model_draft_token = 4
    substep = 2

    # 创建随机张量
    inter_next_tokens = paddle.randint(1, 100, shape=(max_bsz, max_seq_len), dtype="int64")
    draft_tokens = paddle.randint(1, 100, shape=(max_bsz, max_draft_token), dtype="int64")
    pre_ids = paddle.randint(1, 100, shape=(max_bsz, pre_id_length), dtype="int64")
    seq_lens_this_time = paddle.randint(1, 2, shape=(max_bsz,), dtype="int32")
    seq_lens_encoder = paddle.randint(1, 10, shape=(max_bsz,), dtype="int32")
    seq_lens_decoder = paddle.randint(1, 10, shape=(max_bsz,), dtype="int32")
    step_idx = paddle.randint(1, 10, shape=(max_bsz,), dtype="int64")
    output_cum_offsets = paddle.randint(0, 2, shape=(max_bsz,), dtype="int32")
    output_cum_offsets[0] = 0  # 确保第一个偏移量为0
    stop_flags = paddle.zeros([max_bsz], dtype="bool")
    not_need_stop = paddle.zeros([1], dtype="bool")
    max_dec_len = paddle.randint(100, 102, shape=(max_bsz,), dtype="int64")
    end_ids = paddle.to_tensor([2], dtype="int64")
    base_model_draft_tokens = paddle.randint(1, 10, shape=(max_bsz, max_base_model_draft_token), dtype="int64")

    # 打印张量信息
    # print("inter_next_tokens shape:", inter_next_tokens.shape)
    # print("draft_tokens shape:", draft_tokens.shape)
    # print("pre_ids shape:", pre_ids.shape)
    # print("seq_lens_this_time shape:", seq_lens_this_time.shape)
    # print("seq_lens_encoder shape:", seq_lens_encoder.shape)
    # print("seq_lens_decoder shape:", seq_lens_decoder.shape)
    # print("step_idx shape:", step_idx.shape)
    # print("output_cum_offsets shape:", output_cum_offsets.shape)
    # print("stop_flags shape:", stop_flags.shape)
    # print("not_need_stop shape:", not_need_stop.shape)
    # print("max_dec_len shape:", max_dec_len.shape)
    # print("end_ids shape:", end_ids.shape)
    # print("base_model_draft_tokens shape:", base_model_draft_tokens.shape)

    # print("draft_tokens before update:", draft_tokens)
    # print("pre_ids before update:", pre_ids)
    draft_model_update(
        inter_next_tokens,
        draft_tokens,
        pre_ids,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        output_cum_offsets,
        stop_flags,
        not_need_stop,
        max_dec_len,
        end_ids,
        base_model_draft_tokens,
        max_seq_len,
        substep,
    )
    # print("draft_tokens after update:", draft_tokens)
    # print("pre_ids after update:", pre_ids)
    return (
        draft_tokens,
        pre_ids,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        stop_flags,
        not_need_stop,
        base_model_draft_tokens,
    )


if __name__ == "__main__":
    res_xpu = run_paddle_test("xpu")
    res_cpu = run_paddle_test()
    for idx in range(len(res_cpu)):
        # 将结果转换为numpy数组
        cpu_arr = res_cpu[idx].numpy()
        xpu_arr = res_xpu[idx].numpy()

        # 检查是否为布尔类型
        if cpu_arr.dtype == bool:
            assert np.array_equal(cpu_arr, xpu_arr), f"布尔结果在索引 {idx} 处不匹配"
        else:
            # 对于数值类型，使用更宽松的比较条件
            assert np.allclose(
                cpu_arr, xpu_arr, rtol=1e-4, atol=1e-5
            ), f"数值结果在索引 {idx} 处不匹配，最大差异: {np.max(np.abs(cpu_arr - xpu_arr))}"

        print(f"结果 {idx} 验证通过")
