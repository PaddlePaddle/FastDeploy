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
import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import draft_model_postprocess


def draft_model_postprocess_cpu(
    base_model_draft_tokens,  # 2D列表: [bsz, base_model_draft_token_len]  # 1D列表: [bsz]
    base_model_seq_lens_encoder,  # 1D列表: [bsz]
    base_model_stop_flags,  # 1D列表: [bsz]
):
    bsz = base_model_draft_tokens.shape[0]
    base_model_draft_token_len = base_model_draft_tokens.shape[1]
    base_model_seq_lens_this_time = paddle.ones((bsz), dtype=paddle.int32)
    # 遍历每个样本
    for tid in range(bsz):
        if (not base_model_stop_flags[tid]) and (base_model_seq_lens_encoder[tid] == 0):
            # 获取当前样本的草稿token列表
            base_model_draft_tokens_now = base_model_draft_tokens[tid]
            token_num = 0
            for i in range(base_model_draft_token_len):
                if base_model_draft_tokens_now[i] != -1:
                    token_num += 1

            # 更新序列长度
            base_model_seq_lens_this_time[tid] = token_num
        elif base_model_stop_flags[tid]:
            # 已停止的样本序列长度为0
            base_model_seq_lens_this_time[tid] = 0

    return [base_model_seq_lens_this_time]


def test_draft_model_postprocess(batch_size=1, base_model_draft_token_len=8192):  # 批次大小
    paddle.seed(66)
    base_model_draft_tokens = paddle.randint(
        low=-1,
        high=1,
        shape=[batch_size, base_model_draft_token_len],
        dtype="int64",
    )
    # base_model_seq_lens_this_time = paddle.ones((batch_size), dtype=paddle.int32)
    base_model_seq_lens_encoder = paddle.randint(low=0, high=2, shape=[batch_size], dtype="int32")
    random_floats = paddle.rand(shape=[batch_size])
    base_model_stop_flags = random_floats >= 0.5

    base_model_seq_lens_this_time = draft_model_postprocess_cpu(
        base_model_draft_tokens,  # 2D列表: [bsz, base_model_draft_token_len]
        base_model_seq_lens_encoder,  # 1D列表: [bsz]
        base_model_stop_flags,
    )
    base_model_seq_lens_this_time_xpu = paddle.ones((batch_size), dtype=paddle.int32)
    draft_model_postprocess(
        base_model_draft_tokens,  # 2D列表: [bsz, base_model_draft_token_len]
        base_model_seq_lens_this_time_xpu,  # 1D列表: [bsz]
        base_model_seq_lens_encoder,  # 1D列表: [bsz]
        base_model_stop_flags,
    )
    print("test start")
    assert np.allclose(base_model_seq_lens_this_time, base_model_seq_lens_this_time_xpu)
    print("test passed")


def test_enough_cases():
    test_draft_model_postprocess(100, 1024)
    test_draft_model_postprocess(1, 11)
    test_draft_model_postprocess(1, 8192)
    test_draft_model_postprocess(2, 2048)
    test_draft_model_postprocess(3, 1023)
    test_draft_model_postprocess(4, 2047)
    test_draft_model_postprocess(5, 4095)
    test_draft_model_postprocess(10, 9191)
    test_draft_model_postprocess(20, 618)
    test_draft_model_postprocess(30, 703)
    test_draft_model_postprocess(100, 1025)
    test_draft_model_postprocess(1536, 1026)


if __name__ == "__main__":
    test_enough_cases()
