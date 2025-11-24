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
import pytest

from fastdeploy.model_executor.ops.xpu import (
    adjust_batch,
    gather_next_token,
    get_infer_param,
)


def _run_test_base(seq_lens_this_time_data, output_padding_offset):
    """
    通用的基础测试执行函数，包含了两个场景共有的逻辑。
    """
    seq_lens_encoder = paddle.to_tensor([100, 0, 0, 0, 120, 140, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor([0, 5, 0, 25, 64, 0, 128], dtype="int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_data, dtype="int32")

    bsz = seq_lens_this_time.shape[0]
    cum_offsets = paddle.zeros(bsz, dtype="int32")
    block_table = paddle.arange(0, 56, dtype="int32").reshape((bsz, 8))

    infer_params = get_infer_param(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_table, 64)

    (
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        _,
        _,
        _,
        _,
        _,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        _,
        _,
        _,
        _,
        len_info_cpu,
    ) = infer_params

    token_num = seq_lens_this_time.sum().cpu().item()
    hidden_dim = 8192
    row_indices = paddle.arange(token_num, dtype="int32")
    row_indices_bf16 = row_indices.astype("bfloat16")
    input_tensor = paddle.unsqueeze(row_indices_bf16, axis=1).expand(shape=[token_num, hidden_dim])

    # 测试 adjust_batch
    adjusted_output = adjust_batch(
        input_tensor,
        cum_offsets,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        len_info_cpu,
        None,  # output_padding_offset
        -1,  # max_input_length
    )

    adjusted_output_cpu = adjust_batch(
        input_tensor.cpu(),
        cum_offsets,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        len_info_cpu,
        None,  # output_padding_offset
        -1,  # max_input_length
    )
    assert (
        paddle.equal_all(adjusted_output.astype("float32").cpu(), adjusted_output_cpu.astype("float32")).all().item()
    ), "adjust_batch check failed!"

    # 测试 gather_next_token
    gather_out = gather_next_token(
        adjusted_output,
        cum_offsets,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_map,
        decoder_batch_map,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        len_info_cpu,
        output_padding_offset,
        -1,
    )

    gather_out_cpu = gather_next_token(
        adjusted_output.cpu(),
        cum_offsets,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_map,
        decoder_batch_map,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        len_info_cpu,
        output_padding_offset,
        -1,
    )

    if output_padding_offset is not None:
        np.testing.assert_allclose(
            gather_out.astype("float32").cpu().numpy(),
            gather_out_cpu.astype("float32").cpu().numpy(),
            err_msg="gather_next_token check failed!",
        )
    else:
        for i in range(gather_out_cpu.shape[0]):
            if seq_lens_this_time[i] > 0:
                np.testing.assert_allclose(
                    gather_out[i].astype("float32").cpu().numpy(),
                    gather_out_cpu[i].astype("float32").cpu().numpy(),
                    err_msg="gather_next_token check failed!",
                )


def test_mix_with_mtp():
    """测试混合批次处理中的MTP(Multi-Token Prediction)场景。

    验证在不同序列长度(包括零长度)情况下，MTP功能是否能正确处理。

    Args:
        无显式参数，但内部使用:
        seq_lens_this_time_data: 包含不同长度序列的列表，用于模拟混合批次
        output_padding_offset: 用于处理序列填充的偏移量张量

    Returns:
        无返回值，但会打印测试结果
    """
    print("\nRunning test: test_mix_with_mtp")
    seq_lens_this_time_data = [100, 2, 0, 1, 120, 140, 3]
    bsz = len(seq_lens_this_time_data)
    output_padding_offset = paddle.zeros(bsz, dtype="int32")

    _run_test_base(seq_lens_this_time_data, output_padding_offset)
    print("Test passed for scenario: With MTP")


def test_mix_without_mtp():
    """测试非MTP(Single-Token Prediction)场景下的功能。

    该测试用例专门验证在非MTP(多令牌预测)场景下，模型处理不同长度序列的能力。

    Args:
        seq_lens_this_time_data: 本次处理的序列长度列表，包含各种长度的序列
        output_padding_offset: 非MTP场景下此参数应为None
    """
    print("\nRunning test: test_mix_without_mtp")
    seq_lens_this_time_data = [100, 1, 0, 1, 120, 140, 1]
    output_padding_offset = None  # 非MTP场景下，此参数为None

    _run_test_base(seq_lens_this_time_data, output_padding_offset)
    print("Test passed for scenario: Without MTP")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
