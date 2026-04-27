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

import unittest  # 导入 unittest

import numpy as np
import paddle
from utils import init_inplace_tensor

from fastdeploy.model_executor.ops.xpu import (
    adjust_batch,
    gather_next_token,
    get_infer_param,
)


def _run_test_base(seq_lens_this_time_data, is_speculative):
    """
    通用的基础测试执行函数，包含了两个场景共有的逻辑。
    """
    seq_lens_encoder = paddle.to_tensor([100, 0, 0, 0, 120, 140, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor([0, 5, 0, 25, 64, 0, 128], dtype="int32")
    seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_data, dtype="int32")

    bsz = seq_lens_this_time.shape[0]
    block_table = paddle.arange(0, 56, dtype="int32").reshape((bsz, 8))

    (
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
    ) = init_inplace_tensor(seq_lens_encoder.shape[0], block_table.shape)
    (
        slot_mapping_enc,
        slot_mapping_dec,
    ) = get_infer_param(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        block_table,
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
        64,
        0,
    )

    token_num = seq_lens_this_time.sum().cpu().item()
    hidden_dim = 8192
    row_indices = paddle.arange(token_num, dtype="int32")
    row_indices_bf16 = row_indices.astype("bfloat16")
    input_tensor = paddle.unsqueeze(row_indices_bf16, axis=1).expand(shape=[token_num, hidden_dim])

    # 测试 adjust_batch
    adjusted_output = adjust_batch(
        input_tensor,
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

    # 用 np.testing 替代原生 assert，错误信息更友好
    adjusted_output_np = adjusted_output.astype("float32").cpu().numpy()
    adjusted_output_cpu_np = adjusted_output_cpu.astype("float32").cpu().numpy()
    np.testing.assert_allclose(adjusted_output_np, adjusted_output_cpu_np, err_msg="adjust_batch check failed!")

    # 测试 gather_next_token
    gather_out = gather_next_token(
        adjusted_output,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_map,
        decoder_batch_map,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        len_info_cpu,
        is_speculative,
        -1,
    )

    gather_out_cpu = gather_next_token(
        adjusted_output.cpu(),
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_map,
        decoder_batch_map,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        len_info_cpu,
        is_speculative,
        -1,
    )

    gather_out_np = gather_out.astype("float32").cpu().numpy()
    gather_out_cpu_np = gather_out_cpu.astype("float32").cpu().numpy()

    if is_speculative:
        np.testing.assert_allclose(gather_out_np, gather_out_cpu_np, err_msg="gather_next_token check failed!")
    else:
        for i in range(gather_out_cpu.shape[0]):
            if seq_lens_this_time[i] > 0:
                np.testing.assert_allclose(
                    gather_out_np[i], gather_out_cpu_np[i], err_msg=f"gather_next_token check failed at index {i}!"
                )


class TestXPUOps(unittest.TestCase):  # 继承 unittest.TestCase
    """测试 XPU ops 的 adjust_batch 和 gather_next_token 功能"""

    def test_mix_with_mtp(self):
        """测试混合批次处理中的 MTP (Multi-Token Prediction) 场景"""
        print("\nRunning test: test_mix_with_mtp")
        seq_lens_this_time_data = [100, 2, 0, 1, 120, 140, 3]
        _run_test_base(seq_lens_this_time_data, True)
        print("Test passed for scenario: With MTP")

    def test_mix_without_mtp(self):
        """测试非 MTP (Single-Token Prediction) 场景下的功能"""
        print("\nRunning test: test_mix_without_mtp")
        seq_lens_this_time_data = [100, 1, 0, 1, 120, 140, 1]
        _run_test_base(seq_lens_this_time_data, False)
        print("Test passed for scenario: Without MTP")


if __name__ == "__main__":
    unittest.main()  # 使用 unittest 运行测试
