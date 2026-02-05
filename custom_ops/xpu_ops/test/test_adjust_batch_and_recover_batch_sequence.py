# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import (
    adjust_batch,
    get_infer_param,
    recover_batch_sequence,
)


def _run_test_base(seq_lens_this_time_data):
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
    # test adjust_batch
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

    adjusted_output_np = adjusted_output.astype("float32").cpu().numpy()
    adjusted_output_cpu_np = adjusted_output_cpu.astype("float32").cpu().numpy()
    np.testing.assert_allclose(adjusted_output_np, adjusted_output_cpu_np, err_msg="adjust_batch check failed!")

    # test recover_batch_sequence
    recover_out = recover_batch_sequence(
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
    )

    recover_out_cpu = recover_batch_sequence(
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
    )

    recover_out_np = recover_out.astype("float32").cpu().numpy()
    recover_out_cpu_np = recover_out_cpu.astype("float32").cpu().numpy()
    input_np = input_tensor.astype("float32").cpu().numpy()
    np.testing.assert_allclose(recover_out_np, recover_out_cpu_np, err_msg="recover_batch_sequence check failed!")
    np.testing.assert_allclose(recover_out_np, input_np, err_msg="recover_out != input check failed!")


class TestXPUOps(unittest.TestCase):
    """Test the adjust_batch and recover_batch_sequence functions of XPU ops"""

    def test_mix_adjust_recover(self):
        """Test if adjust_batch and recover_batch_sequence can cancel each other out"""
        print("\nRunning test: test_mix_adjust_recover")
        seq_lens_this_time_data = [100, 1, 0, 1, 120, 140, 1]

        _run_test_base(seq_lens_this_time_data)
        print("Test passed for scenario: mix_adjust_recover")


if __name__ == "__main__":
    unittest.main()
