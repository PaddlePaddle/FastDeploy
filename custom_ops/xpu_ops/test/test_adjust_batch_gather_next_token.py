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

import paddle

from fastdeploy.model_executor.ops.xpu import get_infer_param, adjust_batch, gather_next_token
def test_mix_mtp():
    seq_lens_encoder = paddle.to_tensor(  [100, 0, 0, 0,  120, 140, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor(  [0,   5, 0, 25, 64,  0,   128], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([100, 2, 0, 1,  120, 140, 3], dtype="int32")
    # seq_lens_this_time = paddle.to_tensor([100, 1, 0, 1,  120, 140, 1], dtype="int32")
    bsz = seq_lens_this_time.shape[0]
    cum_offsets = paddle.zeros(bsz, dtype="int32")
    block_table = paddle.arange(0, 56, dtype="int32")
    block_table = block_table.reshape((bsz, 8))
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
    ) = get_infer_param(
        seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_table, 64
    )  # block_size
    token_num = seq_lens_this_time.sum().cpu()
    hidden_dim = 8192
    # 生成行索引 [0, 1, 2, ..., m-1]
    row_indices = paddle.arange(token_num, dtype='int32')  # 形状 [m]
    row_indices_bf16 = row_indices.astype("bfloat16")
    # 扩展为 [m, n]：先 unsqueeze 成 [m, 1]，再 expand 成 [m, n]
    input_tensor = paddle.unsqueeze(row_indices_bf16, axis=1).expand(shape=[token_num, hidden_dim])

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
    assert paddle.equal_all(adjusted_output.astype("float32").cpu(), adjusted_output_cpu.astype("float32")).item(), "adjust_batch check failed!"
    output_padding_offset = paddle.zeros(bsz, dtype="int32") # test for mtp
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
    print("adjusted_output: ", adjusted_output)
    print("cum_offsets: ", cum_offsets)
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
    assert paddle.equal_all(gather_out.astype("float32").cpu(), gather_out_cpu.astype("float32")).item(), "adjust_batch check failed!"
    print("test get_infer_param adjust_batch gather_next_token With MTP PASS!")


def test_mix():
    seq_lens_encoder = paddle.to_tensor(  [100, 0, 0, 0,  120, 140, 0], dtype="int32")
    seq_lens_decoder = paddle.to_tensor(  [0,   5, 0, 25, 64,  0,   128], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([100, 1, 0, 1,  120, 140, 1], dtype="int32")
    # seq_lens_this_time = paddle.to_tensor([100, 1, 0, 1,  120, 140, 1], dtype="int32")
    bsz = seq_lens_this_time.shape[0]
    cum_offsets = paddle.zeros(bsz, dtype="int32")
    block_table = paddle.arange(0, 56, dtype="int32")
    block_table = block_table.reshape((bsz, 8))
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
    ) = get_infer_param(
        seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_table, 64
    )  # block_size
    token_num = seq_lens_this_time.sum().cpu()
    hidden_dim = 8192
    # 生成行索引 [0, 1, 2, ..., m-1]
    row_indices = paddle.arange(token_num, dtype='int32')  # 形状 [m]
    row_indices_bf16 = row_indices.astype("bfloat16")
    # 扩展为 [m, n]：先 unsqueeze 成 [m, 1]，再 expand 成 [m, n]
    input_tensor = paddle.unsqueeze(row_indices_bf16, axis=1).expand(shape=[token_num, hidden_dim])

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
    assert paddle.equal_all(adjusted_output.astype("float32").cpu(), adjusted_output_cpu.astype("float32")).item(), "adjust_batch check failed!"
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
                None,
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
                None,
                -1,
            )
    for i in range(gather_out_cpu.shape[0]):
        if seq_lens_this_time[i] > 0:
            assert paddle.equal_all(gather_out[i].astype("float32").cpu(), gather_out_cpu[i].astype("float32")).item(), "adjust_batch check failed!"
    print("test get_infer_param adjust_batch gather_next_token WithOut MTP PASS!")


test_mix()
test_mix_mtp()

