"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import rebuild_padding


def RebuildPaddingKernel(
    out,
    tmp_out,
    cu_seqlens_q,
    seq_len_this_time,
    seq_lens_decoder,
    seq_lens_encoder,
    bsz,
):
    for bi in range(bsz):
        seq_id = 0
        if seq_len_this_time[bi] == 0:
            continue
        if seq_lens_decoder[bi] == 0 and seq_lens_encoder[bi] == 0:
            continue
        if seq_lens_encoder[bi] > 0:
            seq_id = seq_lens_encoder[bi] - 1
        out[bi] = tmp_out[cu_seqlens_q[bi] + seq_id][:]


def RebuildAppendPaddingKernel(
    out,
    tmp_out,
    cu_seqlens_q,
    seq_len_this_time,
    seq_lens_decoder,
    seq_lens_encoder,
    output_padding_offset,
    max_input_length,
    token_num,
    need_delete_token_num,
):
    for token_id in range(token_num - need_delete_token_num):
        bi = int(token_id / max_input_length)
        if seq_len_this_time[bi] == 0 or (seq_lens_decoder[bi] == 0 and seq_lens_encoder[bi] == 0):
            continue
        ori_token_id = token_id + output_padding_offset[token_id]
        seq_id = 0
        if seq_lens_encoder[bi] > 0:
            seq_id = seq_lens_encoder[bi] - 1
        cum_offset_bi = bi * max_input_length - cu_seqlens_q[bi]
        input_token_id = ori_token_id - cum_offset_bi + seq_id
        out[token_id] = tmp_out[input_token_id][:]


def rebuild_padding_ref(
    tmp_out,  # [token_num, dim_embed]
    cu_seqlens_q,  # [bsz+1, 1]
    seq_len_this_time,
    seq_lens_decoder,
    seq_lens_encoder,
    output_padding_offset,
    max_input_length,
):

    tmp_out_shape = tmp_out.shape
    token_num = tmp_out_shape[0]
    dim_embed = tmp_out_shape[1]
    bsz = cu_seqlens_q.shape[0] - 1

    out = np.zeros([bsz, dim_embed])
    if output_padding_offset is not None:
        need_delete_token_num = 0
        for i in range(bsz):
            if seq_lens_encoder[i] > 0:
                need_delete_token_num += seq_lens_encoder[i] - 1
        out = np.zeros([token_num - need_delete_token_num, dim_embed])
    else:
        out = np.zeros([bsz, dim_embed])

    if output_padding_offset is not None:
        RebuildAppendPaddingKernel(
            out,
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            max_input_length,
            token_num,
            need_delete_token_num,
        )
    else:
        RebuildPaddingKernel(
            out,
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            bsz,
        )
    return out


class TestRebuildPadding(unittest.TestCase):
    # test no offset
    def test_rebuild_padding_no_offset(self):
        token_num = 100
        dim_embed = 256
        # bsz = 4
        max_input_length = 512
        # tmp_out: [token_num, dim_embed]
        tmp_out = np.random.randn(token_num, dim_embed).astype(np.float32)
        # cu_seqlens_q: [bsz + 1]，accumulate the number of tokens for each batch.
        cu_seqlens_q = np.array(
            [0, 1, 21, 22, 42, 43, 63, 64, 84], dtype=np.int32
        )  # Assume there are 4 batches, and the total token_num = 100.

        # Simulated sequence length information
        seq_len_this_time = np.array([1, 20, 1, 20, 1, 20, 1, 20], dtype=np.int32)
        seq_lens_encoder = np.array([0, 20, 0, 20, 0, 20, 0, 20], dtype=np.int32)
        seq_lens_decoder = np.array([21, 0, 21, 0, 21, 0, 21, 0], dtype=np.int32)
        out_no_offset_ref = rebuild_padding_ref(
            tmp_out=tmp_out,
            cu_seqlens_q=cu_seqlens_q,
            seq_len_this_time=seq_len_this_time,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            output_padding_offset=None,
            max_input_length=max_input_length,
        )

        tmp_out = paddle.to_tensor(tmp_out)
        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q)
        seq_len_this_time = paddle.to_tensor(seq_len_this_time)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)

        out_no_offset = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            None,
            None,
            max_input_length,
            False,
        )
        np.testing.assert_allclose(out_no_offset.numpy(), out_no_offset_ref)

    # test with offset
    def test_rebuild_padding_with_offset(self):
        paddle.seed(42)
        token_num = 100
        dim_embed = 256
        # bsz = 4
        max_input_length = 512
        # tmp_out: [token_num, dim_embed]
        tmp_out = np.random.randn(token_num, dim_embed).astype(np.float32)
        # cu_seqlens_q: [bsz + 1]，accumulate the number of tokens for each batch.
        cu_seqlens_q = np.array(
            [0, 1, 21, 22, 42, 43, 63, 64, 84], dtype=np.int32
        )  # Assume there are 4 batches, and the total token_num = 100.

        # Simulated sequence length information
        seq_len_this_time = np.array([1, 20, 1, 20, 1, 20, 1, 20], dtype=np.int32)
        seq_lens_encoder = np.array([0, 20, 0, 20, 0, 20, 0, 20], dtype=np.int32)
        seq_lens_decoder = np.array([21, 0, 21, 0, 21, 0, 21, 0], dtype=np.int32)

        num_output_tokens = 80
        output_padding_offset = np.random.randint(0, 10, [num_output_tokens], dtype=np.int32)
        out_with_offset_ref = rebuild_padding_ref(
            tmp_out=tmp_out,
            cu_seqlens_q=cu_seqlens_q,
            seq_len_this_time=seq_len_this_time,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            output_padding_offset=output_padding_offset,
            max_input_length=max_input_length,
        )

        tmp_out = paddle.to_tensor(tmp_out)
        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q)
        seq_len_this_time = paddle.to_tensor(seq_len_this_time)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)
        output_padding_offset = paddle.to_tensor(output_padding_offset)
        out_with_offset = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            None,
            max_input_length,
            False,
        )
        np.testing.assert_allclose(out_with_offset.numpy(), out_with_offset_ref)


if __name__ == "__main__":
    unittest.main()
