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

from fastdeploy.model_executor.ops.xpu import speculate_rebuild_append_padding


def ref_speculate_rebuild_append_padding(
    full_hidden_states,
    cum_offsets,
    seq_len_encoder,
    seq_len_decoder,
    output_padding_offset,
    max_seq_len,
):
    dim_embed = full_hidden_states.shape[1]
    output_token_num = output_padding_offset.shape[0]
    elem_nums = output_token_num * dim_embed

    out = np.zeros(output_token_num * dim_embed, dtype=full_hidden_states.dtype)
    full_hidden_states_flatten = full_hidden_states.flatten()
    cum_offsets_flatten = cum_offsets.flatten()
    seq_len_encoder_flatten = seq_len_encoder.flatten()
    seq_len_decoder_flatten = seq_len_decoder.flatten()
    output_padding_offset_flatten = output_padding_offset.flatten()

    for i in range(elem_nums):
        out_token_id = i // dim_embed
        ori_token_id = out_token_id + output_padding_offset_flatten[out_token_id]
        bi = ori_token_id // max_seq_len

        seq_id = 0
        if seq_len_decoder_flatten[bi] == 0 and seq_len_encoder_flatten[bi] == 0:
            continue
        elif seq_len_encoder_flatten[bi] != 0:
            seq_id = seq_len_encoder[bi] - 1

        input_token_id = ori_token_id - cum_offsets_flatten[bi] + seq_id
        bias_idx = i % dim_embed

        out[i] = full_hidden_states_flatten[input_token_id * dim_embed + bias_idx]
    out = np.reshape(out, (output_token_num, dim_embed))
    return out


def test_speculate_rebuild_append_padding():
    bs = np.random.randint(1, 4 + 1, dtype=np.int32)
    max_seq_len = 1 * 1024
    dim_embed = np.random.randint(1, 4 * 1024 + 1, dtype=np.int32)
    seq_lens = []
    for _ in range(bs):
        seq_lens.append(np.random.randint(1, max_seq_len + 1, dtype=np.int32))
    seq_lens = np.asarray(seq_lens)
    cum_offsets = np.cumsum(np.asarray(max_seq_len) - seq_lens)
    cum_offsets = np.insert(cum_offsets, 0, 0)
    output_padding_offsets = []
    for i in range(bs):
        offset = cum_offsets[i]
        for j in range(seq_lens[i]):
            output_padding_offsets.append(offset)
    output_padding_offsets = np.asarray(output_padding_offsets)
    # TODO: seq_len_encoder with non-zero element
    seq_len_decoder = np.random.randint(0, 2 + 1, bs, dtype=np.int32)
    seq_len_encoder_zeros = np.zeros(bs, dtype=np.int32)

    for dtype in [paddle.bfloat16, paddle.float16]:
        full_hidden_states = np.random.randint(0, 10, (np.sum(seq_lens), dim_embed), dtype=np.int16)
        full_hidden_states_tensor = paddle.to_tensor(full_hidden_states, dtype=dtype)
        cum_offsets_tensor = paddle.to_tensor(cum_offsets, dtype=paddle.int32)
        seq_len_encoder_zeros_tensor = paddle.to_tensor(seq_len_encoder_zeros, dtype=paddle.int32)
        seq_len_decoder_tensor = paddle.to_tensor(seq_len_decoder, dtype=paddle.int32)
        output_padding_offsets_tensor = paddle.to_tensor(output_padding_offsets, dtype=paddle.int32)
        cpu_out = speculate_rebuild_append_padding(
            full_hidden_states_tensor.cpu(),
            cum_offsets_tensor.cpu(),
            seq_len_encoder_zeros_tensor.cpu(),
            seq_len_decoder_tensor.cpu(),
            output_padding_offsets_tensor.cpu(),
            max_seq_len,
        )
        xpu_out = speculate_rebuild_append_padding(
            full_hidden_states_tensor,
            cum_offsets_tensor,
            seq_len_encoder_zeros_tensor,
            seq_len_decoder_tensor,
            output_padding_offsets_tensor,
            max_seq_len,
        )
        assert np.allclose(cpu_out.numpy(), xpu_out.numpy())
    for dtype in [paddle.float32]:
        full_hidden_states = np.random.randint(0, 10, (np.sum(seq_lens), dim_embed), dtype=np.int32)
        full_hidden_states_tensor = paddle.to_tensor(full_hidden_states, dtype=dtype)
        cum_offsets_tensor = paddle.to_tensor(cum_offsets, dtype=paddle.int32)
        seq_len_encoder_zeros_tensor = paddle.to_tensor(seq_len_encoder_zeros, dtype=paddle.int32)
        seq_len_decoder_tensor = paddle.to_tensor(seq_len_decoder, dtype=paddle.int32)
        output_padding_offsets_tensor = paddle.to_tensor(output_padding_offsets, dtype=paddle.int32)
        cpu_out = speculate_rebuild_append_padding(
            full_hidden_states_tensor.cpu(),
            cum_offsets_tensor.cpu(),
            seq_len_encoder_zeros_tensor.cpu(),
            seq_len_decoder_tensor.cpu(),
            output_padding_offsets_tensor.cpu(),
            max_seq_len,
        )
        xpu_out = speculate_rebuild_append_padding(
            full_hidden_states_tensor,
            cum_offsets_tensor,
            seq_len_encoder_zeros_tensor,
            seq_len_decoder_tensor,
            output_padding_offsets_tensor,
            max_seq_len,
        )
        assert np.allclose(cpu_out.numpy(), xpu_out.numpy())

    print("All test passed")


if __name__ == "__main__":
    test_speculate_rebuild_append_padding()
