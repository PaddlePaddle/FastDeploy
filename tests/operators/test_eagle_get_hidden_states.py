# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

from fastdeploy.model_executor.ops.gpu import eagle_get_hidden_states


def ComputeOrderKernel(
    seq_lens_this_time,
    seq_lens_encoder,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    accept_nums,
    position_map,
    output_token_num,
    bsz,
    actual_draft_token_num,
    input_token_num,
):
    in_offset = 0
    out_offset = 0
    for i in range(bsz):
        cur_base_model_seq_lens_this_time = base_model_seq_lens_this_time[i]
        # cur_base_model_seq_lens_encoder = base_model_seq_lens_encoder[i]
        cur_seq_lens_this_time = seq_lens_this_time[i]
        accept_num = accept_nums[i]
        cur_seq_lens_encoder = seq_lens_encoder[i]
        # 1. eagle encoder. Base step=1
        if cur_seq_lens_encoder > 0:
            for j in range(cur_seq_lens_encoder):
                position_map[in_offset] = out_offset
                in_offset += 1
                out_offset += 1
        # 2. Base model stop at last verify-step.
        elif cur_base_model_seq_lens_this_time != 0 and cur_seq_lens_this_time == 0:
            in_offset += cur_base_model_seq_lens_this_time
        # 4. stopped
        elif cur_base_model_seq_lens_this_time == 0 and cur_seq_lens_this_time == 0:  # end
            pass
        else:
            for i in range(accept_num):
                position_map[in_offset] = out_offset
                in_offset += 1
                out_offset += 1
            in_offset += cur_base_model_seq_lens_this_time - accept_num
        output_token_num[0] = out_offset


def rebuildHiddenStatesKernel(input, position_map, out, dim_embed, elem_cnt):
    for elem_idx in range(elem_cnt):
        ori_token_idx = int(elem_idx / dim_embed)
        token_idx = position_map[ori_token_idx]
        if token_idx >= 0:
            offset = elem_idx % dim_embed
            out[token_idx][offset] = input[ori_token_idx][offset]


def eagle_get_hidden_states_ref(
    input,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    stop_flags,
    accept_nums,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    actual_draft_token_num,
):
    input_token_num = input.shape[0]
    dim_embed = input.shape[1]
    bsz = seq_lens_this_time.shape[0]
    position_map = paddle.full([input_token_num], 0xFFFFFFFF, seq_lens_this_time.dtype)
    output_token_num = paddle.empty([1], seq_lens_this_time.dtype)
    ComputeOrderKernel(
        seq_lens_this_time,
        seq_lens_encoder,
        base_model_seq_lens_this_time,
        base_model_seq_lens_encoder,
        accept_nums,
        position_map,
        output_token_num,
        bsz,
        actual_draft_token_num,
        input_token_num,
    )

    output_token_num_cpu = output_token_num[0]
    out = paddle.empty([output_token_num_cpu, dim_embed], input.dtype)
    elem_cnt = input_token_num * dim_embed
    rebuildHiddenStatesKernel(input, position_map, out, dim_embed, elem_cnt)
    return out


class TestEagleGetHiddenStates(unittest.TestCase):
    def test_eagle_get_hidden_states(self):
        np.random.seed(2023)
        paddle.seed(2023)
        bs = 2
        input_token_num = 10
        dim_embed = 512
        actual_draft_token_num = np.random.randint(2, 6, dtype=np.int32)

        seq_lens_this_time = np.random.randint(0, 2, bs, dtype=np.int32)
        seq_lens_encoder = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
        accept_nums = np.random.randint(0, actual_draft_token_num + 1, bs, dtype=np.int32)
        base_model_seq_lens_this_time = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
        base_model_seq_lens_encoder = np.random.randint(0, 2, bs, dtype=np.int32)

        seq_lens_decoder = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
        stop_flags = np.random.randint(0, 2, bs, dtype=np.int32)

        seq_lens_this_time_tensor = paddle.to_tensor(seq_lens_this_time, dtype=paddle.int32)
        seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder, dtype=paddle.int32)
        accept_nums_tensor = paddle.to_tensor(accept_nums, dtype=paddle.int32)
        base_model_seq_lens_this_time_tensor = paddle.to_tensor(base_model_seq_lens_this_time, dtype=paddle.int32)
        base_model_seq_lens_encoder_tensor = paddle.to_tensor(base_model_seq_lens_encoder, dtype=paddle.int32)

        seq_lens_decoder_tensor = paddle.to_tensor(seq_lens_decoder, dtype=paddle.int32)
        stop_flags_tensor = paddle.to_tensor(stop_flags, dtype=paddle.int32)

        input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int32)
        input_tensor = paddle.to_tensor(input, dtype=paddle.float16)
        out = eagle_get_hidden_states(
            input_tensor,
            seq_lens_this_time_tensor,
            seq_lens_encoder_tensor,
            seq_lens_decoder_tensor,
            stop_flags_tensor,
            accept_nums_tensor,
            base_model_seq_lens_this_time_tensor,
            base_model_seq_lens_encoder_tensor,
            actual_draft_token_num,
        )
        out_ref = eagle_get_hidden_states_ref(
            input_tensor,
            seq_lens_this_time_tensor,
            seq_lens_encoder_tensor,
            seq_lens_decoder_tensor,
            stop_flags_tensor,
            accept_nums_tensor,
            base_model_seq_lens_this_time_tensor,
            base_model_seq_lens_encoder_tensor,
            actual_draft_token_num,
        )
        np.testing.assert_allclose(out.numpy(), out_ref.numpy())


if __name__ == "__main__":
    unittest.main()
