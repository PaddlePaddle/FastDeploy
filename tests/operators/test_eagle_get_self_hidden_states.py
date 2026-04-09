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

from fastdeploy.model_executor.ops.gpu import eagle_get_self_hidden_states


def computeOrderKernel(
    last_seq_lens_this_time, seq_lens_this_time, seq_lens_encoder, position_map, output_token_num, bsz
):
    in_offset = 0
    out_offset = 0
    for i in range(bsz):
        cur_seq_lens_this_time = seq_lens_this_time[i]
        cur_last_seq_lens_this_time = last_seq_lens_this_time[i]
        # 1. encoder
        if seq_lens_encoder[i] > 0 and cur_seq_lens_this_time > 0:
            position_map[in_offset] = out_offset
            in_offset += 1
            out_offset += 1
        # 2. decoder
        elif cur_seq_lens_this_time > 0:  # =1
            position_map[in_offset + cur_last_seq_lens_this_time - 1] = out_offset
            in_offset += cur_last_seq_lens_this_time
            out_offset += 1
        # 3. stop
        else:
            # first token end
            if seq_lens_encoder[i] > 0:
                in_offset += 1 if cur_last_seq_lens_this_time > 0 else 0
            # normal end
            else:
                in_offset += cur_last_seq_lens_this_time
    output_token_num[0] = out_offset


def rebuildSelfHiddenStatesKernel(input, position_map, out, dim_embed, elem_cnt):
    for elem_idx in range(elem_cnt):
        ori_token_idx = int(elem_idx / dim_embed)
        token_idx = position_map[ori_token_idx]
        if token_idx >= 0:
            offset = elem_idx % dim_embed
            out[token_idx][offset] = input[ori_token_idx][offset]


def eagle_get_self_hidden_states_ref(input, last_seq_lens_this_time, seq_lens_this_time, seq_lens_encoder):
    input_token_num = input.shape[0]
    dim_embed = input.shape[1]
    bsz = seq_lens_this_time.shape[0]
    position_map = paddle.full([input_token_num], -1, seq_lens_this_time.dtype)
    output_token_num = paddle.full([1], 0, seq_lens_this_time.dtype)

    computeOrderKernel(
        last_seq_lens_this_time, seq_lens_this_time, seq_lens_encoder, position_map, output_token_num, bsz
    )

    out = paddle.empty([input_token_num, dim_embed], input.dtype)

    elem_cnt = input_token_num * dim_embed
    rebuildSelfHiddenStatesKernel(input, position_map, out, dim_embed, elem_cnt)

    return out, output_token_num


class TestEagleGetSelfHiddenStates(unittest.TestCase):
    def test_eagle_get_self_hidden_states(self):
        paddle.seed(2023)
        np.random.seed(2023)
        bs = 2
        input_token_num = 10
        dim_embed = 512

        last_seq_lens_this_time = np.random.randint(0, input_token_num // bs, bs, dtype=np.int32)
        seq_lens_this_time = np.random.randint(0, input_token_num // bs, bs, dtype=np.int32)
        # seq_lens_encoder > 0 means encoder phase, 0 means decoder phase
        seq_lens_encoder = np.array([5, 0], dtype=np.int32)  # bs=2, first is encoder, second is decoder

        last_seq_lens_this_time_tensor = paddle.to_tensor(last_seq_lens_this_time, dtype=paddle.int32)
        seq_lens_this_time_tensor = paddle.to_tensor(seq_lens_this_time, dtype=paddle.int32)
        seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder, dtype=paddle.int32)

        input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int32)
        input_tensor = paddle.to_tensor(input, dtype=paddle.float16)
        out, output_token_num = eagle_get_self_hidden_states(
            input_tensor,
            last_seq_lens_this_time_tensor,
            seq_lens_this_time_tensor,
            seq_lens_encoder_tensor,
        )
        out_ref, output_token_num_ref = eagle_get_self_hidden_states_ref(
            input_tensor,
            last_seq_lens_this_time_tensor,
            seq_lens_this_time_tensor,
            seq_lens_encoder_tensor,
        )
        actual_num = int(output_token_num)
        actual_num_ref = int(output_token_num_ref)
        assert actual_num == actual_num_ref, f"output_token_num mismatch: {actual_num} vs {actual_num_ref}"
        np.testing.assert_allclose(out[:actual_num].numpy(), out_ref[:actual_num].numpy())


if __name__ == "__main__":
    unittest.main()
