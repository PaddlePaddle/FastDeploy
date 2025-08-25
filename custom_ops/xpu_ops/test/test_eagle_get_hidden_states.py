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

from fastdeploy.model_executor.ops.xpu import eagle_get_hidden_states


def test_eagle_get_hidden_states():
    bs = np.random.randint(1, 8 + 1, dtype=np.int32)
    input_token_num = np.random.randint(2 * 1024, 4 * 1024 + 1, dtype=np.int32)
    dim_embed = np.random.randint(1, 4 * 1024 + 1, dtype=np.int32)
    actual_draft_token_num = np.random.randint(2, 6, dtype=np.int32)

    seq_lens_this_time = np.random.randint(0, 2, bs, dtype=np.int32)
    seq_lens_encoder = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
    accept_nums = np.random.randint(0, actual_draft_token_num + 1, bs, dtype=np.int32)
    base_model_seq_lens_this_time = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
    base_model_seq_lens_encoder = np.random.randint(0, 2, bs, dtype=np.int32)
    # dont care
    seq_lens_decoder = np.random.randint(0, input_token_num // bs + 1, bs, dtype=np.int32)
    stop_flags = np.random.randint(0, 2, bs, dtype=np.int32)

    seq_lens_this_time_tensor = paddle.to_tensor(seq_lens_this_time, dtype=paddle.int32)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder, dtype=paddle.int32)
    accept_nums_tensor = paddle.to_tensor(accept_nums, dtype=paddle.int32)
    base_model_seq_lens_this_time_tensor = paddle.to_tensor(base_model_seq_lens_this_time, dtype=paddle.int32)
    base_model_seq_lens_encoder_tensor = paddle.to_tensor(base_model_seq_lens_encoder, dtype=paddle.int32)
    # dont care
    seq_lens_decoder_tensor = paddle.to_tensor(seq_lens_decoder, dtype=paddle.int32)
    stop_flags_tensor = paddle.to_tensor(stop_flags, dtype=paddle.int32)

    # fp32 test
    input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int32)
    input_tensor = paddle.to_tensor(input, dtype=paddle.float32)
    cpu_out = eagle_get_hidden_states(
        input_tensor.cpu(),
        seq_lens_this_time_tensor.cpu(),
        seq_lens_encoder_tensor.cpu(),
        seq_lens_decoder_tensor.cpu(),
        stop_flags_tensor.cpu(),
        accept_nums_tensor.cpu(),
        base_model_seq_lens_this_time_tensor.cpu(),
        base_model_seq_lens_encoder_tensor.cpu(),
        actual_draft_token_num,
    )
    xpu_out = eagle_get_hidden_states(
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
    assert np.allclose(cpu_out.numpy(), xpu_out.numpy())

    # bf16/fp16 test
    for dtype in [paddle.bfloat16, paddle.float16]:
        input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int16)
        input_tensor = paddle.to_tensor(input, dtype=dtype)
        cpu_out = eagle_get_hidden_states(
            input_tensor.cpu(),
            seq_lens_this_time_tensor.cpu(),
            seq_lens_encoder_tensor.cpu(),
            seq_lens_decoder_tensor.cpu(),
            stop_flags_tensor.cpu(),
            accept_nums_tensor.cpu(),
            base_model_seq_lens_this_time_tensor.cpu(),
            base_model_seq_lens_encoder_tensor.cpu(),
            actual_draft_token_num,
        )
        xpu_out = eagle_get_hidden_states(
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
        assert np.allclose(cpu_out.numpy(), xpu_out.numpy())

    print("All test passed")


if __name__ == "__main__":
    test_eagle_get_hidden_states()
