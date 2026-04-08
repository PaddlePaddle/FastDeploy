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

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from fastdeploy.model_executor.ops.gpu import group_swiglu_with_masked

paddle.seed(2024)


def group_swiglu_with_masked_paddle(fc1_out_tensor, token_nums_per_expert):
    group_num, group_size, hidden_dim_x2 = fc1_out_tensor.shape

    if token_nums_per_expert.dtype not in [paddle.int32, paddle.int64]:
        raise ValueError(f"token_nums_per_expert must be int32 or int64, but receive {token_nums_per_expert.dtype}")

    gate, up = paddle.chunk(fc1_out_tensor, chunks=2, axis=-1)

    act_out = (F.silu(gate.to(paddle.float32)) * up.to(paddle.float32)).to(fc1_out_tensor.dtype)

    # [0, 1, 2, ..., group_size-1]
    range_tensor = paddle.arange(group_size, dtype=token_nums_per_expert.dtype)

    mask = range_tensor < token_nums_per_expert.unsqueeze(1)

    mask = mask.unsqueeze(-1)

    output_tensor = act_out * mask.astype(act_out.dtype)

    return output_tensor


class TestGroupSwigluWithMasked(unittest.TestCase):
    def get_input(self):
        self.token_nums_tensor = paddle.to_tensor([5, 8, 0, 3], dtype=self.token_nums_per_expert_dtype)
        self.input_tensor = paddle.randn([self.group_num, self.group_size, self.hidden_dim * 2], dtype="bfloat16")

    def setUp(self) -> None:
        self.group_num = 4
        self.group_size = 8
        self.hidden_dim = 16  # fc1_out_tensor.shape()[2] / 2
        self.input_dtype = paddle.bfloat16
        self.token_nums_per_expert_dtype = paddle.int64
        self.get_input()

    def test_group_swiglu_with_masked(self):
        paddle_output = group_swiglu_with_masked_paddle(self.input_tensor, self.token_nums_tensor)
        output = group_swiglu_with_masked(self.input_tensor, self.token_nums_tensor)

        valid_token_mask = paddle.arange(
            self.group_size, dtype=self.token_nums_per_expert_dtype
        ) < self.token_nums_tensor.unsqueeze(1)

        # Note(ooooo): Because GetEmptyTensor will random.
        np.testing.assert_allclose(
            paddle_output[valid_token_mask].astype("float32").numpy(),
            output[valid_token_mask].astype("float32").numpy(),
        )


class TestGroupSwigluWithMaskedCase1(TestGroupSwigluWithMasked):
    def setUp(self) -> None:
        self.group_num = 4
        self.group_size = 8
        self.hidden_dim = 16  # fc1_out_tensor.shape()[2] / 2
        self.input_dtype = paddle.bfloat16
        self.token_nums_per_expert_dtype = paddle.int32
        self.get_input()


class TestGroupSwigluWithMaskedCase2(TestGroupSwigluWithMasked):
    def setUp(self) -> None:
        self.group_num = 4
        self.group_size = 8
        self.hidden_dim = 16  # fc1_out_tensor.shape()[2] / 2
        self.input_dtype = paddle.bfloat16
        self.token_nums_per_expert_dtype = paddle.int32
        self.get_input()

    def get_input(self):
        self.token_nums_tensor = paddle.randint(
            0, self.group_size + 1, shape=[self.group_num], dtype=self.token_nums_per_expert_dtype
        )
        self.input_tensor = paddle.randn(
            [self.group_num, self.group_size, self.hidden_dim * 2], dtype=self.input_dtype
        )


if __name__ == "__main__":
    unittest.main()
