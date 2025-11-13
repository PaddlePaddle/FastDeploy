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
import paddle.nn.functional as F

from fastdeploy.model_executor.ops.gpu import per_token_quant, per_token_quant_padding

paddle.seed(2024)


def per_token_quant_paddle(input_tensor, block_size):
    MAX_VALUE = 448.0
    epsilon = 1e-10

    input_shape = input_tensor.shape
    token_num = input_shape[0]
    hidden_size = input_shape[1]

    # According to https://github.com/PaddlePaddle/FastDeploy/pull/3659
    padding_size = (block_size - hidden_size % block_size) % block_size

    padded_input = input_tensor
    if padding_size > 0:
        padded_input = F.pad(input_tensor, pad=[0, padding_size], mode="constant", value=0.0)

    padded_hidden_size = hidden_size + padding_size
    hidden_size_scale = padded_hidden_size // block_size

    reshaped_input = paddle.reshape(padded_input, [token_num, hidden_size_scale, block_size]).astype("float32")

    max_abs_val = paddle.max(paddle.abs(reshaped_input), axis=-1, keepdim=True)
    max_abs_val = paddle.clip(max_abs_val, min=epsilon)
    scale = max_abs_val / MAX_VALUE

    quanted_value = reshaped_input / scale

    quanted_x_padded_reshaped = quanted_value.to(paddle.float8_e4m3fn)
    quanted_x_padded = paddle.reshape(quanted_x_padded_reshaped, [token_num, padded_hidden_size])

    quanted_x = quanted_x_padded[:, :hidden_size]

    quanted_scale = paddle.squeeze(scale, axis=-1)

    return quanted_x, quanted_scale


def per_token_quant_padding_paddle(input_tensor, block_size, dtype):
    quanted_x, intermediate_scale = per_token_quant_paddle(input_tensor, block_size)
    token_num = input_tensor.shape[0]

    tma_alignment_elements = 4
    padded_token_num = ((token_num + tma_alignment_elements - 1) // tma_alignment_elements) * tma_alignment_elements

    hidden_size_scale = intermediate_scale.shape[1]
    padded_scale = paddle.zeros([padded_token_num, hidden_size_scale], dtype="float32")

    padded_scale[:token_num, :] = intermediate_scale

    return quanted_x, padded_scale


class TestPerTokenQuant(unittest.TestCase):
    def get_input(self, shape, dtype):
        return paddle.randn(shape=shape, dtype=dtype)

    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 4
        self.hidden_size = 500
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)

    def test_per_token_quant(self):
        paddle_output, paddle_output_scale = per_token_quant_paddle(self.input_tensor, self.block_size)
        output, output_scale = per_token_quant(self.input_tensor, self.block_size)

        np.testing.assert_allclose(paddle_output_scale.numpy(), output_scale.numpy(), rtol=1e-6)

        output_rel_diff = paddle.mean(
            paddle.abs(output.to(paddle.float32) - paddle_output.to(paddle.float32))
        ) / paddle.mean(paddle.abs(paddle_output.to(paddle.float32)))

        assert output_rel_diff < 0.001


class TestPerTokenQuantCase1(TestPerTokenQuant):
    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 4
        self.hidden_size = 128 * 6
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantCase2(TestPerTokenQuant):
    def setUp(self) -> None:
        self.dtype = paddle.bfloat16
        self.token_num = 4
        self.hidden_size = 500
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantCase3(TestPerTokenQuant):
    def setUp(self) -> None:
        self.dtype = paddle.bfloat16
        self.token_num = 4
        self.hidden_size = 128 * 6
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantPadding(TestPerTokenQuant):
    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 6
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)

    def test_per_token_quant_padding(self):
        paddle_output, paddle_output_scale = per_token_quant_padding_paddle(
            self.input_tensor, self.block_size, self.dtype
        )
        output, output_scale = per_token_quant_padding(self.input_tensor, self.block_size)

        self.assertEqual(paddle_output_scale.shape, output_scale.shape)
        np.testing.assert_allclose(
            paddle_output_scale[0 : self.token_num].numpy(),
            output_scale[0 : self.token_num].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        output_rel_diff = paddle.mean(
            paddle.abs(output.to(paddle.float32) - paddle_output.to(paddle.float32))
        ) / paddle.mean(paddle.abs(paddle_output.to(paddle.float32)) + 1e-9)

        assert output_rel_diff < 0.001


class TestPerTokenQuantPaddingCase1(TestPerTokenQuantPadding):
    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 8
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantPaddingCase2(TestPerTokenQuantPadding):
    def setUp(self) -> None:
        self.dtype = paddle.bfloat16
        self.token_num = 6
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantPaddingCase3(TestPerTokenQuantPadding):
    def setUp(self) -> None:
        self.dtype = paddle.bfloat16
        self.token_num = 8
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
