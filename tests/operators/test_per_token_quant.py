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

from fastdeploy.model_executor.ops.gpu import per_token_quant_padding

paddle.seed(2024)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Align x to TMA-required size.

    Args:
        x: size in elements
        element_size: size of each element in bytes

    Returns:
        Aligned size in elements
    """
    kNumTMAAlignmentBytes = 16
    assert kNumTMAAlignmentBytes % element_size == 0
    return align(x, kNumTMAAlignmentBytes // element_size)


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    assert x.dtype == paddle.float and x.dim() in (2, 3)

    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False

    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)

    padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
    padded[:, :mn, :k] = ue8m0_tensor

    padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

    transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
    transposed[:, :, :] = padded

    aligned_x = transposed[:, :mn, :]

    return aligned_x.squeeze(0) if remove_dim else aligned_x


def transform_scale_ue8m0(sf, mn, weight_block_size=None):
    get_mn_major_tma_aligned_packed_ue8m0_tensor = _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
    if weight_block_size:
        assert weight_block_size == [128, 128]
        sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


def ceil_to_ue8m0_paddle(x: paddle.Tensor):
    """
    x > 0
    return 2 ^ ceil(log2(x))
    """
    # log2(x)
    log2_x = paddle.log(x) / paddle.log(paddle.to_tensor(2.0, dtype=x.dtype))
    # ceil
    ceil_log2_x = paddle.ceil(log2_x)
    # 2^k
    return paddle.pow(paddle.to_tensor(2.0, dtype=x.dtype), ceil_log2_x)


def per_token_quant_paddle(input_tensor, block_size, use_ue8m0: bool = False):
    MAX_VALUE = 448.0
    epsilon = 1e-10

    input_shape = input_tensor.shape
    token_num = input_shape[0]
    hidden_size = input_shape[1]

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
    if use_ue8m0:
        scale = ceil_to_ue8m0_paddle(scale)

    quanted_value = reshaped_input / scale

    quanted_x_padded_reshaped = quanted_value.to(paddle.float8_e4m3fn)
    quanted_x_padded = paddle.reshape(quanted_x_padded_reshaped, [token_num, padded_hidden_size])

    quanted_x = quanted_x_padded[:, :hidden_size]

    quanted_scale = paddle.squeeze(scale, axis=-1)
    if use_ue8m0:
        quanted_scale = transform_scale_ue8m0(quanted_scale, mn=quanted_x.shape[-2])

    return quanted_x, quanted_scale


def per_token_quant_padding_paddle(input_tensor, block_size, dtype, use_ue8m0):
    quanted_x, intermediate_scale = per_token_quant_paddle(input_tensor, block_size, use_ue8m0)
    token_num = input_tensor.shape[0]

    tma_alignment_elements = 4
    padded_token_num = ((token_num + tma_alignment_elements - 1) // tma_alignment_elements) * tma_alignment_elements

    hidden_size_scale = intermediate_scale.shape[1]
    if use_ue8m0:
        padded_scale = paddle.zeros([hidden_size_scale, padded_token_num], dtype=intermediate_scale.dtype).mT
    else:
        padded_scale = paddle.zeros([padded_token_num, hidden_size_scale], dtype="float32")

    padded_scale[:token_num, :] = intermediate_scale

    return quanted_x, padded_scale


class TestPerTokenQuant(unittest.TestCase):
    def get_input(self, shape, dtype):
        return paddle.randn(shape=shape, dtype=dtype)

    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 4
        self.hidden_size = 512
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)

    def test_per_token_quant(self):
        for use_ue8m0 in [False, True]:
            paddle_output, paddle_output_scale = per_token_quant_paddle(self.input_tensor, self.block_size, use_ue8m0)
            output, output_scale = per_token_quant_padding(self.input_tensor, self.block_size, use_ue8m0)
            if use_ue8m0:
                assert paddle_output_scale.strides == output_scale.strides
            np.testing.assert_allclose(paddle_output_scale.numpy(), output_scale.numpy(), rtol=1e-6)
            np.testing.assert_allclose(
                paddle.reshape(paddle_output_scale, [-1]).numpy(),
                paddle.reshape(output_scale, [-1]).numpy(),
                rtol=1e-6,
            )

            output_rel_diff = paddle.mean(
                paddle.abs(output.to(paddle.float32) - paddle_output.to(paddle.float32))
            ) / paddle.mean(paddle.abs(paddle_output.to(paddle.float32)))
            assert output_rel_diff < 0.001


class TestPerTokenQuantPadding(TestPerTokenQuant):
    def setUp(self) -> None:
        self.dtype = paddle.float16
        self.token_num = 8
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)

    def test_per_token_quant_padding(self):
        for use_ue8m0 in [False, True]:
            paddle_output, paddle_output_scale = per_token_quant_padding_paddle(
                self.input_tensor, self.block_size, self.dtype, use_ue8m0
            )
            output, output_scale = per_token_quant_padding(self.input_tensor, self.block_size, use_ue8m0)
            if use_ue8m0:
                assert paddle_output_scale.strides == output_scale.strides
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
        self.token_num = 8
        self.hidden_size = 128 * 4
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


class TestPerTokenQuantPaddingCase3(TestPerTokenQuantPadding):
    def setUp(self) -> None:
        self.dtype = paddle.bfloat16
        self.token_num = 8
        self.hidden_size = 128 * 8
        self.block_size = 128
        self.input_tensor = self.get_input(shape=[self.token_num, self.hidden_size], dtype=self.dtype)


if __name__ == "__main__":
    unittest.main()
