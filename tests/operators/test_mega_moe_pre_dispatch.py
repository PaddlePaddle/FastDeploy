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

from fastdeploy.model_executor.ops.gpu import mega_moe_pre_dispatch
from dataclasses import dataclass

@dataclass
class FakeBuffer:
    x: paddle.Tensor
    x_sf: paddle.Tensor
    topk_idx: paddle.Tensor
    topk_weights: paddle.Tensor



def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: paddle.Tensor):
    bits = x.abs().astype("float32").view(paddle.int32)
    mask_ff = paddle.to_tensor(0xFF, dtype=paddle.int32)
    mask_mantissa = paddle.to_tensor(0x7FFFFF, dtype=paddle.int32)
    exp = ((bits >> 23) & mask_ff) + ((bits & mask_mantissa) != 0).astype("int32")
    return (exp.clip(1, 254) << 23).view(paddle.float32)


def pack_ue8m0_to_int(x: paddle.Tensor):
    assert x.dtype == paddle.float32 and x.shape[-1] % 4 == 0
    x_bits = x.view(paddle.int32)
    mantissa_mask = paddle.to_tensor((1 << 23) - 1, dtype=paddle.int32)
    assert bool(((x_bits & mantissa_mask) == 0).all())
    return (x_bits >> 23).astype(paddle.uint8).view(paddle.int32)


def per_token_cast_to_fp8(
    x: paddle.Tensor,
    use_ue8m0: bool,
    gran_k: int = 128,
    use_packed_ue8m0: bool = False,
):
    assert len(x.shape) == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = paddle.zeros((m, padded_n), dtype=x.dtype)
    x_padded[:, :n] = x
    x_view = x_padded.reshape([m, padded_n // gran_k, gran_k])
    x_amax = x_view.abs().astype("float32").amax(axis=2).reshape([m, padded_n // gran_k]).clip(min=1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).astype(paddle.float8_e4m3fn).reshape([m, padded_n])[:, :n]
    return x_fp8.contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


class TestMegaMoEPreDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.seed(2025)

    def setUp(self):
        self.num_experts = 160
        self.num_max_tokens_per_rank = 8192
        self.top_k = 6
        self.hidden_size = 7168
        self.moe_intermediate_size = 3584
        self.group_size = 32
        self.num_tokens = 128

        self.x = paddle.randn([self.num_tokens, self.hidden_size], dtype=paddle.bfloat16)
        scores = paddle.randn((self.num_tokens, self.num_experts), dtype=paddle.float32)
        self.topk_weights, self.topk_idx = paddle.topk(scores, self.top_k, axis=-1, largest=True, sorted=False)
        self.topk_idx = self.topk_idx.astype("int64")
        self.topk_weights = self.topk_weights.astype("float32")

    def _new_buffer(self):
        x = paddle.zeros([self.num_max_tokens_per_rank, self.hidden_size], dtype=paddle.bfloat16).astype("float8_e4m3fn")
        x_sf = paddle.zeros([self.num_max_tokens_per_rank, self.hidden_size // self.group_size // 4], paddle.int32)
        topk_idx = paddle.zeros([self.num_max_tokens_per_rank, self.top_k], dtype=paddle.int64)
        topk_weights = paddle.zeros([self.num_max_tokens_per_rank, self.top_k], dtype=paddle.float32)
        fake_buffer = FakeBuffer(
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights
        )

        return fake_buffer


    def mega_moe_pre_dispatch_ref(self, x: paddle.Tensor, topk_idx: paddle.Tensor, topk_weights: paddle.Tensor):
        num_tokens = x.shape[0]
        x_fp8, x_scale_tensor = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=self.group_size, use_packed_ue8m0=True
        )
        return (
            x_fp8,
            x_scale_tensor,
            topk_idx.astype("int64"),
            topk_weights.astype("float32"),
        )

    def test_mega_moe_pre_dispatch(self):
        buffer = self._new_buffer()
        buffer.topk_idx[self.num_tokens :] = -2
        buffer.topk_weights[self.num_tokens :] = 3.0

        mega_moe_pre_dispatch(
            self.x,
            self.topk_idx,
            self.topk_weights,
            buffer.x,
            buffer.x_sf,
            buffer.topk_idx,
            buffer.topk_weights,
            self.num_max_tokens_per_rank,
            self.group_size,
        )
        paddle.device.synchronize()

        x_ref, x_sf_ref, topk_idx_ref, topk_weights_ref = self.mega_moe_pre_dispatch_ref(
            self.x, self.topk_idx, self.topk_weights
        )

        np.testing.assert_allclose(
            buffer.x[: self.num_tokens].astype("float32").numpy(),
            x_ref.astype("float32").numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            buffer.x_sf[: self.num_tokens].numpy(),
            x_sf_ref.numpy(),
        )
        np.testing.assert_array_equal(
            buffer.topk_idx[: self.num_tokens].numpy(),
            topk_idx_ref.numpy(),
        )
        np.testing.assert_allclose(
            buffer.topk_weights[: self.num_tokens].numpy(),
            topk_weights_ref.numpy(),
            rtol=0,
            atol=0,
        )
        padded_max = buffer.x.shape[0]
        np.testing.assert_array_equal(
            buffer.topk_idx[self.num_tokens :].numpy(),
            np.full((padded_max - self.num_tokens, self.top_k), -2, dtype=np.int64),
        )
        np.testing.assert_allclose(
            buffer.topk_weights[self.num_tokens :].numpy(),
            np.full((padded_max - self.num_tokens, self.top_k), 3.0, dtype=np.float32),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
