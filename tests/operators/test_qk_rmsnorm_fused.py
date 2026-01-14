"""
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
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.triton_ops import qk_rmsnorm_fused
from tests.utils import OpPerformanceTester

paddle.set_default_dtype("bfloat16")
paddle.seed(99)


class TestQKNorm(unittest.TestCase):
    def setUp(self) -> None:
        # Qwen3-30B-A3B TP1
        self.hidden_size = 2048
        self.num_attention_heads = 32
        self.num_key_value_heads = 4
        self.num_hidden_layers = 48
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        self.tp_size = 1

        # # Qwen3-235B-A22B TP4
        # self.hidden_size = 4096
        # self.num_attention_heads = 64
        # self.num_key_value_heads = 4
        # self.num_hidden_layers = 94
        # self.head_dim = 128
        # self.rms_norm_eps = 1e-6
        # self.tp_size = 4

        #  # GLM_4.6 TP4
        # self.hidden_size = 5120
        # self.num_attention_heads = 96
        # self.num_key_value_heads = 8
        # self.num_hidden_layers = 92
        # self.head_dim = 128
        # self.rms_norm_eps = 1e-5
        # self.tp_size = 4

        self.num_kv_heads_replicas = max(1, self.tp_size // self.num_key_value_heads)
        self.q_size = self.num_attention_heads * self.head_dim // self.tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim * self.num_kv_heads_replicas // self.tp_size
        self.q_norm_weight = paddle.randn([self.head_dim], paddle.bfloat16)
        self.k_norm_weight = paddle.randn([self.head_dim], paddle.bfloat16)

    def qk_norm_paddle(self, qkv_out):
        q, k, v = qkv_out.split([self.q_size, self.kv_size, self.kv_size], axis=-1)
        q_by_head = q.reshape([*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim])
        q_by_head = paddle.incubate.nn.functional.fused_rms_norm(
            q_by_head, self.q_norm_weight, None, self.rms_norm_eps, begin_norm_axis=2
        )[0]
        q = q_by_head.reshape(q.shape)

        k_by_head = k.reshape([*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim])
        k_by_head = paddle.incubate.nn.functional.fused_rms_norm(
            k_by_head, self.k_norm_weight, None, self.rms_norm_eps, begin_norm_axis=2
        )[0]
        k = k_by_head.reshape(k.shape)

        qkv_out = paddle.concat([q, k, v], axis=-1)
        return qkv_out

    def qk_norm_triton_fused(self, qkv_out):
        qkv_out = qk_rmsnorm_fused(
            qkv_out,
            self.q_norm_weight,
            self.k_norm_weight,
            self.rms_norm_eps,
            self.q_size,
            self.kv_size,
            self.head_dim,
        )
        return qkv_out

    def test_qk_norm_paddle_performance(self):
        tester_paddle = OpPerformanceTester(
            op_name="qk_norm_paddle",
            op_fn=self.qk_norm_paddle,
            num_layers=self.num_hidden_layers,
        )

        tester_paddle.benchmark(
            input_size=self.head_dim
            * (self.num_attention_heads // self.tp_size + 2 * self.num_key_value_heads // self.tp_size),
            batch_sizes=[1, 8, 64, 128, 1024, 2048, 4096, 8192],
        )

    def test_qk_norm_fused_performance(self):
        tester = OpPerformanceTester(
            op_name="qk_norm_triton_fused",
            op_fn=self.qk_norm_triton_fused,
            num_layers=self.num_hidden_layers,
        )
        tester.benchmark(
            input_size=self.head_dim
            * (self.num_attention_heads // self.tp_size + 2 * self.num_key_value_heads // self.tp_size),
            batch_sizes=[1, 8, 64, 128, 1024, 2048, 4096, 8192],
        )

    def test_qk_norm_result(self):
        x = paddle.randn(
            [
                128,
                self.head_dim
                * (self.num_attention_heads // self.tp_size + 2 * self.num_key_value_heads // self.tp_size),
            ],
            paddle.bfloat16,
        )
        out_paddle = self.qk_norm_paddle(x)
        out_triton_fused = self.qk_norm_triton_fused(x)
        np.testing.assert_allclose(out_triton_fused.numpy(), out_paddle.numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
