"""
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
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import depermute_prefill_combine


def call_depermute_prefill_combine(
    x: paddle.Tensor,
    indice_map: paddle.Tensor,
    topk_weights: paddle.Tensor,
    num_worst_tokens: int,
):
    """
    Depermute and combine expert outputs back to token-major layout.

    Args:
        x: Expert outputs [num_local_experts, max_tokens_per_expert, hidden].
        indice_map: Flat index tensor [num_worst_tokens, topk] (int32).
        topk_weights: Combination weights [num_worst_tokens, topk] (float32).
        num_worst_tokens: Number of output tokens to produce.

    Returns:
        depermuted_x: Combined output [num_worst_tokens, hidden].
    """
    results = depermute_prefill_combine(x, indice_map, topk_weights, num_worst_tokens)

    return results


class TestDepermutePrefillCombine(unittest.TestCase):
    """
    Test cases for depermute_prefill_combine kernel.
    """

    def setUp(self):
        paddle.seed(2024)
        np.random.seed(2024)
        paddle.set_device("gpu")

    def _compute_reference(
        self,
        x_np,
        indice_map_np,
        topk_weights_np,
        num_worst_tokens,
        max_num_tokens_per_expert,
    ):
        hidden = x_np.shape[2]
        topk = indice_map_np.shape[1]
        depermuted_x = np.zeros((num_worst_tokens, hidden), dtype=np.float32)

        for token_idx in range(num_worst_tokens):
            for k in range(topk):
                indice = indice_map_np[token_idx, k]
                if indice >= 0:
                    expert_idx = indice // max_num_tokens_per_expert
                    offset = indice % max_num_tokens_per_expert
                    weight = topk_weights_np[token_idx, k]
                    depermuted_x[token_idx, :] += x_np[expert_idx, offset, :] * weight

        return depermuted_x

    def _run_and_verify(
        self,
        num_worst_tokens,
        num_local_experts,
        max_num_tokens_per_expert,
        hidden,
        topk,
        x_dtype=paddle.bfloat16,
        sparsity=0.2,
        rtol=1e-2,
        atol=1e-2,
    ):
        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        if x_dtype == paddle.bfloat16:
            x = paddle.to_tensor(x_np).cast(paddle.bfloat16)
        elif x_dtype == paddle.float8_e4m3fn:
            x_np = np.clip(x_np, -448, 448)
            x = paddle.to_tensor(x_np).cast(paddle.float8_e4m3fn)
        else:
            x = paddle.to_tensor(x_np)

        indice_map_np = np.zeros((num_worst_tokens, topk), dtype=np.int32)
        for token_idx in range(num_worst_tokens):
            used_positions = {}
            has_valid_index = False
            for k in range(topk):
                if k == topk - 1 and not has_valid_index:
                    should_be_invalid = False
                else:
                    should_be_invalid = np.random.rand() < sparsity

                if should_be_invalid:
                    indice_map_np[token_idx, k] = -1
                else:
                    expert_idx = np.random.randint(0, num_local_experts)
                    if expert_idx not in used_positions:
                        used_positions[expert_idx] = []
                    offset = np.random.randint(0, max_num_tokens_per_expert)
                    attempts = 0
                    while offset in used_positions.get(expert_idx, []) and attempts < 10:
                        offset = np.random.randint(0, max_num_tokens_per_expert)
                        attempts += 1
                    used_positions[expert_idx].append(offset)
                    indice_map_np[token_idx, k] = expert_idx * max_num_tokens_per_expert + offset
                    has_valid_index = True

        indice_map = paddle.to_tensor(indice_map_np).cast(paddle.int32)

        topk_weights_np = np.random.rand(num_worst_tokens, topk).astype(np.float32)
        row_sums = topk_weights_np.sum(axis=1, keepdims=True)
        topk_weights_np = topk_weights_np / (row_sums + 1e-6)
        topk_weights_np[indice_map_np == -1] = 0.0
        topk_weights = paddle.to_tensor(topk_weights_np).cast(paddle.float32)

        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )

        x_ref_np = x.cast(paddle.float32).numpy()
        expected = self._compute_reference(
            x_np=x_ref_np,
            indice_map_np=indice_map_np,
            topk_weights_np=topk_weights_np,
            num_worst_tokens=num_worst_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )

        result = depermuted_x.cast(paddle.float32).numpy()

        self.assertEqual(result.shape, (num_worst_tokens, hidden))

        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol, err_msg="Depermuted output mismatch")

        return True

    def test_basic_topk4(self):
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=7168,
            topk=4,
            sparsity=0.2,
        )

    def test_basic_topk8(self):
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=7168,
            topk=8,
            sparsity=0.2,
        )

    def test_small_tokens(self):
        self._run_and_verify(
            num_worst_tokens=4,
            num_local_experts=4,
            max_num_tokens_per_expert=32,
            hidden=1024,
            topk=4,
            sparsity=0.1,
        )

    def test_large_tokens(self):
        self._run_and_verify(
            num_worst_tokens=512,
            num_local_experts=16,
            max_num_tokens_per_expert=256,
            hidden=4096,
            topk=4,
            sparsity=0.3,
        )

    def test_high_sparsity(self):
        self._run_and_verify(
            num_worst_tokens=128,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=2048,
            topk=4,
            sparsity=0.7,
        )

    def test_no_sparsity(self):
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=128,
            hidden=2048,
            topk=4,
            sparsity=0.0,
        )

    def test_single_expert(self):
        self._run_and_verify(
            num_worst_tokens=32,
            num_local_experts=1,
            max_num_tokens_per_expert=64,
            hidden=1024,
            topk=4,
            sparsity=0.0,
        )

    def test_many_experts(self):
        self._run_and_verify(
            num_worst_tokens=128,
            num_local_experts=32,
            max_num_tokens_per_expert=64,
            hidden=2048,
            topk=8,
            sparsity=0.3,
        )

    def test_small_hidden(self):
        self._run_and_verify(
            num_worst_tokens=64,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=256,
            topk=4,
            sparsity=0.2,
        )

    def test_large_hidden(self):
        self._run_and_verify(
            num_worst_tokens=32,
            num_local_experts=8,
            max_num_tokens_per_expert=64,
            hidden=14336,
            topk=4,
            sparsity=0.2,
        )

    def test_all_minus_one(self):
        num_worst_tokens = 32
        num_local_experts = 4
        max_num_tokens_per_expert = 64
        hidden = 1024
        topk = 4

        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        x = paddle.to_tensor(x_np).cast(paddle.bfloat16)

        indice_map = paddle.full([num_worst_tokens, topk], -1, dtype=paddle.int32)
        topk_weights = paddle.zeros([num_worst_tokens, topk], dtype=paddle.float32)

        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )

        result = depermuted_x.cast(paddle.float32).numpy()
        self.assertEqual(result.shape, (num_worst_tokens, hidden))

    def test_single_token(self):
        self._run_and_verify(
            num_worst_tokens=1,
            num_local_experts=4,
            max_num_tokens_per_expert=32,
            hidden=1024,
            topk=4,
            sparsity=0.0,
        )

    def test_uniform_weights(self):
        num_worst_tokens = 64
        num_local_experts = 8
        max_num_tokens_per_expert = 64
        hidden = 2048
        topk = 4

        x_np = np.random.randn(num_local_experts, max_num_tokens_per_expert, hidden).astype(np.float32)
        x = paddle.to_tensor(x_np).cast(paddle.bfloat16)

        indice_map_np = np.zeros((num_worst_tokens, topk), dtype=np.int32)
        for token_idx in range(num_worst_tokens):
            for k in range(topk):
                expert_idx = k % num_local_experts
                offset = token_idx % max_num_tokens_per_expert
                indice_map_np[token_idx, k] = expert_idx * max_num_tokens_per_expert + offset
        indice_map = paddle.to_tensor(indice_map_np).cast(paddle.int32)

        topk_weights_np = np.ones((num_worst_tokens, topk), dtype=np.float32) / topk
        topk_weights = paddle.to_tensor(topk_weights_np)

        depermuted_x = call_depermute_prefill_combine(
            x=x,
            indice_map=indice_map,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
        )

        x_ref_np = x.cast(paddle.float32).numpy()
        expected = self._compute_reference(
            x_np=x_ref_np,
            indice_map_np=indice_map_np,
            topk_weights_np=topk_weights_np,
            num_worst_tokens=num_worst_tokens,
            max_num_tokens_per_expert=max_num_tokens_per_expert,
        )

        result = depermuted_x.cast(paddle.float32).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
