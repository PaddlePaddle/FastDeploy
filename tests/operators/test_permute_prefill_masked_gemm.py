"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.ops.gpu import prefill_permute_to_masked_gemm


def call_prefill_permute_to_masked_gemm(
    x: paddle.Tensor,
    scale: paddle.Tensor,
    topk_ids: paddle.Tensor,
    num_local_experts: int,
    max_token_num: int,
):
    """
    Permute input tokens and scales from token-major to expert-major layout
    for MoE masked GEMM operations.

    Args:
        x: Input hidden states [num_tokens, hidden].
        scale: Input scales [num_tokens, hidden_scale].
        topk_ids: Expert routing indices [num_tokens, topk] (int64 or int32).
        num_local_experts: Number of local experts on this device.
        max_token_num: Maximum tokens per expert buffer.

    Returns:
        tuple: (permute_x, permute_scale, permuted_indice_map, token_nums_per_expert)
    """
    if topk_ids.dtype != paddle.int64:
        topk_ids = topk_ids.cast(paddle.int64)

    results = prefill_permute_to_masked_gemm(x, scale, topk_ids, num_local_experts, max_token_num)

    return results[0], results[1], results[2], results[3]


class TestPrefillPermuteToMaskedGemm(unittest.TestCase):
    """
    Test cases for prefill_permute_to_masked_gemm kernel.
    """

    def setUp(self):
        paddle.seed(2024)
        np.random.seed(2024)

    def _get_expected_tokens_per_expert(self, x, scale, topk_ids, num_local_experts):
        num_tokens = x.shape[0]
        _, topk = topk_ids.shape

        expert_to_tokens = {i: [] for i in range(num_local_experts)}
        token_nums_per_expert = np.zeros(num_local_experts, dtype=np.int32)

        for token_idx in range(num_tokens):
            for k in range(topk):
                expert_idx = topk_ids[token_idx, k]
                if expert_idx != -1:
                    expert_to_tokens[expert_idx].append((x[token_idx, :].copy(), scale[token_idx, :].copy()))
                    token_nums_per_expert[expert_idx] += 1

        return expert_to_tokens, token_nums_per_expert

    def _run_and_verify(
        self,
        num_tokens,
        hidden_size,
        hidden_scale,
        num_local_experts,
        max_token_num,
        topk,
        x_dtype=paddle.float8_e4m3fn,
        scale_dtype=paddle.int32,
        sparsity=0.3,
    ):

        if x_dtype == paddle.float8_e4m3fn:
            x_np = np.random.randn(num_tokens, hidden_size).astype(np.float32)
            x_np = np.clip(x_np, -448, 448)
            x = paddle.to_tensor(x_np).cast(paddle.float8_e4m3fn)
        elif x_dtype == paddle.bfloat16:
            x_np = np.random.randn(num_tokens, hidden_size).astype(np.float32)
            x = paddle.to_tensor(x_np).cast(paddle.bfloat16)
        else:
            x_np = np.random.randn(num_tokens, hidden_size).astype(np.float32)
            x = paddle.to_tensor(x_np)

        scale_np = np.random.rand(num_tokens, hidden_scale).astype(np.float32)
        scale = paddle.to_tensor(scale_np).cast(scale_dtype).contiguous()

        topk_ids_np = np.zeros((num_tokens, topk), dtype=np.int64)
        for i in range(num_tokens):
            experts = np.random.choice(num_local_experts, size=min(topk, num_local_experts), replace=False)
            if len(experts) < topk:
                topk_ids_np[i, : len(experts)] = experts
                topk_ids_np[i, len(experts) :] = -1
            else:
                topk_ids_np[i, :] = experts
        mask = np.random.rand(num_tokens, topk) < sparsity
        topk_ids_np[mask] = -1

        # The kernel breaks early when a block encounters an all-(-1) row,
        # so valid rows must come first in token order.
        # Sort rows: rows with at least one valid expert first, all-(-1) rows last.
        valid_mask = (topk_ids_np >= 0).any(axis=1)
        sorted_idx = np.concatenate([np.where(valid_mask)[0], np.where(~valid_mask)[0]])
        topk_ids_np = topk_ids_np[sorted_idx]
        x_np = x_np[sorted_idx]
        scale_np = scale_np[sorted_idx]
        x = paddle.to_tensor(x_np).cast(x_dtype)
        scale = paddle.to_tensor(scale_np).cast(scale_dtype).contiguous()
        topk_ids = paddle.to_tensor(topk_ids_np).cast(paddle.int64)

        permute_x, permute_scale, permuted_indice_map, token_nums_per_expert = call_prefill_permute_to_masked_gemm(
            x=x,
            scale=scale,
            topk_ids=topk_ids,
            num_local_experts=num_local_experts,
            max_token_num=max_token_num,
        )

        permute_x_result = permute_x.cast(paddle.float32).numpy()
        permute_scale_result = permute_scale.numpy()
        permuted_indice_map_result = permuted_indice_map.numpy()
        token_nums_result = token_nums_per_expert.numpy().flatten()

        x_ref_np = x.cast(paddle.float32).numpy()
        scale_ref_np = scale.numpy()

        expert_to_tokens, token_nums_ref = self._get_expected_tokens_per_expert(
            x=x_ref_np,
            scale=scale_ref_np,
            topk_ids=topk_ids_np,
            num_local_experts=num_local_experts,
        )

        np.testing.assert_array_equal(
            token_nums_result,
            token_nums_ref,
            err_msg=f"Token counts mismatch: kernel={token_nums_result}, ref={token_nums_ref}",
        )

        for expert_idx in range(num_local_experts):
            num_tokens_for_expert = token_nums_ref[expert_idx]
            if num_tokens_for_expert > 0:
                kernel_x_rows = permute_x_result[expert_idx, :num_tokens_for_expert, :]
                kernel_scale_rows = permute_scale_result[expert_idx, :num_tokens_for_expert, :]

                expected_tokens = expert_to_tokens[expert_idx]
                expected_x_rows = np.array([t[0] for t in expected_tokens])
                expected_scale_rows = np.array([t[1] for t in expected_tokens])

                kernel_x_sums = np.sort(np.sum(kernel_x_rows, axis=1))
                expected_x_sums = np.sort(np.sum(expected_x_rows, axis=1))

                np.testing.assert_allclose(
                    kernel_x_sums,
                    expected_x_sums,
                    rtol=1e-2,
                    atol=1e-1,
                    err_msg=f"Expert {expert_idx}: permute_x row sums mismatch",
                )

                kernel_scale_sums = np.sort(np.sum(kernel_scale_rows, axis=1))
                expected_scale_sums = np.sort(np.sum(expected_scale_rows, axis=1))

                np.testing.assert_allclose(
                    kernel_scale_sums,
                    expected_scale_sums,
                    rtol=1e-2,
                    atol=1e-2,
                    err_msg=f"Expert {expert_idx}: permute_scale row sums mismatch",
                )

        x_ref_np = x.cast(paddle.float32).numpy()
        for token_idx in range(num_tokens):
            for expert_slot in range(topk):
                permuted_idx = permuted_indice_map_result[token_idx, expert_slot]
                if permuted_idx >= 0:
                    expert_idx = permuted_idx // max_token_num
                    offset = permuted_idx % max_token_num
                    permuted_data = permute_x_result[expert_idx, offset, :]
                    original_data = x_ref_np[token_idx, :]
                    np.testing.assert_allclose(
                        permuted_data,
                        original_data,
                        rtol=1e-2,
                        atol=1e-1,
                        err_msg=f"Token {token_idx}: permuted_indice_map points to wrong data",
                    )

        return True

    def test_basic_topk4(self):
        self._run_and_verify(
            num_tokens=64,
            hidden_size=7168,
            hidden_scale=56,
            num_local_experts=8,
            max_token_num=128,
            topk=4,
            sparsity=0.2,
        )

    def test_basic_topk8(self):
        self._run_and_verify(
            num_tokens=64,
            hidden_size=7168,
            hidden_scale=56,
            num_local_experts=8,
            max_token_num=128,
            topk=8,
            sparsity=0.2,
        )

    def test_small_tokens(self):
        self._run_and_verify(
            num_tokens=4, hidden_size=1024, hidden_scale=8, num_local_experts=4, max_token_num=32, topk=4, sparsity=0.1
        )

    def test_large_tokens(self):
        self._run_and_verify(
            num_tokens=512,
            hidden_size=4096,
            hidden_scale=32,
            num_local_experts=16,
            max_token_num=256,
            topk=4,
            sparsity=0.3,
        )

    def test_high_sparsity(self):
        self._run_and_verify(
            num_tokens=128,
            hidden_size=2048,
            hidden_scale=16,
            num_local_experts=8,
            max_token_num=64,
            topk=4,
            sparsity=0.7,
        )

    def test_no_sparsity(self):
        self._run_and_verify(
            num_tokens=64,
            hidden_size=2048,
            hidden_scale=16,
            num_local_experts=8,
            max_token_num=128,
            topk=4,
            sparsity=0.0,
        )

    def test_single_expert(self):
        self._run_and_verify(
            num_tokens=32,
            hidden_size=1024,
            hidden_scale=8,
            num_local_experts=1,
            max_token_num=64,
            topk=4,
            sparsity=0.0,
        )

    def test_many_experts(self):
        self._run_and_verify(
            num_tokens=128,
            hidden_size=2048,
            hidden_scale=16,
            num_local_experts=32,
            max_token_num=64,
            topk=8,
            sparsity=0.3,
        )

    def test_bfloat16_input(self):
        self._run_and_verify(
            num_tokens=64,
            hidden_size=2048,
            hidden_scale=16,
            num_local_experts=8,
            max_token_num=128,
            topk=4,
            x_dtype=paddle.bfloat16,
            sparsity=0.2,
        )

    def test_very_large_tokens(self):
        self._run_and_verify(
            num_tokens=65536,
            hidden_size=7168,
            hidden_scale=56,
            num_local_experts=20,
            max_token_num=16384,
            topk=4,
            sparsity=0.3,
        )

    def test_very_large_tokens_with_fp32_scale(self):
        self._run_and_verify(
            num_tokens=65536,
            hidden_size=7168,
            hidden_scale=56,
            num_local_experts=20,
            max_token_num=16384,
            topk=4,
            sparsity=0.3,
            scale_dtype=paddle.float32,
        )

    def test_all_minus_one(self):
        num_tokens = 32
        hidden_size = 1024
        hidden_scale = 8
        num_local_experts = 4
        max_token_num = 64
        topk = 4

        x_np = np.random.randn(num_tokens, hidden_size).astype(np.float32)
        x_np = np.clip(x_np, -448, 448)
        x = paddle.to_tensor(x_np).cast(paddle.float8_e4m3fn)

        scale_np = np.random.rand(num_tokens, hidden_scale).astype(np.float32)
        scale = paddle.to_tensor(scale_np)

        topk_ids = paddle.full([num_tokens, topk], -1, dtype=paddle.int32)

        permute_x, permute_scale, permuted_indice_map, token_nums_per_expert = call_prefill_permute_to_masked_gemm(
            x=x,
            scale=scale,
            topk_ids=topk_ids,
            num_local_experts=num_local_experts,
            max_token_num=max_token_num,
        )

        token_nums_result = token_nums_per_expert.numpy().flatten()
        expected = np.zeros(num_local_experts, dtype=np.int32)
        np.testing.assert_array_equal(token_nums_result, expected)
        self.assertEqual(permuted_indice_map.shape, [num_tokens, topk])


if __name__ == "__main__":
    unittest.main()
