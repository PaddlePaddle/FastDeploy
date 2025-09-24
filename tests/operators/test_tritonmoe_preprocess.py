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

from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess


class TestTritonMOEPreprocess(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_op(self, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M):
        """Convert numpy to Paddle Tensor and run operator"""
        topk_ids = paddle.to_tensor(topk_ids_np, dtype="int64")
        sorted_ids, expert_ids, num_tokens_post_pad = tritonmoe_preprocess(topk_ids, num_experts, GEMM_BLOCK_SIZE_M)
        return sorted_ids.numpy(), expert_ids.numpy(), num_tokens_post_pad.numpy()

    def _check_output_shapes(
        self, sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
    ):
        """Check output shapes and dtypes"""
        expected_max_num_tokens_padded = topk_ids_np.size + num_experts * (GEMM_BLOCK_SIZE_M - 1)
        self.assertEqual(sorted_ids.shape[0], expected_max_num_tokens_padded)

        expected_max_num_m_blocks = expected_max_num_tokens_padded // GEMM_BLOCK_SIZE_M
        self.assertEqual(expert_ids.shape[0], expected_max_num_m_blocks)

        self.assertEqual(num_tokens_post_pad.shape[0], 1)
        self.assertTrue(sorted_ids.dtype == np.int32)
        self.assertTrue(expert_ids.dtype == np.int32)
        self.assertTrue(num_tokens_post_pad.dtype == np.int32)

    def _check_output_values_basic(self, sorted_ids, expert_ids, num_tokens_post_pad):
        """Check expected values for the fixed example"""
        expected_sorted_ids = np.array(
            [
                8,
                12,
                16,
                16,
                4,
                9,
                15,
                16,
                5,
                10,
                14,
                16,
                6,
                11,
                13,
                16,
                3,
                7,
                16,
                16,
                2,
                16,
                16,
                16,
                1,
                16,
                16,
                16,
                0,
                16,
                16,
                16,
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(sorted_ids[: len(expected_sorted_ids)], expected_sorted_ids)

        expected_expert_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        np.testing.assert_array_equal(expert_ids[: len(expected_expert_ids)], expected_expert_ids)

        self.assertTrue(num_tokens_post_pad[0] % 4 == 0)

    def test_basic_case(self):
        """Basic fixed example test"""
        num_experts = 8
        GEMM_BLOCK_SIZE_M = 4
        topk_ids_np = np.array([[7, 6, 5, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int64)

        sorted_ids, expert_ids, num_tokens_post_pad = self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)
        self._check_output_shapes(
            sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
        )
        self._check_output_values_basic(sorted_ids, expert_ids, num_tokens_post_pad)

    def test_unsupported_num_experts(self):
        """Test unsupported num_experts raises OSError"""
        topk_ids_np = np.array([[0, 1], [1, 0]], dtype=np.int64)
        unsupported_experts = [3, 9, 65, 129]
        GEMM_BLOCK_SIZE_M = 4

        for num_experts in unsupported_experts:
            with self.subTest(num_experts=num_experts):
                with self.assertRaises(OSError):
                    self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)


if __name__ == "__main__":
    unittest.main()
