import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess


class TestTritonMOEPreprocess(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_op(self, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M):
        # Convert numpy to Paddle Tensor and run operator
        topk_ids = paddle.to_tensor(topk_ids_np, dtype="int64")
        sorted_ids, expert_ids, num_tokens_post_pad = tritonmoe_preprocess(topk_ids, num_experts, GEMM_BLOCK_SIZE_M)
        return sorted_ids.numpy(), expert_ids.numpy(), num_tokens_post_pad.numpy()

    def _check_output_shapes(
        self, sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
    ):
        expected_max_num_tokens_padded = topk_ids_np.size + num_experts * (GEMM_BLOCK_SIZE_M - 1)
        self.assertEqual(sorted_ids.shape[0], expected_max_num_tokens_padded)

        expected_max_num_m_blocks = expected_max_num_tokens_padded // GEMM_BLOCK_SIZE_M
        self.assertEqual(expert_ids.shape[0], expected_max_num_m_blocks)

        self.assertEqual(num_tokens_post_pad.shape[0], 1)
        self.assertTrue(sorted_ids.dtype == np.int32)
        self.assertTrue(expert_ids.dtype == np.int32)
        self.assertTrue(num_tokens_post_pad.dtype == np.int32)

    def test_basic_case(self):
        """Basic fixed example test"""
        num_experts = 8
        GEMM_BLOCK_SIZE_M = 4
        topk_ids_np = np.array([[7, 6, 5, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 3, 2, 1]], dtype=np.int64)

        sorted_ids, expert_ids, num_tokens_post_pad = self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)
        self._check_output_shapes(
            sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
        )

    def test_all_supported_num_experts(self):
        """Test all supported num_experts branches"""
        GEMM_BLOCK_SIZE_M = 4
        supported_experts = [2, 8, 64, 128, 160, 256]

        for num_experts in supported_experts:
            with self.subTest(num_experts=num_experts):
                batch, topk = 4, 4
                topk_ids_np = np.random.randint(0, num_experts, size=(batch, topk), dtype=np.int64)
                sorted_ids, expert_ids, num_tokens_post_pad = self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)
                self._check_output_shapes(
                    sorted_ids, expert_ids, num_tokens_post_pad, topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M
                )

    def test_unsupported_num_experts(self):
        """Test unsupported num_experts raises OSError"""
        topk_ids_np = np.array([[0, 1], [1, 0]], dtype=np.int64)
        unsupported_experts = [3, 9, 65, 129]  # unsupported values
        GEMM_BLOCK_SIZE_M = 4

        for num_experts in unsupported_experts:
            with self.subTest(num_experts=num_experts):
                with self.assertRaises(OSError):
                    self._run_op(topk_ids_np, num_experts, GEMM_BLOCK_SIZE_M)


if __name__ == "__main__":
    unittest.main()
