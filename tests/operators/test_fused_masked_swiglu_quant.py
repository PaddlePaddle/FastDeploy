import unittest

import numpy as np
import paddle

paddle.seed(2026)


def run_fused(x, token_nums, block_size):
    import fastdeploy.model_executor.ops.gpu as ops

    return ops.fused_mask_swiglu_fp8_quant(
        x,
        token_nums,
        block_size,
    )


def run_separate(x, token_nums, block_size):
    """Run separate operations (FastDeploy non-fused kernels)"""
    from fastdeploy.model_executor.ops.gpu import (
        group_swiglu_with_masked,
        masked_per_token_quant,
    )

    swiglu = group_swiglu_with_masked(x, token_nums)
    q, scale = masked_per_token_quant(swiglu, token_nums, block_size)
    return q, scale


# ------------------------------------------------------------
# Test case
# ------------------------------------------------------------


def benchmark_cuda(fn, warmup=10, repeat=10):
    """
    Benchmark a CUDA function using paddle.device.Event
    fn: callable with no return dependency on CPU
    """
    # warmup
    for _ in range(warmup):
        fn()
    paddle.device.synchronize()

    start = paddle.device.Event(enable_timing=True)
    end = paddle.device.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    end.synchronize()
    elapsed_ms = start.elapsed_time(end)  # ms

    return elapsed_ms / repeat


class TestFusedSwigluFP8Quant(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        # 10, 2048, 7168
        self.group_num = 10
        self.group_size = 2048
        self.hidden_dim = 7168
        self.block_size = 128
        self.x = paddle.randn(
            [self.group_num, self.group_size, self.hidden_dim * 2],
            dtype="bfloat16",
        )
        self.token_nums = paddle.to_tensor([5, 8, 0, 3, 3, 6, 8, 10, 5, 7], dtype="int32")

    def fused_vs_separate_exact_match(self):
        """
        Test fused kernel vs separate operations - should be exact match
        This compares FastDeploy's fused kernel vs FastDeploy's separate kernels
        """
        # Run separate operations
        q_ref, s_ref = run_separate(self.x, self.token_nums, self.block_size)

        # Run fused kernel
        q_fused, s_fused = run_fused(self.x, self.token_nums, self.block_size)

        def run_sep():
            run_separate(self.x, self.token_nums, self.block_size)

        def run_fus():
            run_fused(self.x, self.token_nums, self.block_size)

        t_sep = benchmark_cuda(run_sep)
        t_fus = benchmark_cuda(run_fus)

        print("\n====== Fused vs Separate Benchmark ======")
        print(f"Separate: {t_sep:.3f} ms")
        print(f"Fused   : {t_fus:.3f} ms")
        print(f"Speedup : {t_sep / t_fus:.2f}x")

        # ---------------- valid mask ----------------
        arange = paddle.arange(self.group_size, dtype="int32")
        valid = arange < self.token_nums.unsqueeze(1)  # [G, S]

        valid_flat = valid.reshape([-1])

        # ---------------- FP8 output ----------------
        q_ref_flat = q_ref.reshape([-1, q_ref.shape[-1]]).astype("float32")
        q_fused_flat = q_fused.reshape([-1, q_fused.shape[-1]]).astype("float32")

        np.testing.assert_array_equal(
            q_ref_flat[valid_flat].numpy(),
            q_fused_flat[valid_flat].numpy(),
        )

        # ---------------- scale ----------------
        s_ref_flat = s_ref.reshape([-1, s_ref.shape[-1]])
        s_fused_flat = s_fused.reshape([-1, s_fused.shape[-1]])

        np.testing.assert_array_equal(
            s_ref_flat[valid_flat].numpy(),
            s_fused_flat[valid_flat].numpy(),
        )

    def test_fused(self):
        self.fused_vs_separate_exact_match()


if __name__ == "__main__":
    unittest.main()
