# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import random
import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)


class TestBatchInvariantForLogsoftmax(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def create_softmax_trap_tensor(self, B, D, dtype):
        """
        Constructs a "trap" tensor designed to trigger batch-invariance issues in Softmax/LogSoftmax.
        Inspired by https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

        Principle:
        The goal is to make the result of `exp(a - max(a))` contain numbers spanning an extremely wide numerical range
        (e.g., 1.0, 1e-5, 1e-10, and many numbers close to 0).
        When summing these numbers using parallel reduction, different summation orders (due to parallelism)
        can produce different accumulated rounding errors, leading to a subtle difference between
        batch (parallel) and single-sample (serial) computation results.
        """
        # 1. Determine the desired values after `exp` and calculate the required input values using log().
        max_val = 20.0

        # Offsets relative to max_val. These offsets result in values spanning vastly different orders of magnitude after exp.
        trap_values = [
            max_val,  # Corresponds to exp(a-max) -> 1.0
            max_val - 4.6,  # Corresponds to exp(a-max) -> ~1e-2
            max_val - 11.5,  # Corresponds to exp(a-max) -> ~1e-5
            max_val - 23.0,  # Corresponds to exp(a-max) -> ~1e-10
        ]

        # 2. Create a background tensor filled with a very large negative number.
        background_val = -1000.0
        a = paddle.full((B, D), background_val, dtype=dtype)

        # 3. Scatter these "trap" values at random positions in each row.
        for i in range(B):
            # Randomly shuffle the positions of the trap values for each row to increase non-determinism.
            indices = random.sample(range(D), k=len(trap_values))
            for j, val in enumerate(trap_values):
                a[i, indices[j]] = val

        return a

    def test_batch_invariance(self, B: int = 2048, D: int = 4096, dtype=paddle.float32):
        a = self.create_softmax_trap_tensor(B, D, dtype)

        # Method 1: log_softmax on batch size 1 (first row)
        out1 = paddle.nn.functional.log_softmax(a[:1])

        # Method 2: log_softmax on full batch, then slice (first row)
        out2 = paddle.nn.functional.log_softmax(a)[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item() == 0, diff

    def run_iters(self, iters=10, ass=False):
        for dtype in [paddle.float32, paddle.bfloat16, paddle.float16]:
            is_deterministic = True
            difflist = []
            for i in range(iters):
                isd, df = self.test_batch_invariance(dtype=dtype)
                is_deterministic = is_deterministic and isd
                difflist.append(df)
            print(
                f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations"
            )
            if ass:
                assert max(difflist) == 0

    def test_case(self):
        # Test with standard Paddle (likely to show differences)
        print("Standard Paddle:")
        with set_batch_invariant_mode(False):
            self.run_iters(ass=False)
        # Test with batch-invariant operations
        print("\nBatch-Invariant Mode:")
        with set_batch_invariant_mode(True):
            self.run_iters(ass=True)


if __name__ == "__main__":
    unittest.main()
    """
    Even in Standard Paddle, we can achieve deterministic results, so maybe the standard implementation is already batch-invariant?

    After reviewing the four implementations called by the dispatcher function `SoftmaxForwardCUDAKernelDriverImpl` (dispatched by 'D')
    in `paddle/phi/kernels/gpudnn/softmax_gpudnn.h`:

    1. SwitchWarpSoftmaxForward (one Warp processes 1-2 rows)
    2. LaunchKeMatrixSoftmaxForwardKernel (one Block processes one row)
    3. LaunchSoftmaxForwardCudnnKernel (the Cudnn implementation)
    4. LaunchNormalSoftmaxForward (in one Block, threads with the same threadIdx.x [a "thread column"] cooperate to process one row)

    Excluding the Cudnn implementation, the other three custom implementations are almost certainly batch-invariant.(Need someone check again)
    The determinism of the Cudnn implementation is uncertain.

    However, in practice, this testcase (D=4096) is dispatched to the Cudnn implementation,
    while Qwen-3 8B is dispatched to the LaunchKeMatrixSoftmaxForwardKernel implementation.

    Result:

    Standard Paddle:
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float16 in 10 iterations

    Batch-Invariant Mode:
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float16 in 10 iterations
    """
