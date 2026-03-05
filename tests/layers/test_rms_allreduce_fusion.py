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

import time
import unittest
from unittest.mock import Mock

import numpy as np
import paddle
import paddle.distributed as dist


class TestFlashInferAllReduceResidualRMSNorm(unittest.TestCase):
    """Test FlashInfer AllReduce + Residual + RMSNorm fused operator"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
        dist.init_parallel_env()

    def setUp(self):
        """Initialize each test case"""
        # Fix random seed for reproducibility
        paddle.seed(42)
        np.random.seed(42)

        self.dtype = paddle.float32
        self.token_num = 128
        self.hidden_dim = 768
        self.eps = 1e-6
        self.epsilon = 1e-6
        self.max_token_num = 2048

        # Create mock FDConfig
        self.fd_config = Mock()
        self.fd_config.parallel_config = Mock()
        self.fd_config.parallel_config.tensor_parallel_size = dist.get_world_size()
        self.begin_norm_axis = 1

        # Performance test params - increase iterations for stability
        self.warmup_iterations = 20  # Increase warmup
        self.test_iterations = 200  # Increase test iterations

    def tearDown(self):
        """Clean up resources"""
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()

    def create_test_tensors(self):
        """Create test tensors"""
        input_tensor = paddle.randn([self.token_num, self.hidden_dim], dtype=self.dtype)
        residual = paddle.randn([self.token_num, self.hidden_dim], dtype=self.dtype)
        weight = paddle.randn([self.hidden_dim], dtype=self.dtype)
        return input_tensor, residual, weight

    def compute_reference_output(self, input_tensor, residual, weight, eps):
        """Reference implementation: manually compute AllReduce + Residual + RMSNorm"""
        # # Step 1: AllReduce (identity on single device)
        # allreduce_out = input_tensor.clone()
        # Apply all reduce operator
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        # Step 2: Add residual
        residual_out = input_tensor + residual

        # Step 3: RMSNorm
        variance = residual_out.pow(2).mean(axis=-1, keepdim=True)
        norm_out = residual_out * paddle.rsqrt(variance + eps)
        norm_out = norm_out * weight

        # dist.all_reduce(residual_out, op=dist.ReduceOp.SUM)
        return norm_out, residual_out

    def paddle_rms_fuse(self, input_tensor, residual, weight, eps):
        from paddle.incubate.nn.functional import fused_rms_norm

        # Apply all reduce operator
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        out_fused = fused_rms_norm(
            input_tensor,
            norm_weight=weight,
            norm_bias=None,
            epsilon=eps,
            begin_norm_axis=self.begin_norm_axis,
            bias=None,
            residual=residual,
        )

        return out_fused[0], out_fused[1]

    def flashinfer_rms_fuse(self, input_tensor, residual, weight, eps):
        """FlashInfer fused operator"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        norm_out, residual_out = flashinfer_allreduce_residual_rmsnorm(
            fd_config=self.fd_config,
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            eps=eps,
            max_token_num=self.max_token_num,
            use_oneshot=False,
        )
        return norm_out, residual_out

    def benchmark_function(self, func, *args, name="", **kwargs):
        """
        Improved performance benchmark
        - Wait for GPU frequency stabilization
        - Use median instead of mean (more stable)
        - Filter outliers
        """
        # Force GPU frequency stabilization
        if paddle.is_compiled_with_cuda():
            for _ in range(5):
                paddle.device.cuda.synchronize()
                time.sleep(0.01)

        # Warmup - thorough warm-up
        for _ in range(self.warmup_iterations):
            result = func(*args, **kwargs)
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

        # Extra wait to ensure GPU stability
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
            time.sleep(0.1)

        # Benchmark run
        times = []
        for i in range(self.test_iterations):
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            start = time.perf_counter()
            result = func(*args, **kwargs)

            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            end = time.perf_counter()
            elapsed = (end - start) * 1000  # Convert to milliseconds
            times.append(elapsed)

        times = np.array(times)

        # Filter outliers using IQR method
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_times = times[(times >= lower_bound) & (times <= upper_bound)]

        # Fall back to raw data if too many samples filtered out
        if len(filtered_times) < self.test_iterations * 0.5:
            filtered_times = times

        # Statistics
        avg_time = np.mean(filtered_times)
        median_time = np.median(filtered_times)
        std_time = np.std(filtered_times)
        min_time = np.min(filtered_times)
        max_time = np.max(filtered_times)
        cv = (std_time / avg_time) * 100  # Coefficient of variation (%)

        print(f"\n{'='*70}")
        print(f"Performance Benchmark: {name}")
        print(f"{'='*70}")
        print(f"Iterations: {len(filtered_times)}/{self.test_iterations} (after {self.warmup_iterations} warmup)")
        print(f"Median:     {median_time:.4f} ms  (most stable metric)")
        print(f"Average:    {avg_time:.4f} ms")
        print(f"Std Dev:    {std_time:.4f} ms  (CV: {cv:.2f}%)")
        print(f"Min:        {min_time:.4f} ms")
        print(f"Max:        {max_time:.4f} ms")
        print(f"{'='*70}\n")

        # Return median (more stable) and result
        return median_time, result

    def test_accuracy_fused_vs_reference(self):
        """Test accuracy of fused operator vs reference implementation"""
        input_tensor, residual, weight = self.create_test_tensors()
        reference_output, ref_res = self.compute_reference_output(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        fused_output, paddle_res = self.paddle_rms_fuse(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        flashinfer_output, flashinfer_res = self.flashinfer_rms_fuse(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        # Verify results
        np.testing.assert_allclose(fused_output.numpy(), reference_output.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_res.numpy(), paddle_res.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(flashinfer_output.numpy(), reference_output.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_res.numpy(), flashinfer_res.numpy(), rtol=1e-5, atol=1e-5)


class TestFlashInferWorkspaceManager(unittest.TestCase):
    """Test FlashInferWorkspaceManager"""

    def setUp(self):
        """Initialize"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            FlashInferWorkspaceManager,
        )

        self.manager = FlashInferWorkspaceManager()

    def test_initialization(self):
        """Test initialization state"""
        self.assertIsNone(self.manager.workspace_tensor)
        self.assertIsNone(self.manager.ipc_handles)
        self.assertIsNone(self.manager.world_size)
        self.assertIsNone(self.manager.rank)
        self.assertFalse(self.manager.initialized)

    def test_cleanup(self):
        """Test cleanup functionality"""
        self.manager.cleanup()
        self.assertFalse(self.manager.initialized)
        self.assertIsNone(self.manager.workspace_tensor)


def run_tests():
    """Run tests directly (called by subprocess after distributed launch)"""
    unittest.main(verbosity=2)


def check_gpus(gpu_ids):
    """Check whether the specified GPUs are available, raise error if not"""
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("Paddle is not compiled with CUDA support, cannot use GPU")

    available_count = paddle.device.cuda.device_count()
    if available_count == 0:
        raise RuntimeError("No available GPU detected")

    missing = [gid for gid in gpu_ids if gid >= available_count]
    if missing:
        raise RuntimeError(
            f"Required GPU {missing}, but only {available_count} GPU(s) detected (index 0~{available_count - 1})"
        )

    print(f"GPU check passed: required {gpu_ids}, {available_count} GPU(s) available")


def run_distributed():
    """Launch multi-GPU distributed test via paddle.distributed.launch as subprocess"""
    import os
    import subprocess
    import sys

    gpu_ids = [0, 1]
    check_gpus(gpu_ids)

    gpus_str = ",".join(str(g) for g in gpu_ids)
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        f"--gpus={gpus_str}",
        script_path,
        "--run-tests",
    ]
    print(f"Launching distributed test: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(script_path))
    sys.exit(result.returncode)


if __name__ == "__main__":
    import sys

    if "--run-tests" in sys.argv:
        # Launched by paddle.distributed.launch, run tests directly
        sys.argv.remove("--run-tests")
        run_tests()
    else:
        # Default entry: launch distributed test as subprocess
        run_distributed()
