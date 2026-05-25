"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle
import paddle.distributed as dist


class TestFlashInferAllReduceResidualRMSNorm(unittest.TestCase):
    """Test FlashInfer AllReduce + Residual + RMSNorm fused operator"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if paddle.is_compiled_with_cuda():
            # Bind each rank to its own GPU explicitly; otherwise all ranks
            # default to "gpu:0" and cudaIpcOpenMemHandle fails with
            # "invalid device context".
            local_rank = int(
                os.environ.get("PADDLE_LOCAL_RANK", os.environ.get("FLAGS_selected_gpus", "0").split(",")[0])
            )
            paddle.set_device(f"gpu:{local_rank}")

            # paddle.distributed.launch remaps each rank's visible GPU to
            # index 0 inside the worker process. flashinfer's IPC calls go
            # through the cudart runtime API (cuda-python), which maintains
            # its own primary context separate from Paddle's driver context.
            # Explicitly activate cudart's primary context on device 0 here,
            # otherwise cudaIpcOpenMemHandle reports "invalid device context".
            try:
                from cuda import cudart

                cudart.cudaSetDevice(0)
                cudart.cudaFree(0)  # force primary context creation
            except ImportError:
                pass
        else:
            paddle.set_device("cpu")
        dist.init_parallel_env()
        if paddle.is_compiled_with_cuda():
            # Force the CUDA primary context to be created on the current
            # device before flashinfer's cudart IPC calls run.
            paddle.zeros([1]).cuda()
            paddle.device.cuda.synchronize()

    def setUp(self):
        """Initialize each test case"""
        # Fix random seed for reproducibility
        paddle.seed(42)
        np.random.seed(42)

        # NOTE: switched fp32 -> bf16 to mirror real model dtype on B GPUs.
        # Combined with use_oneshot=None below, this exercises the bf16 +
        # oneshot Lamport path, which is the suspected garbled-output path
        # on Blackwell (sm100).
        self.dtype = paddle.bfloat16
        self.token_num = 128
        self.hidden_dim = 4096
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
            # NOTE: do NOT pass use_oneshot=False here. We want the auto path
            # (use_oneshot=None) so the oneshot Lamport kernel is exercised,
            # matching how normalization.py calls it in the real model.
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

        # bf16 needs much looser tolerance than fp32. Cast to fp32 for
        # comparison to avoid numpy bf16 issues.
        if self.dtype == paddle.bfloat16:
            rtol, atol = 5e-2, 5e-2
            to_np = lambda t: t.astype("float32").numpy()  # noqa: E731
        else:
            rtol, atol = 1e-5, 1e-5
            to_np = lambda t: t.numpy()  # noqa: E731

        # Verify results
        np.testing.assert_allclose(to_np(fused_output), to_np(reference_output), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_np(ref_res), to_np(paddle_res), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_np(flashinfer_output), to_np(reference_output), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_np(ref_res), to_np(flashinfer_res), rtol=rtol, atol=atol)


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


class TestFlashInferWorkspaceManagerEdgeCases(unittest.TestCase):
    """Test FlashInferWorkspaceManager edge cases and fallback paths"""

    def setUp(self):
        """Initialize test fixtures"""
        # Patch before importing to test fallback paths
        self.patcher_has_flashinfer = patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer")
        self.mock_has_flashinfer = self.patcher_has_flashinfer.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher_has_flashinfer.stop()

    def test_initialization_early_return_when_already_initialized(self):
        """Test line 47: early return when already initialized with same world_size"""
        # Patch _flashinfer_comm to be available
        with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm") as mock_comm:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
            )

            manager = FlashInferWorkspaceManager()

            # First initialization
            manager.initialized = True
            manager.world_size = 2

            # Mock the comm functions
            mock_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion = Mock(return_value=(Mock(), Mock()))

            # Second initialization with same world_size - should return early
            manager.initialize(
                world_size=2,
                rank=0,
                max_token_num=2048,
                hidden_dim=4096,
            )

    def test_initialization_warning_when_comm_none(self):
        """Test lines 50-51: warning when _flashinfer_comm is None"""
        # Patch to ensure _get_flashinfer_comm returns None
        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._get_flashinfer_comm",
            return_value=None,
        ):
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
            )

            manager = FlashInferWorkspaceManager()

            # Should not raise, just log warning and return
            manager.initialize(
                world_size=2,
                rank=0,
                max_token_num=2048,
                hidden_dim=4096,
            )

            # Verify not initialized
            self.assertFalse(manager.initialized)

    def test_cleanup_with_exception(self):
        """Test lines 73-80: cleanup with exception handling"""
        with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm") as mock_comm:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
            )

            manager = FlashInferWorkspaceManager()
            manager.initialized = True
            manager.ipc_handles = Mock()
            manager.workspace_tensor = Mock()

            # Mock the destroy function to raise exception
            mock_comm.trtllm_destroy_ipc_workspace_for_all_reduce = Mock(side_effect=RuntimeError("Cleanup error"))

            # Should not raise, just log warning
            manager.cleanup()

            # Verify cleanup happened
            self.assertFalse(manager.initialized)
            self.assertIsNone(manager.workspace_tensor)
            self.assertIsNone(manager.ipc_handles)

    def test_cleanup_without_initialization(self):
        """Test cleanup when not initialized"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            FlashInferWorkspaceManager,
        )

        manager = FlashInferWorkspaceManager()
        manager.initialized = False

        # Should not raise
        manager.cleanup()

        # Verify state
        self.assertFalse(manager.initialized)


class TestEnsureWorkspaceInitialized(unittest.TestCase):
    """Test ensure_workspace_initialized fallback paths"""

    def setUp(self):
        """Initialize test fixtures"""
        self.patcher_has_flashinfer = patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer")
        self.mock_has_flashinfer = self.patcher_has_flashinfer.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher_has_flashinfer.stop()

    def test_ensure_workspace_when_flashinfer_not_available(self):
        """Test line 91: early return when flashinfer not available"""
        self.mock_has_flashinfer.return_value = False

        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            ensure_workspace_initialized,
        )

        fd_config = Mock()
        fd_config.parallel_config = Mock()
        fd_config.parallel_config.tensor_parallel_size = 2

        result = ensure_workspace_initialized(fd_config)

        # Should return False (not initialized)
        self.assertFalse(result)

    def test_ensure_workspace_when_comm_none(self):
        """Test ensure_workspace_initialized when _flashinfer_comm is None"""
        self.mock_has_flashinfer.return_value = True

        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._get_flashinfer_comm",
            return_value=None,
        ):
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                ensure_workspace_initialized,
            )

            fd_config = Mock()
            fd_config.parallel_config = Mock()
            fd_config.parallel_config.tensor_parallel_size = 2

            result = ensure_workspace_initialized(fd_config)

            # Should return False
            self.assertFalse(result)

    def test_ensure_workspace_single_gpu(self):
        """Test line 96: early return when world_size <= 1"""
        self.mock_has_flashinfer.return_value = True

        with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"):
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                ensure_workspace_initialized,
            )

            fd_config = Mock()
            fd_config.parallel_config = Mock()
            fd_config.parallel_config.tensor_parallel_size = 1

            with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion.dist.get_rank", return_value=0):
                result = ensure_workspace_initialized(fd_config)

            # Should return False for single GPU
            self.assertFalse(result)


class TestFlashInferAllReduceResidualRMSNormFallbacks(unittest.TestCase):
    """Test flashinfer_allreduce_residual_rmsnorm fallback paths"""

    def setUp(self):
        """Initialize test fixtures"""
        self.patcher_has_flashinfer = patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer")
        self.mock_has_flashinfer = self.patcher_has_flashinfer.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher_has_flashinfer.stop()

    def test_flashinfer_not_available_fallback(self):
        """Test lines 140-141: fallback when flashinfer not available"""
        self.mock_has_flashinfer.return_value = False

        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        fd_config = Mock()
        fd_config.parallel_config = Mock()
        fd_config.parallel_config.tensor_parallel_size = 2

        input_tensor = paddle.randn([128, 768])
        residual = paddle.randn([128, 768])
        weight = paddle.randn([768])

        norm_out, residual_out = flashinfer_allreduce_residual_rmsnorm(
            fd_config=fd_config,
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            eps=1e-6,
            max_token_num=2048,
        )

        # Should return None, None when flashinfer not available
        self.assertIsNone(norm_out)
        self.assertIsNone(residual_out)

    def test_single_gpu_fallback(self):
        """Test lines 146-147: fallback for single GPU"""
        self.mock_has_flashinfer.return_value = True

        with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"):
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm,
            )

            fd_config = Mock()
            fd_config.parallel_config = Mock()
            fd_config.parallel_config.tensor_parallel_size = 1

            input_tensor = paddle.randn([128, 768])
            residual = paddle.randn([128, 768])
            weight = paddle.randn([768])

            norm_out, residual_out = flashinfer_allreduce_residual_rmsnorm(
                fd_config=fd_config,
                input_tensor=input_tensor,
                residual=residual,
                weight=weight,
                eps=1e-6,
                max_token_num=2048,
            )

            # Should return None, None for single GPU
            self.assertIsNone(norm_out)
            self.assertIsNone(residual_out)

    def test_empty_tensor_handling(self):
        """Test line 166: empty tensor handling"""
        self.mock_has_flashinfer.return_value = True

        with (
            patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm") as mock_comm,
            patch(
                "fastdeploy.model_executor.layers.flashinfer_comm_fusion.ensure_workspace_initialized",
                return_value=True,
            ),
        ):
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm,
            )

            fd_config = Mock()
            fd_config.parallel_config = Mock()
            fd_config.parallel_config.tensor_parallel_size = 2

            # Empty tensor (0 tokens)
            input_tensor = paddle.zeros([0, 768])
            residual = paddle.zeros([0, 768])
            weight = paddle.randn([768])

            # Mock the trtllm_allreduce_fusion to not be called
            mock_comm.trtllm_allreduce_fusion = Mock()

            norm_out, residual_out = flashinfer_allreduce_residual_rmsnorm(
                fd_config=fd_config,
                input_tensor=input_tensor,
                residual=residual,
                weight=weight,
                eps=1e-6,
                max_token_num=2048,
            )

            # Should return empty tensors, not call flashinfer
            self.assertEqual(norm_out.shape[0], 0)
            self.assertEqual(residual_out.shape[0], 0)
            mock_comm.trtllm_allreduce_fusion.assert_not_called()


class TestCleanupFlashInferWorkspace(unittest.TestCase):
    """Test cleanup_flashinfer_workspace function"""

    def test_cleanup_workspace_function(self):
        """Test lines 211-212: cleanup function"""
        with patch("fastdeploy.model_executor.layers.flashinfer_comm_fusion._workspace_manager") as mock_manager:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                cleanup_flashinfer_workspace,
            )

            mock_manager.cleanup = Mock()

            cleanup_flashinfer_workspace()

            mock_manager.cleanup.assert_called_once()


class TestRMSNormProxyAllreduceFused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The outer test_run_distributed in test_trtllm_allreduce_rms_fusion.py
        # has already done paddle.set_device + init_parallel_env, so we don't
        # repeat that here. (unittest.main runs in the same process.)
        cls.tp_size = dist.get_world_size()
        cls.tp_rank = dist.get_rank()

    def _make_fd_config(self, enable_fusion: bool):
        """Mock fd_config with the minimal attributes RMSNorm.__init__ touches."""
        fd_config = Mock()
        fd_config.parallel_config = Mock()
        fd_config.parallel_config.tensor_parallel_size = self.tp_size
        fd_config.parallel_config.tensor_parallel_rank = self.tp_rank
        fd_config.parallel_config.tp_group = dist.get_group()
        fd_config.parallel_config.expert_parallel_size = 1
        fd_config.parallel_config.enable_flashinfer_allreduce_fusion = enable_fusion
        fd_config.parallel_config.use_sequence_parallel_moe = False
        fd_config.model_config = Mock()
        fd_config.model_config.moe_layer_start_index = -1
        fd_config.quant_config = None
        return fd_config

    def _build_rmsnorm(self, enable_fusion: bool, hidden_size: int, layer_id: int = 1):
        """Build a real RMSNorm whose enable_all_reduce_fusion resolves to
        `enable_fusion` (use post_attention_layernorm prefix to ensure the
        prefix-match in __init__ passes)."""
        from fastdeploy.model_executor.layers.normalization import RMSNorm

        fd_config = self._make_fd_config(enable_fusion=enable_fusion)
        norm = RMSNorm(
            fd_config=fd_config,
            hidden_size=hidden_size,
            eps=1e-6,
            prefix=f"model.layers.{layer_id}.post_attention_layernorm",
            layer_id=layer_id,
            dtype="bfloat16",
        )
        # Initialize weight to a known reproducible value (constant=1.0 by default).
        with paddle.no_grad():
            paddle.seed(2024)
            new_w = paddle.randn([hidden_size], dtype=paddle.bfloat16)
            dist.broadcast(new_w, src=0)
            norm.weight.set_value(new_w)
        return norm

    @staticmethod
    def _proxy_rmsnorm_fn(x, weight, eps):
        """Stand-in for phi rmsnorm used as proxy_rmsnorm — standard formula
        in fp32 to keep reference numerics clean."""
        x_fp32 = x.astype("float32")
        var = x_fp32.pow(2).mean(axis=-1, keepdim=True)
        out = x_fp32 * paddle.rsqrt(var + eps)
        out = out * weight.astype("float32")
        return out.astype(x.dtype)

    def _reference(self, x_partial, residual, weight, eps):
        """Manual: all_reduce(x_partial) + residual, then standard RMSNorm.
        Mirrors what proxy path WOULD produce after explicit allreduce+add."""
        x = x_partial.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        residual_out = x + residual
        norm_out = self._proxy_rmsnorm_fn(residual_out, weight, eps)
        return norm_out, residual_out

    def _make_inputs(self, token_num, hidden_size, seed=123):
        """Each rank gets a different x_partial (simulates RowParallelLinear's
        un-reduced output); residual is identical across ranks."""
        paddle.seed(seed + self.tp_rank * 7919)
        x_partial = paddle.randn([token_num, hidden_size], dtype=paddle.bfloat16) * 0.1
        paddle.seed(seed + 99)
        residual = paddle.randn([token_num, hidden_size], dtype=paddle.bfloat16)
        dist.broadcast(residual, src=0)
        return x_partial, residual

    def _assert_close_bf16(self, a, b, rtol=5e-2, atol=5e-2, msg=""):
        a32 = a.astype("float32").numpy()
        b32 = b.astype("float32").numpy()
        np.testing.assert_allclose(a32, b32, rtol=rtol, atol=atol, err_msg=msg)

    # ---------- Tests ----------

    def test_proxy_path_takes_fused_branch(self):
        """fusion=on, tp>1, shape<=2048, residual!=None
            -> proxy branch picks flashinfer_allreduce_residual_rmsnorm.
        Verify by patching the symbol and asserting it was called.
        """
        if self.tp_size < 2:
            self.skipTest("Requires tp_size >= 2")
        hidden = 512
        norm = self._build_rmsnorm(enable_fusion=True, hidden_size=hidden)
        self.assertTrue(norm.enable_all_reduce_fusion)
        x_partial, residual = self._make_inputs(token_num=64, hidden_size=hidden)

        # Patch within the normalization module's namespace.
        with patch(
            "fastdeploy.model_executor.layers.normalization.flashinfer_allreduce_residual_rmsnorm",
            wraps=__import__(
                "fastdeploy.model_executor.layers.normalization", fromlist=["flashinfer_allreduce_residual_rmsnorm"]
            ).flashinfer_allreduce_residual_rmsnorm,
        ) as spy:
            out, res = norm.forward(
                x_partial.clone(),
                residual_input=residual.clone(),
                proxy_rmsnorm=self._proxy_rmsnorm_fn,
            )
            spy.assert_called_once()

        # Numerics: must match reference (allreduce + add + std rmsnorm).
        ref_norm, ref_res = self._reference(x_partial, residual, norm.weight, norm.eps)
        self._assert_close_bf16(out, ref_norm, msg="proxy fused-branch norm output mismatch")
        self._assert_close_bf16(res, ref_res, msg="proxy fused-branch residual mismatch")

    def test_proxy_path_falls_back_when_fusion_disabled(self):
        """fusion=off -> proxy branch must call proxy_rmsnorm directly,
        no fused allreduce path used. Input is treated as already-reduced."""
        if self.tp_size < 2:
            self.skipTest("Requires tp_size >= 2")
        hidden = 512
        norm = self._build_rmsnorm(enable_fusion=False, hidden_size=hidden)
        self.assertFalse(norm.enable_all_reduce_fusion)

        # Each rank uses the SAME x (already-reduced) — that's the contract
        # when fusion is off (RowParallelLinear has done its own allreduce).
        paddle.seed(777)
        x = paddle.randn([64, hidden], dtype=paddle.bfloat16) * 0.1
        dist.broadcast(x, src=0)
        residual = paddle.randn([64, hidden], dtype=paddle.bfloat16)
        dist.broadcast(residual, src=0)

        proxy_called = {"n": 0}

        def proxy_spy(_x, _w, _eps):
            proxy_called["n"] += 1
            return self._proxy_rmsnorm_fn(_x, _w, _eps)

        with patch(
            "fastdeploy.model_executor.layers.normalization.flashinfer_allreduce_residual_rmsnorm"
        ) as fused_spy:
            out, res = norm.forward(
                x.clone(),
                residual_input=residual.clone(),
                proxy_rmsnorm=proxy_spy,
            )
            fused_spy.assert_not_called()

        self.assertEqual(proxy_called["n"], 1, "proxy_rmsnorm must be invoked exactly once")

        # Reference: x is already full -> just add + rmsnorm, no allreduce.
        residual_full = x + residual
        ref_norm = self._proxy_rmsnorm_fn(residual_full, norm.weight, norm.eps)
        self._assert_close_bf16(out, ref_norm, msg="fallback norm output mismatch")
        self._assert_close_bf16(res, residual_full, msg="fallback residual mismatch")

    def test_proxy_path_falls_back_when_token_too_large(self):
        """fusion=on but shape[0] > 2048 -> proxy branch must NOT call fused;
        in this regime upstream RowParallelLinear didn't skip its own
        all-reduce, so x is already full and proxy_rmsnorm is invoked directly."""
        if self.tp_size < 2:
            self.skipTest("Requires tp_size >= 2")
        hidden = 256
        norm = self._build_rmsnorm(enable_fusion=True, hidden_size=hidden)
        # shape[0] > 2048 forces use_allreduce_fused=False
        token_num = 2049
        paddle.seed(555)
        x = paddle.randn([token_num, hidden], dtype=paddle.bfloat16) * 0.1
        dist.broadcast(x, src=0)
        residual = paddle.randn([token_num, hidden], dtype=paddle.bfloat16)
        dist.broadcast(residual, src=0)

        with patch(
            "fastdeploy.model_executor.layers.normalization.flashinfer_allreduce_residual_rmsnorm"
        ) as fused_spy:
            out, res = norm.forward(
                x.clone(),
                residual_input=residual.clone(),
                proxy_rmsnorm=self._proxy_rmsnorm_fn,
            )
            fused_spy.assert_not_called()

        residual_full = x + residual
        ref_norm = self._proxy_rmsnorm_fn(residual_full, norm.weight, norm.eps)
        self._assert_close_bf16(out, ref_norm, msg="large-shape fallback norm mismatch")
        self._assert_close_bf16(res, residual_full, msg="large-shape fallback residual mismatch")


if __name__ == "__main__":
    """Run tests directly (called by subprocess after distributed launch)"""
    unittest.main(verbosity=2)
