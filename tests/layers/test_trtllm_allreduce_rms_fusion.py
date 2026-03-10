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

import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import paddle


def test_run_distributed():
    """Launch multi-GPU distributed test via paddle.distributed.launch as subprocess"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_script = os.path.join(current_dir, "trtllm_allreduce_rms_fusion.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        "--gpus",
        "0,1",
        run_script,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(timeout=400)
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return_code = -1
    assert return_code in (0, 250), f"Process exited with code {return_code}"


test_run_distributed()


class TestFlashInferWorkspaceManagerEdgeCases(unittest.TestCase):
    """Test FlashInferWorkspaceManager edge cases and fallback paths"""

    def setUp(self):
        """Initialize test fixtures"""
        # Patch before importing to test fallback paths
        self.patcher_has_flashinfer = patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer"
        )
        self.mock_has_flashinfer = self.patcher_has_flashinfer.start()

    def tearDown(self):
        """Clean up patches"""
        self.patcher_has_flashinfer.stop()

    def test_initialization_early_return_when_already_initialized(self):
        """Test line 47: early return when already initialized with same world_size"""
        # Patch _flashinfer_comm to be available
        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"
        ) as mock_comm:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
            )

            manager = FlashInferWorkspaceManager()

            # First initialization
            manager.initialized = True
            manager.world_size = 2

            # Mock the comm functions
            mock_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion = Mock(
                return_value=(Mock(), Mock())
            )

            # Second initialization with same world_size - should return early
            manager.initialize(
                world_size=2,
                rank=0,
                max_token_num=2048,
                hidden_dim=4096,
            )

    def test_initialization_warning_when_comm_none(self):
        """Test lines 50-51: warning when _flashinfer_comm is None"""
        # Patch to ensure _flashinfer_comm is None
        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm",
            None,
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
        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"
        ) as mock_comm:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
            )

            manager = FlashInferWorkspaceManager()
            manager.initialized = True
            manager.ipc_handles = Mock()
            manager.workspace_tensor = Mock()

            # Mock the destroy function to raise exception
            mock_comm.trtllm_destroy_ipc_workspace_for_all_reduce = Mock(
                side_effect=RuntimeError("Cleanup error")
            )

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
        self.patcher_has_flashinfer = patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer"
        )
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
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm",
            None,
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

        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"
        ):
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
        self.patcher_has_flashinfer = patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion.has_flashinfer"
        )
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

        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"
        ):
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

        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._flashinfer_comm"
        ) as mock_comm, patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion.ensure_workspace_initialized",
            return_value=True,
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


class TestFakeFlashInferFunction(unittest.TestCase):
    """Test fake_flashinfer_allreduce_residual_rmsnorm function"""

    def test_fake_function_basic(self):
        """Test lines 204-206: fake function basic functionality"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            fake_flashinfer_allreduce_residual_rmsnorm,
        )

        input_tensor = paddle.randn([128, 768])
        residual = paddle.randn([128, 768])
        weight = paddle.randn([768])

        norm_out, residual_out = fake_flashinfer_allreduce_residual_rmsnorm(
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            eps=1e-6,
            max_token_num=16384,
            use_oneshot=None,
            trigger_completion_at_end=False,
            fp32_acc=False,
        )

        # Should return empty-like tensors
        self.assertEqual(norm_out.shape, input_tensor.shape)
        self.assertEqual(residual_out.shape, residual.shape)


class TestCleanupFlashInferWorkspace(unittest.TestCase):
    """Test cleanup_flashinfer_workspace function"""

    def test_cleanup_workspace_function(self):
        """Test lines 211-212: cleanup function"""
        with patch(
            "fastdeploy.model_executor.layers.flashinfer_comm_fusion._workspace_manager"
        ) as mock_manager:
            from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
                cleanup_flashinfer_workspace,
            )

            mock_manager.cleanup = Mock()

            cleanup_flashinfer_workspace()

            mock_manager.cleanup.assert_called_once()
