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
from unittest.mock import Mock

import numpy as np
import paddle
import paddle.nn as nn

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
    SchedulerConfig,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.graph_optimization.utils import sot_warmup_guard


@support_graph_optimization
class Attention(nn.Layer):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()
        paddle.seed(2024)
        self.embed_tokens = nn.Embedding(num_embeddings=100, embedding_dim=32)
        self.qkv_proj = nn.Linear(32, 64)
        self.attn = nn.MultiHeadAttention(embed_dim=64, num_heads=1)
        self.o_proj = nn.Linear(64, 32)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(forward_meta.ids_remove_padding)
        qkv_out = self.qkv_proj(hidden_states)
        attn_out = self.attn(qkv_out)
        output = self.o_proj(attn_out)

        return output

    def forward_dynamic(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(forward_meta.ids_remove_padding)
        qkv_out = self.qkv_proj(hidden_states)
        attn_out = self.attn(qkv_out)
        output = self.o_proj(attn_out)

        return output


class TestGraphOptBackend(unittest.TestCase):
    """
    Test graph_opt_backend
    """

    def setUp(self):
        paddle.seed(2025)

        """Set up test fixtures, compute baseline once for all tests"""
        # Setup common test data that will be reused across all tests
        self.input_shape = (4, 8)
        self.dtype = "int32"
        self.model_config = {}
        self.max_num_seqs = 4

        # Create baseline configuration (dynamic graph, no cudagraph)
        baseline_graph_opt_config = GraphOptimizationConfig(args={})
        baseline_graph_opt_config.use_cudagraph = False
        baseline_graph_opt_config.graph_opt_level = 0

        baseline_scheduler_config = SchedulerConfig(args={})
        baseline_scheduler_config.max_num_seqs = self.max_num_seqs

        baseline_cache_config = CacheConfig({})
        baseline_parallel_config = ParallelConfig(args={})
        model_config = Mock()
        model_config.max_model_len = 512
        self.baseline_fd_config = FDConfig(
            graph_opt_config=baseline_graph_opt_config,
            scheduler_config=baseline_scheduler_config,
            cache_config=baseline_cache_config,
            parallel_config=baseline_parallel_config,
            model_config=model_config,
            test_mode=True,
        )

        # Create input data
        self.input_tensor = paddle.randint(32, shape=self.input_shape, dtype=self.dtype)
        self.forward_meta = ForwardMeta(ids_remove_padding=self.input_tensor, step_use_cudagraph=True)

        # Compute baseline result once
        baseline_model = Attention(fd_config=self.baseline_fd_config, **self.model_config)
        self.baseline_result = baseline_model.forward_dynamic(
            ids_remove_padding=self.input_tensor, forward_meta=self.forward_meta
        ).numpy()

    def _setup_test_config(
        self,
        graph_opt_level=0,
        use_cudagraph=False,
    ):
        """Helper method: Setup test configuration for specific optimization mode

        Args:
            graph_opt_level (int): Graph optimization level (0: dynamic, 1: static, 2: cinn)
            use_cudagraph (bool): Whether to use cudagraph

        Returns:
            FDConfig: Configured FDConfig for testing
        """
        # Setup graph optimization config
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = use_cudagraph
        graph_opt_config.graph_opt_level = graph_opt_level

        # Setup parallel config
        scheduler_config = SchedulerConfig(args={})
        scheduler_config.max_num_seqs = self.max_num_seqs

        # Setup cache config
        cache_config = CacheConfig({})
        parallel_config = ParallelConfig(args={})
        model_config = Mock()
        model_config.max_model_len = 512

        # Create FD config
        return FDConfig(
            graph_opt_config=graph_opt_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            model_config=model_config,
            test_mode=True,
        )

    def _run_model_test(self, fd_config, test_name, compare_with_baseline=True):
        """Helper method: Run model test and validate results

        Args:
            fd_config: FastDeploy configuration
            test_name (str): Test name for error reporting
            compare_with_baseline (bool): Whether to compare with baseline result
        """
        test_model = Attention(fd_config=fd_config, **self.model_config)

        with sot_warmup_guard(True):
            _ = test_model(ids_remove_padding=self.input_tensor, forward_meta=self.forward_meta)

        # Run model test
        output = test_model(ids_remove_padding=self.input_tensor, forward_meta=self.forward_meta)

        # Validate results if comparison is requested
        if compare_with_baseline:
            np.testing.assert_allclose(
                self.baseline_result,
                output.numpy(),
                err_msg=f"Test {test_name} failed: output mismatch",
                atol=1e-4,  # for CINN
                rtol=1e-2,
            )

    def tearDown(self):
        paddle.jit.sot.opcode_translator.executor.executor_cache.OpcodeExecutorCache().clear()

    def test_dynamic_graph(self):
        """Test dynamic graph mode"""
        fd_config = self._setup_test_config(graph_opt_level=0, use_cudagraph=False)
        self._run_model_test(fd_config, "dynamic_graph", compare_with_baseline=False)

    def test_static_graph(self):
        """Test static graph mode"""
        fd_config = self._setup_test_config(graph_opt_level=1, use_cudagraph=False)
        self._run_model_test(fd_config, "static_graph")

    def test_cinn_graph(self):
        """Test CINN optimization mode"""
        # Note: CINN is not opened yet
        fd_config = self._setup_test_config(graph_opt_level=2, use_cudagraph=False)
        self._run_model_test(fd_config, "cinn_graph")

    def test_dynamic_graph_with_cudagraph(self):
        """Test dynamic graph + CudaGraph mode"""
        fd_config = self._setup_test_config(graph_opt_level=0, use_cudagraph=True)
        self._run_model_test(fd_config, "dynamic_graph_cudagraph")

    def test_static_graph_with_cudagraph(self):
        """Test static graph + CudaGraph mode"""
        fd_config = self._setup_test_config(graph_opt_level=1, use_cudagraph=True)
        self._run_model_test(fd_config, "static_graph_cudagraph")

    def test_cinn_graph_with_cudagraph(self):
        """Test CINN + CudaGraph mode"""
        # Note: CINN is not opened yet
        fd_config = self._setup_test_config(graph_opt_level=2, use_cudagraph=True)
        self._run_model_test(fd_config, "cinn_graph_cudagraph")


if __name__ == "__main__":
    unittest.main()
