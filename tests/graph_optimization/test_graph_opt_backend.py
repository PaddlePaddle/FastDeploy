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

import numpy as np
import paddle
from paddle.nn import functional as F

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)


@support_graph_optimization
class TinyModel(paddle.nn.Layer):
    """Test Model"""

    def __init__(self, fd_config: FDConfig, d_model: int, d_hidden: int):
        super().__init__()
        self.fd_config = fd_config

        self.W1 = paddle.ones([d_model, d_hidden]).astype("float32")
        self.b1 = paddle.ones([d_hidden]).astype("float32")
        self.W2 = paddle.ones([d_hidden, d_model]).astype("float32")
        self.b2 = paddle.ones([d_model]).astype("float32")

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model forward pass"""
        h = F.relu(F.linear(forward_meta.input_ids, self.W1, self.b1))
        return forward_meta.input_ids + F.linear(h, self.W2, self.b2)


def numpy_baseline(d_model: int, d_hidden: int, x: np.ndarray):

    W1 = np.ones((d_model, d_hidden), dtype="float32")
    b1 = np.ones(d_hidden, dtype="float32")
    W2 = np.ones((d_hidden, d_model), dtype="float32")
    b2 = np.ones(d_model, dtype="float32")

    h = np.maximum(0, x @ W1 + b1)
    return x + (h @ W2 + b2)


class TestGrpahOptBackend(unittest.TestCase):
    """
    Test graph_opt_backend
    """

    def _setup_common_test_components(
        self,
        graph_opt_level=0,
        use_cudagraph=False,
        input_shape=(2, 4, 16),
        dtype="float32",
        model_config=None,
        max_num_seqs=1,
    ):
        """Helper method: Setup common test components

        Args:
            graph_opt_level (int): Graph optimization level (0: dynamic, 1: static, 2: cinn)
            use_cudagraph (bool): Whether to use cudagraph
            input_shape (tuple): Input data shape (batch_size, seq_len, d_model)
            dtype (str): Data type
            model_config (dict): Model configuration parameters, default: {"d_model": 16, "d_hidden": 32}
            max_num_seqs (int): Maximum number of sequences

        Returns:
            tuple: (fd_config, input_tensor, forward_meta, model_config)
        """
        # Default model configuration
        if model_config is None:
            model_config = {"d_model": 16, "d_hidden": 32}

        # Setup graph optimization config
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = use_cudagraph
        graph_opt_config.graph_opt_level = graph_opt_level

        # Setup parallel config
        parallel_config = ParallelConfig(args={})
        parallel_config.max_num_seqs = max_num_seqs

        # Setup cache config
        cache_config = CacheConfig({})

        # Initialize cuda graph capture list
        graph_opt_config._set_cudagraph_sizes(max_num_seqs=parallel_config.max_num_seqs)
        graph_opt_config.init_with_cudagrpah_size(max_num_seqs=parallel_config.max_num_seqs)

        # Create FD config
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )

        # Create input data
        input_tensor = paddle.ones(input_shape, dtype=dtype)

        # Create forward_meta
        forward_meta = ForwardMeta(input_ids=input_tensor, ids_remove_padding=input_tensor, step_use_cudagraph=True)

        return fd_config, input_tensor, forward_meta, model_config

    def _run_model_test(
        self, fd_config, input_tensor, forward_meta, model_config, test_name, model_class=None, baseline_func=None
    ):
        """Helper method: Run model test and validate results

        Args:
            fd_config: FastDeploy configuration
            input_tensor: Input tensor
            forward_meta: Forward meta object
            model_config (dict): Model configuration parameters
            test_name (str): Test name for error reporting
            model_class: Model class, default uses TinyModel
            baseline_func: Baseline function, default uses numpy_baseline
        """
        if model_class is None:
            model_class = TinyModel
        if baseline_func is None:
            baseline_func = numpy_baseline

        # Calculate baseline results
        input_np = input_tensor.numpy()
        output_numpy = baseline_func(**model_config, x=input_np)

        # Run model test
        test_model = model_class(fd_config=fd_config, **model_config)
        output = test_model(ids_remove_padding=input_tensor, forward_meta=forward_meta)

        # Validate results
        np.testing.assert_allclose(output_numpy, output.numpy(), err_msg=f"Test {test_name} failed: output mismatch")

    def test_dynamic_graph(self):
        """Test dynamic graph mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=0, use_cudagraph=False
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "dynamic_graph")

    def test_static_graph(self):
        """Test static graph mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=1, use_cudagraph=False
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "static_graph")

    def test_cinn_graph(self):
        """Test CINN optimization mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=2, use_cudagraph=False
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "cinn_graph")

    def test_dynamic_graph_with_cudagraph(self):
        """Test dynamic graph + CudaGraph mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=0, use_cudagraph=True
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "dynamic_graph_cudagraph")

    def test_static_graph_with_cudagraph(self):
        """Test static graph + CudaGraph mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=1, use_cudagraph=True
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "static_graph_cudagraph")

    def test_cinn_graph_with_cudagraph(self):
        """Test CINN + CudaGraph mode"""
        fd_config, input_tensor, forward_meta, model_config = self._setup_common_test_components(
            graph_opt_level=2, use_cudagraph=True
        )
        self._run_model_test(fd_config, input_tensor, forward_meta, model_config, "cinn_graph_cudagraph")


if __name__ == "__main__":
    unittest.main()
