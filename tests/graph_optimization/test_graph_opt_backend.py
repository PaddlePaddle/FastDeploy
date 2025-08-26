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

    def test_graph_opt_backend(self):
        """Run test case"""
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = False
        parallel_config = ParallelConfig(args={})
        parallel_config.max_num_seqs = 1
        cache_config = CacheConfig({})
        # Initialize cuda graph capture list
        graph_opt_config._set_cudagraph_sizes(max_num_seqs=parallel_config.max_num_seqs)
        graph_opt_config.init_with_cudagrpah_size(max_num_seqs=parallel_config.max_num_seqs)
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        input_np = np.ones([2, 4, 16], dtype="float32")
        # Run Numpy Baseline
        output_numpy = numpy_baseline(16, 32, input_np)

        input_tensor = paddle.ones([2, 4, 16], dtype="float32")
        # Run Test Dynamic Graph
        test_model_dynamic = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        forward_meta = ForwardMeta(input_ids=input_tensor, ids_remove_padding=input_tensor, step_use_cudagraph=True)
        output_dynamic = test_model_dynamic(ids_remove_padding=input_tensor, forward_meta=forward_meta)
        np.testing.assert_allclose(output_numpy, output_dynamic.numpy())

        # Run Test Static Graph
        graph_opt_config.graph_opt_level = 1
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        test_model_static = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        output_static = test_model_static(ids_remove_padding=input_tensor, forward_meta=forward_meta)
        np.testing.assert_allclose(output_numpy, output_static.numpy())

        # Run Test CINN
        graph_opt_config.graph_opt_level = 2
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        test_model_cinn = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        output_cinn = test_model_cinn(ids_remove_padding=input_tensor, forward_meta=forward_meta)
        np.testing.assert_allclose(output_numpy, output_cinn.numpy())

        graph_opt_config.use_cudagraph = True
        # Run Test Dynamic + CudaGraph
        graph_opt_config.graph_opt_level = 0
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        test_model_dynamic_cudagraph = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        output_dynamic_cudagraph = test_model_dynamic_cudagraph(
            ids_remove_padding=input_tensor, forward_meta=forward_meta
        )
        np.testing.assert_allclose(output_numpy, output_dynamic_cudagraph.numpy())

        # Run Test Static + CudaGraph
        graph_opt_config.graph_opt_level = 1
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        test_model_static_cudagraph = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        output_static_cudagraph = test_model_static_cudagraph(
            ids_remove_padding=input_tensor, forward_meta=forward_meta
        )
        np.testing.assert_allclose(output_numpy, output_static_cudagraph.numpy())

        # Run Test CINN + CudaGraph
        graph_opt_config.graph_opt_level = 2
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )
        test_model_cinn_cudagraph = TinyModel(fd_config=fd_config, d_model=16, d_hidden=32)
        output_cinn_cudagraph = test_model_cinn_cudagraph(ids_remove_padding=input_tensor, forward_meta=forward_meta)
        np.testing.assert_allclose(output_numpy, output_cinn_cudagraph.numpy())


if __name__ == "__main__":
    unittest.main()
