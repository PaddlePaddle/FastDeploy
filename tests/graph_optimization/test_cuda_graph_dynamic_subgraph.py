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

import paddle

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


@support_graph_optimization
class TestCase1SubLayer1(paddle.nn.Layer):
    """Sub layer 1 of test case 1"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 forward pass"""

        output = paddle.add(forward_meta.ids_remove_padding, forward_meta.ids_remove_padding)
        return output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 Correct forward pass"""
        output = paddle.add(forward_meta.ids_remove_padding, forward_meta.ids_remove_padding)
        return output


class TestCase1SubLayer2(paddle.nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer2 forward pass"""
        x = forward_meta.ids_remove_padding
        y = forward_meta.ids_remove_padding
        output = x + y
        return output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer2 Correct forward pass"""
        x = forward_meta.ids_remove_padding
        y = forward_meta.ids_remove_padding
        output = x + y
        return output


@support_graph_optimization
class TestCase1SubLayer3(paddle.nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer3 forward pass"""
        output = paddle.matmul(forward_meta.ids_remove_padding, forward_meta.ids_remove_padding)
        return output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer3 Correct forward pass"""
        output = paddle.matmul(forward_meta.ids_remove_padding, forward_meta.ids_remove_padding)
        return output


class TestModel1(paddle.nn.Layer):
    """Test Model"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()
        self.fd_config = fd_config

        self.sublayer1 = TestCase1SubLayer1(self.fd_config)
        self.sublayer2 = TestCase1SubLayer2(self.fd_config)  # Attention
        self.sublayer3 = TestCase1SubLayer3(self.fd_config)

        self.sublayer2_output_buffer = paddle.zeros([1])

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model for ward pass"""
        # sublayer1 use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1(ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1)

        # sublayer2 not use cuda graph
        sub_meta2 = ForwardMeta(ids_remove_padding=sublayer1_output)
        sublayer2_output = self.sublayer2(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)
        self.sublayer2_output_buffer.copy_(sublayer2_output, False)

        # sublayer3 use cuda graph
        sub_meta3 = ForwardMeta(
            ids_remove_padding=self.sublayer2_output_buffer,
            step_use_cudagraph=True,
        )
        sublayer3_output = self.sublayer3(ids_remove_padding=self.sublayer2_output_buffer, forward_meta=sub_meta3)

        return sublayer3_output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model for ward pass"""

        # sublayer1 not use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1.forward_correct(
            ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1
        )

        # sublayer2 not use cuda graph
        sub_meta2 = ForwardMeta(ids_remove_padding=sublayer1_output)
        sublayer2_output = self.sublayer2.forward_correct(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)

        # sublayer3 not use cuda graph
        sub_meta3 = ForwardMeta(ids_remove_padding=sublayer2_output)
        sublayer3_output = self.sublayer3.forward_correct(ids_remove_padding=sublayer2_output, forward_meta=sub_meta3)

        return sublayer3_output


class TestCUDAGrpahSubgraph(unittest.TestCase):
    """
    Test CUDAGraph Memory change
    """

    def test_cuda_graph_subgraph(self):
        """Run test case"""
        # Set FastDeploy config
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = True
        scheduler_config = SchedulerConfig(args={})
        scheduler_config.max_num_seqs = 8
        cache_config = CacheConfig({})
        parallel_config = ParallelConfig(args={})
        model_config = Mock()
        model_config.max_model_len = 512
        # Initialize cuda graph capture list
        graph_opt_config._set_cudagraph_sizes(max_capture_size=scheduler_config.max_num_seqs)
        graph_opt_config.init_with_cudagrpah_size(max_capture_size=scheduler_config.max_num_seqs)
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            model_config=model_config,
            test_mode=True,
        )

        # Run Test Case1
        test_model1 = TestModel1(fd_config=fd_config)
        input_tensor1 = paddle.ones([8])
        forward_meta1 = ForwardMeta(ids_remove_padding=input_tensor1, step_use_cudagraph=True)

        # Trigger Capture
        _ = test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)

        # Replay
        _ = test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)
        output1 = test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)

        # Correct output
        output1_correct = test_model1.forward_correct(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)

        assert (output1 == output1_correct).all()


if __name__ == "__main__":
    unittest.main()
