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

import paddle

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
class TestCase1SubLayer1(paddle.nn.Layer):
    """Sub layer 1 of test case 1"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 forward pass"""

        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        return output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 Correct forward pass"""

        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        return output


class TestModel1(paddle.nn.Layer):
    """Tast Model"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()
        self.fd_config = fd_config

        self.sublayer1 = TestCase1SubLayer1(self.fd_config)
        sublayer1_copy = TestCase1SubLayer1(self.fd_config)
        self.sublayer2 = sublayer1_copy

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model forward pass"""
        # sublayer1 use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1(ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1)

        # sublayer2 use cuda graph
        sub_meta2 = ForwardMeta(
            input_ids=sublayer1_output, ids_remove_padding=sublayer1_output, step_use_cudagraph=True
        )
        sublayer2_output = self.sublayer2(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)

        return sublayer2_output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model Correct forward pass"""
        # sublayer1 not use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1.forward_correct(
            ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1
        )

        # sublayer2 not use cuda graph
        sub_meta2 = ForwardMeta(input_ids=sublayer1_output, ids_remove_padding=sublayer1_output)
        sublayer2_output = self.sublayer2.forward_correct(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)

        return sublayer2_output


class TestCUDAGrpahSpecDecode(unittest.TestCase):
    """
    Test CUDAGraph Memory change
    """

    def test_cuda_graph_spec_decode(self):
        """Run test case"""
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = True
        parallel_config = ParallelConfig(args={})
        parallel_config.max_num_seqs = 1
        cache_config = CacheConfig({})
        # Initialize cuda graph capture list
        graph_opt_config._set_cudagraph_sizes(max_num_seqs=parallel_config.max_num_seqs)
        graph_opt_config.init_with_cudagrpah_size(max_capture_size=parallel_config.max_num_seqs)
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            test_mode=True,
        )

        # Run Test Case1
        test_model1 = TestModel1(fd_config=fd_config)
        input_tensor1 = paddle.ones([1, 32768])
        forward_meta1 = ForwardMeta(input_ids=input_tensor1, ids_remove_padding=input_tensor1, step_use_cudagraph=True)

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
