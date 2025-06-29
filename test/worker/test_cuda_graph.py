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
import paddle

from fastdeploy.config import FDConfig, GraphOptimizationConfig
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_optimization
from fastdeploy.worker.forward_meta import ForwardMeta


@support_graph_optimization
class TestCase1SubLayer1(paddle.nn.Layer):
    """ Sub layer 1 of test case 1 """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, _, forward_meta: ForwardMeta):
        """ Sub layer1 forward pass """

        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        print(" SubLayer1 Output: {output}")
        return output


class TestCase1SubLayer2(paddle.nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, _, forward_meta: ForwardMeta):
        """ Sub layer2 forward pass """
        x = paddle.ones_like(forward_meta.input_ids)
        y = paddle.ones_like(forward_meta.input_ids)
        output = x + y
        print(" SubLayer2 Output: {output}")
        return output


@support_graph_optimization
class TestCase1SubLayer3(paddle.nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, _, forward_meta: ForwardMeta):
        """ Sub layer3 forward pass """
        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        print(" SubLayer3 Output: {output}")
        return output


class TestModel1(paddle.nn.Layer):
    """ Tast Model """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()
        self.fd_config = fd_config

    def forward(self, _, forward_meta: ForwardMeta):
        """ Test model for ward pass """
        self.sublayer1 = TestCase1SubLayer1(self.fd_config)
        self.sublayer2 = TestCase1SubLayer2(self.fd_config)
        self.sublayer3 = TestCase1SubLayer3(self.fd_config)

        # sublayer1 use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1(_=None, forward_meta=sub_meta1)

        # sublayer2 not use cuda garph
        sub_meta2 = ForwardMeta(input_ids=sublayer1_output)
        sublayer2_output = self.sublayer2(_=None, forward_meta=sub_meta2)

        # sublayer3 use cuda graph
        sub_meta3 = ForwardMeta(input_ids=sublayer2_output)
        sublayer3_output = self.sublayer3(_=None, forward_meta=sub_meta3)

        return sublayer3_output


@support_graph_optimization
class TestModel2(paddle.nn.Layer):
    """ Tast Model """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, _, forward_meta: ForwardMeta):
        """ Test model for ward pass """
        return forward_meta.input_ids + forward_meta.input_ids


def run_test_case():
    """ Run test case """
    # Set llm config1
    graph_opt_config = GraphOptimizationConfig()
    graph_opt_config.use_cudagraph = True
    graph_opt_config.cudagraph_capture_sizes = [1]
    fd_config = FDConfig(graph_opt_config=graph_opt_config)

    # Run Test Case1
    test_model1 = TestModel1(fd_config=fd_config)
    input_tensor1 = paddle.zeros([1, 8])
    forward_meta1 = ForwardMeta(input_ids=input_tensor1)
    output1 = test_model1(_=None, forward_meta=forward_meta1)
    print(output1)

    # Run Test Case2
    test_model2 = TestModel2(fd_config=fd_config)
    input_tensor2 = paddle.zeros([1, 8])
    forward_meta2 = ForwardMeta(input_ids=input_tensor2)
    output2 = test_model2(_=None, forward_meta=forward_meta2)
    print(output2)


if __name__ == '__main__':
    run_test_case()
