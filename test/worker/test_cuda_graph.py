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

from fastdeploy.config import GraphOptimizationConfig, LLMConfig
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_opt


@support_graph_opt
class TestModel(paddle.nn.Layer):
    """ Tast Model """

    def __init__(self, llm_config: LLMConfig, **kwargs):
        self.llm_config = llm_config

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        """前向传播"""
        input_ids: paddle.Tensor = kwargs["input_ids"]
        return input_ids + input_ids


if __name__ == '__main__':
    graph_opt_config = GraphOptimizationConfig()
    graph_opt_config.use_cudagraph = True
    graph_opt_config.cudagraph_capture_sizes = [1, 4]
    llm_config = LLMConfig(graph_opt_config=graph_opt_config)
    model = TestModel(llm_config=llm_config)

    output = model(input_ids=paddle.zeros([1, 8]))
    print(output)
    output = model(input_ids=paddle.ones([1, 8]))
    print(output)
    output = model(input_ids=paddle.zeros([4, 9]))
    print(output)
    output = model(input_ids=paddle.ones([4, 9]))
    print(output)
