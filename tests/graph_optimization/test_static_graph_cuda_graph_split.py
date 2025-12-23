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
import paddle.nn as nn

from fastdeploy.model_executor.graph_optimization.utils import sot_warmup_guard

paddle.set_flags({"FLAGS_cuda_graph_blacklist": "pd_op.matmul,pd_op.transpose"})


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
class Attention(nn.Layer):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=100, embedding_dim=32)
        self.qkv_proj = nn.Linear(32, 64)
        self.attn = nn.MultiHeadAttention(embed_dim=64, num_heads=1)
        self.o_proj = nn.Linear(64, 32)

    def forward(
        self,
        ids_remove_padding,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(forward_meta.ids_remove_padding)
        qkv_out = self.qkv_proj(hidden_states)
        attn_out = self.attn(qkv_out)
        output = self.o_proj(attn_out)

        return output

    def forward_dynamic(
        self,
        ids_remove_padding,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(forward_meta.ids_remove_padding)
        qkv_out = self.qkv_proj(hidden_states)
        attn_out = self.attn(qkv_out)
        output = self.o_proj(attn_out)

        return output


class TestModel(nn.Layer):
    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()
        self.model = Attention(fd_config)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def forward_correct(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model.forward_dynamic(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)


class TestStaticGraphCUDAGraphSplit(unittest.TestCase):

    def test(self):
        """Run test case"""
        # Set FastDeploy config
        graph_opt_config = GraphOptimizationConfig({"use_cudagraph": True, "graph_opt_level": 1})
        scheduler_config = SchedulerConfig({"max_num_seqs": 1})
        graph_opt_config._set_cudagraph_sizes(max_capture_size=scheduler_config.max_num_seqs)
        graph_opt_config.init_with_cudagrpah_size(max_capture_size=scheduler_config.max_num_seqs)
        cache_config = CacheConfig({})
        parallel_config = ParallelConfig(args={})
        model_config = Mock()
        model_config.max_model_len = 512
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            model_config=model_config,
            test_mode=True,
        )

        test_model1 = TestModel(fd_config=fd_config)
        x = paddle.randint(32, shape=[1, 8])
        forward_meta1 = ForwardMeta(ids_remove_padding=x, step_use_cudagraph=True)

        # Trigger Capture
        with sot_warmup_guard(True):
            _ = test_model1(x, forward_meta=forward_meta1)

        # Replay
        _ = test_model1(x, forward_meta=forward_meta1)
        output1 = test_model1(x, forward_meta=forward_meta1)

        # Correct output
        output1_correct = test_model1.forward_correct(x, forward_meta=forward_meta1)

        assert (output1 == output1_correct).all()


if __name__ == "__main__":
    unittest.main()
