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

from unittest.mock import Mock

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
)


class FakeModelConfig:
    def __init__(self):
        self.hidden_size = 768
        self.intermediate_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.rms_norm_eps = 1e-6
        self.tie_word_embeddings = True
        self.ori_vocab_size = 32000
        self.moe_layer_start_index = 8
        self.pretrained_config = Mock()
        self.pretrained_config.prefix_name = "test"
        self.num_key_value_heads = 1
        self.head_dim = 1
        self.is_quantized = False
        self.hidden_act = "relu"
        self.vocab_size = 32000
        self.hidden_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.max_position_embeddings = 512
        self.tie_word_embeddings = True
        self.model_format = "auto"


def get_default_test_fd_config():
    graph_opt_config = GraphOptimizationConfig(args={})
    parallel_config = ParallelConfig(args={})
    parallel_config.max_num_seqs = 1
    parallel_config.data_parallel_rank = 1
    cache_config = CacheConfig({})
    fd_config = FDConfig(
        graph_opt_config=graph_opt_config, parallel_config=parallel_config, cache_config=cache_config, test_mode=True
    )
    fd_config.model_config = FakeModelConfig()
    return fd_config
