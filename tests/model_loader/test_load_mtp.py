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

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import paddle
import paddle.distributed.fleet as fleet

from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.models.ernie4_5_mtp import Ernie4_5_MTPForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.utils import get_default_test_fd_config

strategy = fleet.DistributedStrategy()
fleet.init(strategy=strategy)


class TestErnie4_5_MTPLoadWeights(unittest.TestCase):
    def setUp(self):
        self.fd_config = get_default_test_fd_config()
        self.fd_config.speculative_config = Mock()
        self.fd_config.speculative_config.sharing_model = Mock()
        self.fd_config.speculative_config.sharing_model.ernie = Mock()
        self.fd_config.parallel_config.tp_group = None
        self.fd_config.speculative_config.sharing_model.ernie.embed_tokens = VocabParallelEmbedding(
            fd_config=self.fd_config,
            num_embeddings=self.fd_config.model_config.vocab_size,
            embedding_dim=self.fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=("embed_tokens"),
        )
        self.fd_config.speculative_config.sharing_model.ernie.lm_head = Mock()
        self.model = Ernie4_5_MTPForCausalLM(self.fd_config)

    def test_load_weights_normal_case(self):
        weights_iterator = [
            ("ernie.embed_tokens.weight", paddle.rand([32000, 768], dtype="float32")),
            ("ernie.mtp_block.0.self_attn.qkv_proj.weight", paddle.rand([768, 768 * 3], dtype="float32")),
        ]
        for k, v in self.model.named_parameters():
            print("{}".format(k))

        self.model.load_weights(iter(weights_iterator))

        self.assertTrue(np.allclose(self.model.ernie.embed_tokens.embeddings.weight.numpy(), weights_iterator[0][1]))

    def test_load_weights_with_unexpected_keys(self):
        weights_iterator = [
            ("unknown_key", paddle.rand([10, 10], dtype="float32")),
            ("ernie.embed_tokens.weight", paddle.rand([32000, 768], dtype="float32")),
        ]

        self.model.load_weights(iter(weights_iterator))

        self.assertTrue(np.allclose(self.model.ernie.embed_tokens.embeddings.weight.numpy(), weights_iterator[1][1]))

    def test_load_weights_empty_iterator(self):
        weights_iterator = []

        self.model.load_weights(iter(weights_iterator))


if __name__ == "__main__":
    unittest.main()
