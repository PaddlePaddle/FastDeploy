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

from __future__ import annotations

import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.pooler import DispatchPooler, Pooler
from fastdeploy.model_executor.utils import process_weights_before_loading

from .interfaces_base import default_pooling_type
from .model_base import ModelCategory, ModelRegistry
from .qwen2 import Qwen2ForCausalLM, Qwen2Model


class Qwen2RewardBaseModel(nn.Layer):
    """
    Qwen2RewardBaseModel
    """

    is_pooling_model = True
    pooler: Pooler

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.model = Qwen2Model(fd_config=fd_config)
        self.head_dtype = paddle.float32

        self.score = nn.Sequential(
            ColumnParallelLinear(
                fd_config=fd_config,
                input_size=fd_config.model_config.hidden_size,
                output_size=fd_config.model_config.hidden_size,
                skip_quant=True,
                weight_dtype=self.head_dtype,
                with_bias=True,
            ),
            nn.ReLU(),
            RowParallelLinear(
                fd_config=fd_config,
                input_size=fd_config.model_config.hidden_size,
                output_size=fd_config.model_config.num_labels,
                skip_quant=True,
                weight_dtype=self.head_dtype,
                with_bias=True,
            ),
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        hidden_states = hidden_states.to(self.head_dtype)
        logits = self.score(hidden_states)
        return logits


@ModelRegistry.register_model_class(
    architecture="Qwen2ForProcessRewardModel",
    module_name="qwen2_rm",
    category=[ModelCategory.REWARD],
    primary_use=ModelCategory.REWARD,
)
@default_pooling_type("STEP")
class Qwen2ForProcessRewardModel(Qwen2RewardBaseModel):

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        fd_config.model_config.num_labels = 2
        super().__init__(fd_config=fd_config)

        pooler_config = fd_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler({"encode": Pooler.for_encode(pooler_config)})

        self.process_weights_before_loading_fn = process_weights_before_loading(skip_prefixes=["lm_head"])

    @classmethod
    def name(self):
        """ """
        return "Qwen2ForProcessRewardModel"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        # Filter out lm_head weights of Qwen2ForCausalLM
        Qwen2ForCausalLM.load_weights(self, weights_iterator)
