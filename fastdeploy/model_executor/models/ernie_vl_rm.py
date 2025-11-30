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

from typing import Optional

import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.pooler import DispatchPooler, Pooler
from fastdeploy.model_executor.utils import process_weights_before_loading

from .ernie4_5_vl.ernie4_5_vl_moe import (
    Ernie4_5_VLModel,
    Ernie4_5_VLMoeForConditionalGeneration,
)
from .interfaces_base import default_pooling_type
from .model_base import ModelCategory, ModelRegistry


class Ernie4_5_VLMoeRewardBaseModel(nn.Layer):
    """
    Ernie4_5_VLMoeRewardBaseModel
    """

    is_pooling_model = True
    pooler: Pooler

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        # ----------- vision model ------------
        self.vision_model = Ernie4_5_VLMoeForConditionalGeneration._init_vision_model(self, fd_config.model_config)
        # -----------  resampler_model ------------
        self.resampler_model = Ernie4_5_VLMoeForConditionalGeneration._init_resampler_model_model(
            self, fd_config.model_config
        )
        self.ernie = Ernie4_5_VLModel(fd_config=fd_config)
        self.head_dtype = paddle.bfloat16

        # Persistent buffers for CUDA graphs.
        if fd_config.graph_opt_config.use_cudagraph:
            self._decoder_input_embeddings = paddle.zeros(
                [fd_config.graph_opt_config.max_capture_size, fd_config.model_config.hidden_size],
                dtype=fd_config.model_config.dtype,
            )

        self.rm_head = nn.Sequential(
            (
                "up_gate_proj",
                MergedColumnParallelLinear(
                    fd_config=fd_config,
                    prefix="",
                    input_size=fd_config.model_config.hidden_size,
                    output_size=fd_config.model_config.hidden_size * 2,
                    with_bias=False,
                ),
            ),
            ("act_fn", SiluAndMul(fd_config=fd_config, bias=None, act_method=fd_config.model_config.hidden_act)),
            (
                "down_proj",
                RowParallelLinear(
                    fd_config=fd_config,
                    input_size=fd_config.model_config.hidden_size,
                    output_size=fd_config.model_config.num_labels,
                    skip_quant=True,
                    weight_dtype=self.head_dtype,
                    with_bias=False,
                ),
            ),
        )

    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_token_num: int,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_embeddings = self.ernie.get_input_embeddings(ids_remove_padding=ids_remove_padding)
        if image_token_num > 0:
            input_embeddings[ids_remove_padding == self.ernie.im_patch_id] = image_features.cast(self.ernie._dtype)
        return input_embeddings

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ):
        vl_moe_meta = self.ernie.prepare_vl_moe_meta(ids_remove_padding=ids_remove_padding)
        input_embeddings = self.get_input_embeddings(
            ids_remove_padding=ids_remove_padding,
            image_features=image_features,
            image_token_num=vl_moe_meta.image_token_num.item(),
        )

        if forward_meta.step_use_cudagraph:
            self._decoder_input_embeddings.copy_(input_embeddings, False)
            input_embeddings = self._decoder_input_embeddings

        hidden_states = self.ernie(
            input_embeddings=input_embeddings,
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            vl_moe_meta=vl_moe_meta,
        )
        hidden_states = hidden_states.to(self.head_dtype)
        logits = self.rm_head(hidden_states)
        return logits


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_VLMoeForProcessRewardModel",
    module_name="ernie_vl_rm",
    category=[ModelCategory.REWARD],
    primary_use=ModelCategory.REWARD,
)
@default_pooling_type("ALL")
class Ernie4_5_VLMoeForProcessRewardModel(Ernie4_5_VLMoeRewardBaseModel):

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        fd_config.model_config.num_labels = 1
        super().__init__(fd_config=fd_config)
        self.tie_word_embeddings = False

        pooler_config = fd_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler({"encode": Pooler.for_encode(pooler_config)})

        self.process_weights_before_loading_fn = process_weights_before_loading(skip_prefixes=["lm_head"])

    @classmethod
    def name(self):
        """ """
        return "Ernie4_5_VLMoeForProcessRewardModel"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        # Filter out lm_head weights of Ernie4_5_VLMoeForConditionalGeneration
        Ernie4_5_VLMoeForConditionalGeneration.load_weights(self, weights_iterator)
