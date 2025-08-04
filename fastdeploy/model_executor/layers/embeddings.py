"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.utils import set_weight_attrs

from .utils import get_tensor


class VocabParallelEmbedding(nn.Layer):
    """
    VocabParallelEmbedding Layer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        num_embeddings: int,
        embedding_dim: int = 768,
        params_dtype: str = "bfloat16",
        prefix="",
    ) -> None:
        """
        Initialize the VocabParallelEmbedding layer for the model.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings (int)  : vocabulary size.
            embedding_dim (int) : size of hidden state.
            params_dtype  (str) : data type of parameters.
            prefix (str): The name of current layer. Defaults to "".
        """
        super().__init__()
        self.fd_config = fd_config
        hcg = fleet.get_hybrid_communicate_group()
        self.mp_rank: int = hcg.get_model_parallel_rank()
        self.column_cut = False
        self.world_size: int = hcg.get_model_parallel_world_size()
        self.ring_id: int = hcg.get_model_parallel_group().id
        self.use_ep: bool = fd_config.parallel_config.use_ep
        self.hidden_dropout_prob: float = fd_config.model_config.hidden_dropout_prob
        self.initializer_range: float = fd_config.model_config.initializer_range
        self.max_position_embeddings: int = fd_config.model_config.max_position_embeddings
        self.tie_word_embeddings: bool = fd_config.model_config.tie_word_embeddings
        self.params_dtype: str = params_dtype

        if self.use_ep:
            self.embeddings = nn.Embedding(
                num_embeddings,
                embedding_dim,
            )
        else:
            if not self.column_cut:
                self.embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                    num_embeddings,
                    embedding_dim,
                    mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range),
                    ),
                )
                set_weight_attrs(self.embeddings.weight, {"output_dim": False})
            else:
                # column cut embedding
                self.embeddings = nn.Embedding(
                    num_embeddings,
                    embedding_dim // self.world_size,
                )

                self.embeddings.weight.is_distributed = True
                self.embeddings.weight.split_axis = 1
                set_weight_attrs(self.embeddings.weight, {"output_dim": True})

        self.prefix = prefix
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def load_state_dict(self, state_dict: Dict[str, paddle.Tensor | np.ndarray]):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if self.tie_word_embeddings:
            self.embeddings.weight.set_value(
                get_tensor(state_dict[self.prefix + ".weight"]).astype(paddle.get_default_dtype())
            )
        else:
            self.embeddings.weight.set_value(
                get_tensor(state_dict.pop(self.prefix + ".weight")).astype(paddle.get_default_dtype())
            )

    def forward(self, ids_remove_padding=None) -> paddle.Tensor:
        """
        Defines the forward computation of the layer.

        Args:
            ids_remove_padding (Tensor, optional): Tensor of token IDs, with padding removed.
                If None, no input is provided.

        Returns:
            Tensor: Embedded tensor representation of the input IDs.
        """
        if self.use_ep:
            input_embedings = self.embeddings(ids_remove_padding)
        else:
            if self.column_cut:
                input_embedings = self.embeddings(ids_remove_padding)
                inputs_embeds_temp = []
                paddle.distributed.all_gather(
                    inputs_embeds_temp,
                    input_embedings,
                    group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                    sync_op=True,
                )
                input_embedings = paddle.concat(inputs_embeds_temp, -1)
            else:
                input_embedings = self.embeddings(ids_remove_padding)

        return input_embedings
