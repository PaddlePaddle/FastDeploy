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

import paddle
from paddle import nn
from paddle.distributed import fleet

from .utils import get_tensor


class VocabParallelEmbedding(nn.Layer):
    """
    VocabParallelEmbedding Layer
    """

    def __init__(
        self,
        fd_config,
        num_embeddings,
        embedding_dim=768,
        params_dtype="bfloat16",
        prefix="",
    ):
        """
        Initialize the VocabParallelEmbedding layer for the model.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings : vocabulary size.
            embedding_dim : size of hidden state.
            params_dtype : data type of parameters.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
        """
        super().__init__()
        self.fd_config = fd_config
        hcg = fleet.get_hybrid_communicate_group()
        self.mp_rank = hcg.get_model_parallel_rank()
        self.column_cut = fd_config.parallel_config.column_cut
        self.world_size = hcg.get_model_parallel_world_size()
        self.ring_id = hcg.get_model_parallel_group().id
        self.use_rope = fd_config.model_config.use_rope
        self.rope_head_dim = fd_config.model_config.rope_head_dim
        self.use_ep = fd_config.parallel_config.use_ep
        self.hidden_dropout_prob = fd_config.model_config.hidden_dropout_prob
        self.initializer_range = fd_config.model_config.initializer_range
        self.sequence_parallel = fd_config.parallel_config.sequence_parallel
        self.max_position_embeddings = fd_config.model_config.max_position_embeddings
        self.freeze_embedding = fd_config.model_config.freeze_embedding
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

        if self.use_ep:
            self.word_embeddings = nn.Embedding(
                num_embeddings,
                embedding_dim,
            )
        else:
            if not self.column_cut:
                self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                    num_embeddings,
                    embedding_dim,
                    mp_group=fleet.get_hybrid_communicate_group().
                    get_model_parallel_group(),
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range), ),
                )
            else:
                # column cut embedding
                self.word_embeddings = nn.Embedding(
                    num_embeddings,
                    embedding_dim // self.world_size,
                )

                self.word_embeddings.weight.is_distributed = True
                self.word_embeddings.weight.split_axis = 1

        if not self.use_rope:
            self.position_embeddings = nn.Embedding(
                self.max_position_embeddings,
                embedding_dim,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=self.initializer_range), ),
            )

        self.prefix = prefix

        if self.freeze_embedding:
            self.word_embeddings.weight.learning_rate = 0.0
            if not self.use_rope:
                self.position_embeddings.weight.learning_rate = 0.0

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.rope_head_dim_shape_tensor = paddle.ones((self.rope_head_dim),
                                                      dtype="int8")

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if self.tie_word_embeddings:
            self.word_embeddings.weight.set_value(
                get_tensor(state_dict[self.prefix + ".weight"]).astype(
                    paddle.get_default_dtype()))
        else:
            self.word_embeddings.weight.set_value(
                get_tensor(state_dict.pop(self.prefix + ".weight")).astype(
                    paddle.get_default_dtype()))

    def forward(self, ids_remove_padding=None):
        """
        Defines the forward computation of the layer.

        Args:
            ids_remove_padding (Tensor, optional): Tensor of token IDs, with padding removed.
                If None, no input is provided.

        Returns:
            Tensor: Embedded tensor representation of the input IDs.
        """
        if self.use_ep:
            input_embedings = self.word_embeddings(ids_remove_padding)
        else:
            if self.column_cut:
                input_embedings = self.word_embeddings(ids_remove_padding)
                inputs_embeds_temp = []
                paddle.distributed.all_gather(
                    inputs_embeds_temp,
                    input_embedings,
                    group=fleet.get_hybrid_communicate_group().
                    get_model_parallel_group(),
                    sync_op=True,
                )
                input_embedings = paddle.concat(inputs_embeds_temp, -1)
            else:
                input_embedings = self.word_embeddings(ids_remove_padding)

        return input_embedings
