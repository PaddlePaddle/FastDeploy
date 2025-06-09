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

from paddlenlp.utils.log import logger

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    VocabParallelEmbedding,
)

from .utils import get_tensor


class ResBlock(nn.Layer):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, num_condition=0):
        super().__init__()
        self.linear = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        if num_condition > 0:
            self.res_connection = nn.Linear(
                hidden_size * (num_condition + 1), hidden_size
            )
        else:
            self.res_connection = nn.Identity()
        # Initialize as an identity mapping
        # _no_grad_fill_(self.linear.weight, 0)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.Silu()

    @paddle.no_grad()
    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: Output after the residual connection and activation.
        """
        return self.res_connection(x) + self.act(self.linear(x))


class HydraHead(nn.Layer):
    """
    A Hydra Head module.

    This module performs multi hydra head layers,
    each of which is a hydra_lm_head followed by a head

    Args:
        hydra_num_heads (int): The number of hyhra heads.
        hydra_num_layers (int): The number of layers.
        hidden_size (int): The size of the hidden layers in the block.
        tensor_parallel_degree(int): TP degree.
        vocab_size (int): The size of vocabulary.
    """

    def __init__(
        self,
        hydra_num_heads,
        hydra_num_layers,
        hidden_size,
        tensor_parallel_degree,
        vocab_size,
    ):
        super().__init__()
        self.hydra_num_heads = hydra_num_heads
        self.hydra_num_layers = hydra_num_layers
        self.hidden_size = hidden_size
        self.tensor_parallel_degree = tensor_parallel_degree
        self.vocab_size = vocab_size

        self.hydra_mlp = nn.LayerList(
            [
                nn.Sequential(
                    ResBlock(self.hidden_size, hydra_head_idx + 1),
                    *([ResBlock(self.hidden_size)] * (self.hydra_num_layers - 1)),
                )
                for hydra_head_idx in range(self.hydra_num_heads)
            ]
        )

        if self.tensor_parallel_degree > 1:
            self.hydra_lm_head = nn.LayerList(
                [
                    ColumnParallelLinear(
                        self.hidden_size,
                        self.vocab_size,
                        weight_attr=paddle.ParamAttr(
                            initializer=nn.initializer.Normal(mean=0.0, std=0.0)
                        ),
                        gather_output=True,
                        has_bias=False,
                    )
                    for _ in range(self.hydra_num_heads)
                ]
            )
        else:
            self.hydra_lm_head = nn.LayerList(
                [
                    nn.Linear(self.hidden_size, self.vocab_size, bias_attr=False)
                    for _ in range(self.hydra_num_heads)
                ]
            )

        self.word_embeddings = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0)),
        )

    def custom_set_state_dict(self, state_dict):
        """
        Load Parameter of Hydra Head from state_dict with custom names.

        Args:
            state_dict (dict): KV pair of name and parameters.
        """
        for hydra_head_idx in range(self.hydra_num_heads):
            self.hydra_mlp[hydra_head_idx][0].res_connection.weight.set_value(
                get_tensor(
                    state_dict.pop(f"0.{hydra_head_idx}.0.res_connection.weight")
                )
            )
            self.hydra_mlp[hydra_head_idx][0].res_connection.bias.set_value(
                get_tensor(state_dict.pop(f"0.{hydra_head_idx}.0.res_connection.bias"))
            )

            for layer_idx in range(self.hydra_num_layers):
                self.hydra_mlp[hydra_head_idx][layer_idx].linear.weight.set_value(
                    get_tensor(
                        state_dict.pop(f"0.{hydra_head_idx}.{layer_idx}.linear.weight")
                    )
                )
                self.hydra_mlp[hydra_head_idx][layer_idx].linear.bias.set_value(
                    get_tensor(
                        state_dict.pop(f"0.{hydra_head_idx}.{layer_idx}.linear.bias")
                    )
                )

            self.hydra_lm_head[hydra_head_idx].weight.set_value(
                get_tensor(state_dict.pop(f"1.{hydra_head_idx}.weight"))
            )

        self.word_embeddings.weight.set_value(
            get_tensor(state_dict.pop("word_embeddings.weight"))
        )

    def set_state_dict(self, state_dict):
        """
        Load Parameter of Hydra Head from state_dict.

        Args:
            state_dict (dict): KV pair of name and parameters.
        """
        is_custom = True
        for key in state_dict.keys():
            if key != "word_embeddings.weight" and (
                "hydra_mlp" in key or "hydra_head" in key
            ):
                is_custom = False
                break

        if is_custom:
            logger.info("Hydra use custom set_state_dict")
            self.custom_set_state_dict(state_dict)
        else:
            logger.info("Hydra use default set_state_dict")
            super().set_state_dict(state_dict)

    @paddle.no_grad()
    def forward(self, input_ids, hidden_states, next_tokens):
        """
        Forward pass of Hydra Head

        Args:
            input_ids: [batch_size, 1] The tokens sampled by the previous head go through the embedding,
                                        starting with the last accept token
            hidden_states: [batch_size, hidden_size] The hidden_states of the last accept_tokens
        """
        hydra_inputs = [hidden_states]
        input_embeds = self.word_embeddings(input_ids)
        for hydra_head_idx in range(self.hydra_num_heads):
            hydra_inputs.append(input_embeds)
            head_input = paddle.concat(hydra_inputs, axis=-1)
            hidden_states = self.hydra_mlp[hydra_head_idx](head_input)
            logits = self.hydra_lm_head[hydra_head_idx](hidden_states)
            probs = F.softmax(logits)
            _, topk_tokens = paddle.topk(probs, k=1, axis=-1)
            next_tokens[:, 1 + hydra_head_idx : 2 + hydra_head_idx] = topk_tokens[:]

            input_embeds = self.word_embeddings(next_tokens[:, 1 + hydra_head_idx])
