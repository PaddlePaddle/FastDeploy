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


class ParallelLMHead(nn.Layer):
    """
    "Parallelized LM head.
    """

    def __init__(
        self,
        fd_config,
        num_embeddings,
        embedding_dim,
        prefix="",
        with_bias=False,
    ):
        """
        Parallelized LMhead.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings (int): vocabulary size.
            embedding_dim (int): size of hidden state.
            prefix (str): full name of the layer in the state dict
        """
        super(ParallelLMHead, self).__init__()
        self.linear_weight_key = prefix + ".weight"
        if with_bias:
            self.linear_bias_key = prefix + ".bias"
        else:
            self.linear_bias_key = None
        self.use_ep = fd_config.parallel_config.use_ep
        self.column_cut = True

        ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
        RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

        if self.use_ep:
            self.weight = self.create_parameter(
                shape=[embedding_dim, num_embeddings],
                dtype=paddle.get_default_dtype(),
                is_bias=False,
            )
        else:
            if self.column_cut:
                need_gather = True
                self.out_linear = ColumnParallelLinear(
                    embedding_dim,
                    num_embeddings,
                    mp_group=fleet.get_hybrid_communicate_group().
                    get_model_parallel_group(),
                    weight_attr=None,
                    has_bias=True
                    if self.linear_bias_key is not None else False,
                    gather_output=need_gather,
                    fuse_matmul_bias=False,  # False diff更小
                )
            else:
                self.out_linear = RowParallelLinear(
                    embedding_dim,
                    num_embeddings,
                    mp_group=fleet.get_hybrid_communicate_group().
                    get_model_parallel_group(),
                    weight_attr=None,
                    has_bias=True
                    if self.linear_bias_key is not None else False,
                    input_is_parallel=False,
                    fuse_matmul_bias=False,  # False diff更小
                )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        if self.use_ep:
            self.weight.set_value(
                get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                    paddle.get_default_dtype()))
        else:
            if self.tie_word_embeddings:
                self.out_linear.weight.set_value(
                    get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype()).transpose([1, 0]))
            else:
                weight_tensor = get_tensor(
                    state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype())
                if self.out_linear.weight.shape != weight_tensor.shape:
                    weight_tensor = weight_tensor.transpose([1, 0])
                self.out_linear.weight.set_value(weight_tensor)

            if self.linear_bias_key is not None:
                bias = get_tensor(state_dict.pop(self.linear_bias_key)).astype(
                    paddle.get_default_dtype())
                self.out_linear.bias.set_value(bias)

    def forward(self, input):
        """
        Defines the forward computation of the layer.

        Args:
            input (Tensor): The input tensor to the layer.

        Returns:
            Tensor: The output tensor after processing through the layer.
        """
        logits = input
        if self.use_ep:
            logits = paddle.matmul(logits, self.weight)
        else:
            logits = self.out_linear(logits)
        return logits
