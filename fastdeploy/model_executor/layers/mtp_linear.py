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
from paddle import nn
from paddle.distributed import fleet

from fastdeploy.model_executor.utils import set_weight_attrs

from .utils import get_tensor


class ParallelEHProjection(nn.Layer):
    """
    "Parallelized Embedding Hidden States Projection.
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
        Parallelized Embedding Hidden States Projection.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings (int): vocabulary size.
            embedding_dim (int): size of hidden state.
            prefix (str): full name of the layer in the state dict
        """
        super(ParallelEHProjection, self).__init__()
        self.weight_key = prefix + ".weight"
        if with_bias:
            self.bias_key = prefix + ".bias"
        else:
            self.bias_key = None
        self.use_ep = fd_config.parallel_config.use_ep
        self.column_cut = True

        ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
        RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        if self.use_ep:
            self.weight = self.create_parameter(
                shape=[embedding_dim, num_embeddings],
                dtype=paddle.get_default_dtype(),
                is_bias=False,
            )
        else:
            if self.column_cut:
                need_gather = True
                self.linear = ColumnParallelLinear(
                    embedding_dim,
                    num_embeddings,
                    mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                    weight_attr=None,
                    has_bias=True if self.bias_key is not None else False,
                    gather_output=need_gather,
                    fuse_matmul_bias=False,  # False diff更小
                )
                set_weight_attrs(self.linear.weight, {"output_dim": True})
                if self.bias_key is not None:
                    set_weight_attrs(self.linear.bias, {"output_dim": True})
            else:
                self.linear = RowParallelLinear(
                    embedding_dim,
                    num_embeddings,
                    mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                    weight_attr=None,
                    has_bias=True if self.bias_key is not None else False,
                    input_is_parallel=False,
                    fuse_matmul_bias=False,  # False diff更小
                )
                set_weight_attrs(self.linear.weight, {"output_dim": False})

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        if self.use_ep:
            self.weight.set_value(get_tensor(state_dict.pop(self.weight_key)).astype(paddle.get_default_dtype()))
        else:
            weight_tensor = get_tensor(state_dict.pop(self.weight_key)).astype(paddle.get_default_dtype())
            if self.linear.weight.shape != weight_tensor.shape:
                weight_tensor = weight_tensor.transpose([1, 0])
            self.linear.weight.set_value(weight_tensor)

            if self.bias_key is not None:
                bias = get_tensor(state_dict.pop(self.bias_key)).astype(paddle.get_default_dtype())
                self.linear.bias.set_value(bias)

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
            logits = self.linear(logits)
        return logits
