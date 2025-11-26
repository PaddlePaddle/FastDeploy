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

from fastdeploy.model_executor.utils import default_weight_loader, set_weight_attrs

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
        self.fd_config = fd_config
        self.tp_group = fd_config.parallel_config.tp_group
        self.column_cut = True
        self.tp_size = fd_config.parallel_config.tensor_parallel_size

        ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
        RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        if self.column_cut:
            need_gather = True
            self.linear = ColumnParallelLinear(
                embedding_dim,
                num_embeddings,
                mp_group=self.tp_group,
                weight_attr=None,
                has_bias=True if self.bias_key is not None else False,
                gather_output=need_gather,
                fuse_matmul_bias=False,  # False diff更小
            )
            set_weight_attrs(
                self.linear.weight,
                {
                    "weight_loader": default_weight_loader(self.fd_config),
                    "weight_need_transpose": self.fd_config.model_config.model_format == "torch",
                },
            )
            if self.bias_key is not None:
                set_weight_attrs(
                    self.linear.bias,
                    {"rl_need_attr": {"rl_tp_degree": fd_config.parallel_config.tensor_parallel_size}},
                )
            if self.tp_size > 1:
                set_weight_attrs(self.linear.weight, {"output_dim": True})
        else:
            self.linear = RowParallelLinear(
                embedding_dim,
                num_embeddings,
                mp_group=self.tp_group,
                weight_attr=None,
                has_bias=True if self.bias_key is not None else False,
                input_is_parallel=False,
                fuse_matmul_bias=False,  # False diff更小
            )
            set_weight_attrs(
                self.linear.weight,
                {
                    "weight_loader": default_weight_loader(self.fd_config),
                    "weight_need_transpose": self.fd_config.model_config.model_format == "torch",
                },
            )
            if self.tp_size > 1:
                set_weight_attrs(self.linear.weight, {"output_dim": True})
        set_weight_attrs(
            self.linear.weight, {"rl_need_attr": {"rl_tp_degree": fd_config.parallel_config.tensor_parallel_size}}
        )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

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
        logits = self.linear(logits)
        return logits
