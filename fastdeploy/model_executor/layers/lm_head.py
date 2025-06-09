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


def parallel_matmul(lm_output, logit_weights, parallel_output):
    """
    Performs parallel matrix multiplication for large-scale language models.

    Args:
        lm_output (Tensor): The output tensor from the language model layers,
            which will be multiplied with the logit weights.
        logit_weights (Tensor): The weights used in the matrix multiplication,
            typically the weights of the output layer.
        parallel_output (bool): A flag indicating whether to return the parallel
            outputs or concatenate them. If True, returns the outputs from the
            parallel computation directly. If False, concatenates the outputs
            across the model parallel group before returning.

    Returns:
        Tensor: The result of the matrix multiplication. If `parallel_output` is True,
            returns the parallel outputs. If `parallel_output` is False and
            model parallel world size is greater than 1, returns the concatenated
            outputs across the model parallel group. Otherwise, returns the direct
            matrix multiplication result.
    """
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    # rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class ParallelLMHead(nn.Layer):
    """
    "Parallelized LM head.
    """

    def __init__(
        self,
        llm_config,
        num_embeddings,
        embedding_dim,
        prefix="",
        with_bias=False,
        tie_word_embeddings=None,
    ):
        """
        Parallelized LMhead.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings (int): vocabulary size.
            embedding_dim (int): size of hidden state.
            tie_embeddings_weight (bool, optional): Whether to share weights across model parallel ranks,
                defaults to None.
            prefix (str): full name of the layer in the state dict
        """
        super(ParallelLMHead, self).__init__()
        self.use_moe = llm_config.model_config.use_moe
        self.linear_weight_key = prefix + ".weight"
        if with_bias:
            self.linear_bias_key = prefix + ".bias"
        else:
            self.linear_bias_key = None
        self.use_ep = llm_config.parallel_config.use_ep
        self.column_cut = True
        self.fused_linear = True

        hcg = fleet.get_hybrid_communicate_group()
        mp_rank = hcg.get_model_parallel_rank()
        ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
        RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        self.tie_word_embeddings = tie_word_embeddings

        if self.tie_word_embeddings is None:
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
                        has_bias=True,
                        gather_output=need_gather,
                        fuse_matmul_bias=self.fused_linear,  # False diff更小
                    )
                else:
                    self.out_linear = RowParallelLinear(
                        embedding_dim,
                        num_embeddings,
                        mp_group=fleet.get_hybrid_communicate_group().
                        get_model_parallel_group(),
                        weight_attr=None,
                        has_bias=True,
                        input_is_parallel=False,
                        fuse_matmul_bias=self.fused_linear,  # False diff更小
                    )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        if self.tie_word_embeddings is None:
            if self.use_ep:
                self.weight.set_value(
                    get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype()))
            else:
                self.out_linear.weight.set_value(
                    get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype()))

                bias = (
                    get_tensor(state_dict.pop(self.linear_bias_key)).astype(
                        paddle.get_default_dtype()
                    )
                    if self.linear_bias_key is not None
                    else paddle.zeros(
                        self.out_linear.bias.shape, dtype=paddle.get_default_dtype()
                    )
                )
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
        if self.tie_word_embeddings is not None:
            logits = parallel_matmul(logits, self.tie_word_embeddings, False)
        else:
            if self.use_ep:
                logits = paddle.matmul(logits, self.weight)
            else:
                logits = self.out_linear(logits)
        return logits
