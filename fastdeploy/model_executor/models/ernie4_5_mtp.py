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

from functools import partial
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_DecoderLayer
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.worker.forward_meta import ForwardMeta


class Ernie4_5_MTPPretrainedModel(PretrainedModel):
    """
    Ernie4_5_MTPPretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("erine inference model _get_tensor_parallel_mappings")

        from paddleformers.transformers.conversion_utils import \
            split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def gqa_qkv_split_func(
            weight,
            tensor_parallel_degree,
            tensor_parallel_rank,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
        ):

            def get_shape(tensor):
                return (tensor.get_shape()
                        if hasattr(tensor, "get_shape") else tensor.shape)

            def slice_tensor(tensor, start, end):
                shape = get_shape(tensor)
                if len(shape) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_end = num_attention_heads * head_dim
            k_end = q_end + num_key_value_heads * head_dim
            v_end = k_end + num_key_value_heads * head_dim

            q = slice_tensor(weight, 0, q_end)
            k = slice_tensor(weight, q_end, k_end)
            v = slice_tensor(weight, k_end, v_end)

            def split_tensor(tensor, degree):
                shape = get_shape(tensor)
                size = shape[-1]
                block_size = size // degree
                if hasattr(tensor, "get_shape"):
                    return [
                        slice_tensor(tensor, i * block_size,
                                     (i + 1) * block_size)
                        for i in range(degree)
                    ]
                else:
                    return np.split(tensor, degree, axis=-1)

            q_list = split_tensor(q, tensor_parallel_degree)
            k_list = split_tensor(k, tensor_parallel_degree)
            v_list = split_tensor(v, tensor_parallel_degree)

            if tensor_parallel_rank is None:
                return [
                    np.concatenate([q_i, k_i, v_i], axis=-1)
                    for q_i, k_i, v_i in zip(q_list, k_list, v_list)
                ]
            else:
                return np.concatenate(
                    [
                        q_list[tensor_parallel_rank],
                        k_list[tensor_parallel_rank],
                        v_list[tensor_parallel_rank],
                    ],
                    axis=-1,
                )

        def gqa_qkv_merge_func(weight_list, num_attention_heads,
                               num_key_value_heads, head_dim):
            tensor_parallel_degree = len(weight_list)
            num_attention_heads = num_attention_heads // tensor_parallel_degree
            num_key_value_heads = num_key_value_heads // tensor_parallel_degree

            is_paddle_tensor = not isinstance(weight_list[0], np.ndarray)

            def get_shape(tensor):
                return (tensor.get_shape()
                        if hasattr(tensor, "get_shape") else tensor.shape)

            def slice_tensor(tensor, start, end):
                if len(get_shape(tensor)) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_list, k_list, v_list = [], [], []

            for weight in weight_list:
                q_end = num_attention_heads * head_dim
                k_end = q_end + num_key_value_heads * head_dim
                v_end = k_end + num_key_value_heads * head_dim

                q = slice_tensor(weight, 0, q_end)
                k = slice_tensor(weight, q_end, k_end)
                v = slice_tensor(weight, k_end, v_end)

                q_list.append(q)
                k_list.append(k)
                v_list.append(v)

            merged = q_list + k_list + v_list

            if is_paddle_tensor:
                tensor = paddle.concat(merged, axis=-1)
                if tensor.place.is_gpu_place():
                    tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
                return tensor
            else:
                return np.concatenate(merged, axis=-1)

        if (config.num_key_value_heads is not None
                and config.num_key_value_heads != config.num_attention_heads):
            if is_split:
                qkv_fn = partial(
                    gqa_qkv_split_func,
                    tensor_parallel_degree=config.tensor_parallel_degree,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
            else:
                qkv_fn = partial(
                    gqa_qkv_merge_func,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
        else:
            qkv_fn = partial(fn, is_column=True)

        def get_tensor_parallel_split_mappings(num_layers, moe_num_experts,
                                               moe_layer_start_index):
            """
            get tensor from parallel-split-mappings
            """
            final_actions = {}
            base_model_prefix = "ernie.mtp_block"

            base_actions = {}

            base_actions["ernie.mtp_linear_proj.0.weight"] = partial(
                fn, is_column=True)
            base_actions[
                f"{base_model_prefix}.0.self_attn.qkv_proj.weight"] = qkv_fn
            base_actions[
                f"{base_model_prefix}.0.self_attn.o_proj.weight"] = partial(
                    fn, is_column=False)
            base_actions[
                f"{base_model_prefix}.0.mlp.up_gate_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True)
            base_actions[f"{base_model_prefix}.0.mlp.down_proj.weight"] = (
                partial(fn, is_column=False))

            for expert_idx in range(moe_num_experts):
                base_actions[
                    f"{base_model_prefix}.{moe_layer_start_index}"
                    f".mlp.experts.{expert_idx}.up_gate_proj.weight"] = partial(
                        fn, is_column=True, is_naive_2fuse=True)
                base_actions[
                    f"{base_model_prefix}.{moe_layer_start_index}"
                    f".mlp.experts.{expert_idx}.down_proj.weight"] = partial(
                        fn, is_column=False)

            for key, action in base_actions.items():
                if (f"{base_model_prefix}.0.mlp.up_gate_proj.weight" in key or
                        f"{base_model_prefix}.0.mlp.down_proj.weight" in key):
                    for i in range(moe_layer_start_index):
                        final_actions[key.replace("0.", f"{i}.")] = action
                elif f"{moe_layer_start_index}.mlp.experts." in key:
                    for i in range(moe_layer_start_index, num_layers):
                        final_actions[key.replace(f"{moe_layer_start_index}.",
                                                  f"{i}.")] = action
                elif f"{base_model_prefix}.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("0.", f"{i}.")] = action
                final_actions[key] = action
            return final_actions

        moe_num_experts = 0
        mappings = get_tensor_parallel_split_mappings(
            config.num_layers,
            moe_num_experts,
            config.moe_layer_start_index,
        )

        return mappings


class Ernie4_5_MTPModel(nn.Layer):
    """
    Ernie4_5_MTPModel
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie4_5_MTPModel class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        self.embeddings = fd_config.speculative_config.sharing_model.model.embeddings

        self.hidden_layers = nn.LayerList([
            Ernie4_5_DecoderLayer(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.{i}")
            for i in range(self.num_layers)
        ])

        self.enorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix="ernie.mtp_emb_norm.0",
        )

        self.hnorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix="ernie.mtp_hidden_norm.0",
        )

        self.eh_proj = ParallelLMHead(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.hidden_size,
            embedding_dim=fd_config.model_config.hidden_size * 2,
            prefix="ernie.mtp_linear_proj.0",
        )

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        # self.embeddings.load_state_dict(state_dict)
        self.enorm.load_state_dict(state_dict)
        self.hnorm.load_state_dict(state_dict)
        self.eh_proj.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        previous_hidden_states: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        forward
        """
        inputs_embedding = self.embeddings(
            ids_remove_padding=ids_remove_padding)
        inputs_embedding = paddle.concat(
            [self.enorm(inputs_embedding),
             self.hnorm(previous_hidden_states)],
            axis=-1)
        hidden_states = self.eh_proj(inputs_embedding)
        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.hidden_layers[i](forward_meta,
                                                            hidden_states,
                                                            residual)

        hidden_states = hidden_states + residual

        return hidden_states


class Ernie4_5_MTPForCausalLM(ModelForCasualLM):
    """
    Ernie4_5_MTPForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_MTPForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.model = Ernie4_5_MTPModel(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = fd_config.speculative_config.sharing_model.lm_head
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

    @classmethod
    def name(self):
        """
        """
        return "Ernie4_5_MTPForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray,
                                                         paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.model.load_state_dict(state_dict)
        # if self.tie_word_embeddings:
        #     self.lm_head.out_linear.weight.set_value(
        #         self.model.embeddings.word_embeddings.weight.transpose([1, 0]))
        # else:
        #     self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """
        compute logits
        """
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(self.fd_config.moe_config.moe_layer_start_index,
                       self.fd_config.model_config.num_layers):
            self.model.hidden_layers[i].mlp.fused_moe(fake_hidden_states)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        previous_hidden_states: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        forward
        """
        hidden_states = self.model(ids_remove_padding, previous_hidden_states,
                                   forward_meta)

        return hidden_states
