"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import re

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import (
        get_position_ids_and_mask_encoder_batch,
    )


from .deepseek_v3 import DeepSeekV3DecoderLayer

# ======================= GLM-5 MoE DSA Model =======================


@support_graph_optimization
class GlmMoeDsaModel(nn.Layer):
    """
    GlmMoeDsaModel - GLM-5 with MLA/DSA attention
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the GlmMoeDsaModel class.
        """
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix="model.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                DeepSeekV3DecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix="model.norm",
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """ """
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
                position_ids,
                mask_encoder_batch,
            )
        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


@ModelRegistry.register_model_class(
    architecture="GlmMoeDsaForCausalLM",
    module_name="glm_moe_dsa",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class GlmMoeDsaForCausalLM(ModelForCasualLM):
    """
    GlmMoeDsaForCausalLM - GLM-5 with MLA/DSA attention
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super().__init__(fd_config)
        self.model = GlmMoeDsaModel(fd_config)
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.position_ids_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens], dtype=paddle.int32
        )
        self.mask_encoder_batch_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens, 1], dtype=paddle.int32
        )

    @classmethod
    def name(cls):
        """ """
        return "GlmMoeDsaForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.
        """
        pass

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.
        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
            ("qkv_a_proj_with_mqa", "q_a_proj", "q_a"),
            ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", "kv_a"),
        ]
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)

                if model_param_name not in params_dict:
                    continue

                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            if "kv_b_proj" in model_sublayer_name:
                kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                process_weights_after_loading_fn(kv_model_sublayer_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """ """
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def pre_process(self, forward_meta):
        """ """
        seq_lens_encoder = forward_meta.seq_lens_encoder
        seq_lens_decoder = forward_meta.seq_lens_decoder
        seq_lens_this_time = forward_meta.seq_lens_this_time

        current_total_tokens = forward_meta.ids_remove_padding.shape[0]
        position_ids = self.position_ids_buffer[:current_total_tokens]
        mask_encoder_batch = self.mask_encoder_batch_buffer[:current_total_tokens]

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            position_ids,
            mask_encoder_batch,
        )
        return position_ids, mask_encoder_batch

    def empty_input_forward(self, forward_meta):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[1, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.first_k_dense_replace,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.model.layers[i].mlp.experts(fake_hidden_states, self.model.layers[i].mlp.gate, forward_meta)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        position_ids, mask_encoder_batch = self.pre_process(forward_meta)
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            position_ids=position_ids,
            mask_encoder_batch=mask_encoder_batch,
        )
        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class GlmMoeDsaPretrainedModel(PretrainedModel):
    """
    GlmMoeDsaPretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "GlmMoeDsaForCausalLM"
