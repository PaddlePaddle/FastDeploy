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
from typing import List, Optional

import paddle
from paddle import nn
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
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry
from fastdeploy.model_executor.models.qwen3_vl.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from fastdeploy.model_executor.models.qwen3moe import (
    Qwen3DecoderLayer as Qwen3MoeDecoderLayer,
)


@support_graph_optimization
class Qwen3VLMoeModel(nn.Layer):
    """Language backbone for Qwen3-VL-MOE."""

    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        self.image_token_id = fd_config.model_config.image_token_id
        self.video_token_id = fd_config.model_config.video_token_id
        self._dtype = fd_config.model_config.dtype
        fd_config.model_config.pretrained_config.prefix_name = "model"
        self.fd_config = fd_config

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                Qwen3MoeDecoderLayer(
                    fd_config=fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def get_input_embeddings(self, ids_remove_padding: paddle.Tensor) -> paddle.Tensor:
        return self.embed_tokens(ids_remove_padding=ids_remove_padding)

    def forward(
        self,
        input_embeddings: paddle.Tensor,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        deepstack_inputs: Optional[List[paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        hidden_states = input_embeddings
        residual = None
        for layer_id, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                forward_meta,
                hidden_states,
                residual,
            )
            if deepstack_inputs is not None and layer_id < len(deepstack_inputs):
                hidden_states = hidden_states + deepstack_inputs[layer_id]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@ModelRegistry.register_model_class(
    architecture="Qwen3VLMoeForConditionalGeneration",
    module_name="qwen3_vl.qwen3_vl_moe",
    category=ModelCategory.MULTIMODAL,
    primary_use=ModelCategory.MULTIMODAL,
)
class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, fd_config: FDConfig) -> None:
        # Skip one layer of super to prevent cudagraph from initializing Qwen3_VLModel
        super(Qwen3VLForConditionalGeneration, self).__init__(fd_config)
        self.visual = self._init_vision_model(fd_config.model_config)
        self.model = Qwen3VLMoeModel(fd_config=fd_config)

        # Persistent buffers for CUDA graphs.
        if fd_config.graph_opt_config.use_cudagraph:
            self._buffer_input_embeddings = paddle.zeros(
                [fd_config.graph_opt_config.max_capture_size, fd_config.model_config.hidden_size],
                dtype=fd_config.model_config.dtype,
            )

        # token ids (convenience aliases)
        self.image_token_id = fd_config.model_config.image_token_id
        self.video_token_id = fd_config.model_config.video_token_id
        self.context_hidden_size = fd_config.model_config.hidden_size

        vision_config = fd_config.model_config.vision_config
        self.visual_hidden_size = vision_config.out_hidden_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes

        self.use_deepstack = hasattr(vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        self._deepstack_cache_capacity = fd_config.model_config.max_model_len if self.use_deepstack else 0
        self._deepstack_cache_len = 0
        self.deepstack_input_embeds: Optional[List[paddle.Tensor]] = None
        if self.use_deepstack:
            dtype = fd_config.model_config.dtype
            # Persistent buffers for CUDA graphs.
            # Note that the current multimodal model does not limit the number of tokens
            # per batch to max_num_batched_tokens, so we use model_config.max_model_len here
            buffer_seq_len = fd_config.model_config.max_model_len
            # self.deepstack_input_embeds = [
            #     paddle.zeros([fd_config.scheduler_config.max_num_batched_tokens, self.context_hidden_size], dtype=dtype)
            #     for _ in range(self.deepstack_num_level)
            # ]
            self.deepstack_input_embeds = [
                paddle.zeros([buffer_seq_len, self.context_hidden_size], dtype=dtype)
                for _ in range(self.deepstack_num_level)
            ]

        self.visual_dim = vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        self.fd_config = fd_config

    @classmethod
    def name(cls) -> str:
        return "Qwen3VLMoeForConditionalGeneration"

    def get_expert_mapping(
        self,
    ) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: paddle.Tensor,
        shard_id: str,
        num_experts: int,
    ):
        param = params_dict[name]
        weight_loader = param.weight_loader
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            weight_loader(
                param,
                curr_expert_weight,
                shard_id=shard_id,
                expert_id=expert_id,
            )

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """Load model parameters from a given weights iterator."""

        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("visual", "model.visual", None),
            ("qk_norm.q_norm", "q_norm", None),
            ("qk_norm.k_norm", "k_norm", None),
        ]

        expert_params_mapping = self.get_expert_mapping()  # Not actually used
        params_dict = dict(self.named_parameters())
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.up_gate_proj_weight", "experts.gate_up_proj", 0, "gate"),
            ("experts.down_proj_weight", "experts.down_proj", 0, "down"),
        ]
        num_experts = self.fd_config.model_config.num_experts
        # params_name model.embed_tokens.embeddings.weight
        # weight_name model.language_model.embed_tokens.weight
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        logger.info(f"[Qwen3Moe-VL] params_dict names: {list(params_dict.keys())} ")
        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            loaded_weight_name = loaded_weight_name.replace(".language_model", "")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in loaded_weight_name or "experts.down_proj" in loaded_weight_name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts" in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                # Note: mlp.experts.gate_up_proj in qwen3moe_vl is merged and should be processed separately when loading weights
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in loaded_weight_name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            self.load_fused_expert_weights(
                                model_param_name,
                                params_dict,
                                gate_weight,
                                "gate",
                                num_experts,
                            )
                            self.load_fused_expert_weights(
                                model_param_name,
                                params_dict,
                                up_weight,
                                "up",
                                num_experts,
                            )
                        else:
                            # down_proj
                            self.load_fused_expert_weights(
                                model_param_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        break
                    else:
                        model_param_name = loaded_weight_name.replace(weight_name, param_name)
                        if model_param_name not in params_dict:
                            continue
                        param = params_dict[model_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                        break
                else:
                    if is_expert_weight:
                        continue
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(
                self.model.embed_tokens.embeddings.weight.transpose([1, 0]).astype(self.lm_head.linear.weight.dtype)
            )
