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
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.models.qwen3 import Qwen3DecoderLayer
from fastdeploy.model_executor.models.qwen3_vl.dfnrope.modeling import (
    Qwen3VisionTransformerPretrainedModel,
)
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
from fastdeploy.model_executor.models.utils import WeightMeta


@support_graph_optimization
class Qwen3_VLModel(nn.Layer):
    """Language backbone for Qwen3-VL."""

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
                Qwen3DecoderLayer(
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
    architecture="Qwen3VLForConditionalGeneration",
    module_name="qwen3_vl.qwen3_vl",
    category=ModelCategory.MULTIMODAL,
    primary_use=ModelCategory.MULTIMODAL,
)
class Qwen3VLForConditionalGeneration(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__(fd_config)
        self.visual = self._init_vision_model(fd_config.model_config)
        self.model = Qwen3_VLModel(fd_config=fd_config)

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

    def _init_vision_model(self, model_config) -> nn.Layer:
        visual = Qwen3VisionTransformerPretrainedModel(model_config, prefix_name="visual")
        visual = paddle.amp.decorate(models=visual, level="O2", dtype="bfloat16")
        visual.eval()
        return visual

    @classmethod
    def name(cls) -> str:
        return "Qwen3VLForConditionalGeneration"

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

        params_dict = dict(self.named_parameters())
        # params_name model.embed_tokens.embeddings.weight
        # weight_name model.language_model.embed_tokens.weight
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        logger.info(f"[Qwen3-VL] params_dict names: {list(params_dict.keys())} ")
        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            loaded_weight_name = loaded_weight_name.replace(".language_model", "")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                # logger.info(
                #     f"[Qwen3-VL] loaded_weight_name: {loaded_weight_name}, weight_name {weight_name}, param_name {param_name}, model_param_name {model_param_name} 1"
                # )
                if model_param_name not in params_dict:
                    logger.info(f"[Qwen3-VL] {model_param_name} not in params_dict1")
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    logger.info(f"[Qwen3-VL] {model_param_name} not in params_dict2")
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(
                self.model.embed_tokens.embeddings.weight.transpose([1, 0]).astype(self.lm_head.linear.weight.dtype)
            )

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]) -> None:
        self.model.load_state_dict(state_dict)
        self.visual.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.model.embed_tokens.embeddings.weight})
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: paddle.Tensor) -> None:
        num_tokens = deepstack_input_embeds.shape[1]
        # Expanding self.deepstack_input_embeds is not allowed,
        # otherwise the CUDA graph will access illegal addresses
        assert num_tokens <= self.deepstack_input_embeds[0].shape[0], (
            f"num_tokens ({num_tokens}) is greater than the current "
            f"max size of deepstack_input_embeds ({self.deepstack_input_embeds[0].shape[0]})"
        )
        # Expanding self.deepstack_input_embeds
        # if num_tokens > self.deepstack_input_embeds[0].shape[0]:
        #     self.deepstack_input_embeds = [
        #         paddle.zeros(
        #             num_tokens,
        #             self.context_hidden_size,
        #             dtype=self.deepstack_input_embeds[0].dtype,
        #             # device=self.deepstack_input_embeds[0].place,
        #         )
        #         for _ in range(self.deepstack_num_level)
        #     ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(deepstack_input_embeds[idx], False)

    def _get_deepstack_input_embeds(self, num_tokens: int) -> Optional[List[paddle.Tensor]]:
        return [tensor[:num_tokens] for tensor in self.deepstack_input_embeds]

    def _clear_deepstack_input_embeds(self, num_token: int) -> None:
        if num_token > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_token].zero_()

    def _compute_deepstack_embeds_v0(
        self,
        input_embeddings: paddle.Tensor,
        image_features: paddle.Tensor,
        vision_mask: paddle.Tensor,
    ):
        """For only image inputs case"""
        mm_embeddings_main, mm_embeddings_multiscale = paddle.split(
            image_features, num_or_sections=[self.visual_dim, self.multiscale_dim], axis=-1
        )

        deepstack_input_embeds = input_embeddings.new_zeros(
            size=[input_embeddings.shape[0], self.deepstack_num_level * input_embeddings.shape[1]],
        )

        deepstack_input_embeds[vision_mask] = mm_embeddings_multiscale
        deepstack_input_embeds = deepstack_input_embeds.view(
            input_embeddings.shape[0], self.deepstack_num_level, self.visual_dim
        )
        deepstack_input_embeds = deepstack_input_embeds.transpose([1, 0, 2])

        return deepstack_input_embeds, mm_embeddings_main

    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_embeddings = self.model.get_input_embeddings(ids_remove_padding=ids_remove_padding)

        if image_features is None:
            return input_embeddings

        image_mask = ids_remove_padding == self.model.image_token_id
        image_token_num = image_mask.sum()

        if image_token_num.item() <= 0:
            return input_embeddings

        deepstack_input_embeds = None

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                mm_embeddings,
            ) = self._compute_deepstack_embeds_v0(
                input_embeddings,
                image_features,
                image_mask,
            )

        if image_token_num.item() > 0:
            input_embeddings[image_mask] = mm_embeddings

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return input_embeddings

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        input_embeddings = self.get_input_embeddings(ids_remove_padding, image_features)
        deepstack_inputs = None
        if self.use_deepstack:
            deepstack_inputs = self._get_deepstack_input_embeds(input_embeddings.shape[0])

        if forward_meta.step_use_cudagraph:
            self._buffer_input_embeddings.copy_(input_embeddings, False)
            input_embeddings = self._buffer_input_embeddings

        hidden_states = self.model(
            input_embeddings=input_embeddings,
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            deepstack_inputs=deepstack_inputs,
        )

        if self.use_deepstack:
            self._clear_deepstack_input_embeds(input_embeddings.shape[0])

        return hidden_states


class Qwen3VLPretrainedModel(PretrainedModel):
    """Utilities for tensor-parallel weight splitting."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(cls) -> str:
        return "Qwen3VLForConditionalGeneration"

    weight_infos = [
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.q_proj.weight", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.q_proj.bias", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.k_proj.weight", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.k_proj.bias", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.v_proj.weight", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.v_proj.bias", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.weight", False),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.mlp.gate_proj.weight", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.mlp.up_proj.weight", True),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.mlp.down_proj.weight", False),
        WeightMeta(".embed_tokens.weight", False),
        WeightMeta("lm_head.weight", True),
    ]

    weight_vision = [
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.attn.proj.weight", False),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.linear_fc1.weight", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.linear_fc1.bias", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.linear_fc2.weight", False),
        WeightMeta(
            f"visual.blocks.{{{layerid.LAYER_ID}}}.attn.qkv.weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(
            f"visual.blocks.{{{layerid.LAYER_ID}}}.attn.qkv.bias",
            True,
            tsm.GQA,
        ),
        WeightMeta("visual.merger.linear_fc1.weight", True),
        WeightMeta("visual.merger.linear_fc1.bias", True),
        WeightMeta("visual.merger.linear_fc2.weight", False),
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split: bool = True):
        return {}
        # logger.info("qwen3_vl inference model _get_tensor_parallel_mappings")
        # from fastdeploy.model_executor.models.tp_utils import (
        #     build_expanded_keys,
        #     has_prefix,
        #     split_or_merge_func_v1,
        # )

        # fn = split_or_merge_func_v1(
        #     is_split=is_split,
        #     tensor_model_parallel_size=config.tensor_model_parallel_size,
        #     tensor_parallel_rank=config.tensor_parallel_rank,
        #     num_attention_heads=config.num_attention_heads,
        #     num_key_value_heads=config.num_key_value_heads,
        #     head_dim=config.head_dim,
        # )

        # vision_num_heads = config.vision_config.get("num_heads")
        # vision_hidden = config.vision_config.get("hidden_size")
        # vision_head_dim = vision_hidden // vision_num_heads
        # vision_fn = split_or_merge_func_v1(
        #     is_split=is_split,
        #     tensor_model_parallel_size=config.tensor_model_parallel_size,
        #     tensor_parallel_rank=config.tensor_parallel_rank,
        #     num_attention_heads=vision_num_heads,
        #     num_key_value_heads=vision_num_heads,
        #     head_dim=vision_head_dim,
        # )

        # def get_tensor_parallel_split_mappings(num_layers: int, prefix_name: str):
        #     base_actions = {}
        #     for weight_name, is_column, extra in cls.weight_infos:
        #         params = {"is_column": is_column, **({extra.value: True} if extra else {})}

        #         if "lm_head.weight" in weight_name or weight_name.startswith("."):
        #             key = weight_name
        #         elif not has_prefix(prefix_name, weight_name):
        #             key = f"{prefix_name}{weight_name}"
        #         else:
        #             key = weight_name
        #         base_actions[key] = partial(fn, **params)

        #     return build_expanded_keys(base_actions, num_layers)

        # def get_vision_parallel_split_mappings(num_layers: int, deepstack_count: int):
        #     base_actions = {}
        #     for weight_name, is_column, extra in cls.weight_vision:
        #         params = {"is_column": is_column, **({extra.value: True} if extra else {})}
        #         base_actions[weight_name] = partial(vision_fn, **params)

        #     actions = build_expanded_keys(
        #         {k: v for k, v in base_actions.items() if "visual.blocks." in k},
        #         num_layers,
        #     )

        #     for key, action in base_actions.items():
        #         if "visual.blocks." not in key:
        #             actions[key] = action

        #     for idx in range(deepstack_count):
        #         actions[f"visual.deepstack_merger_list.{idx}.linear_fc1.weight"] = partial(vision_fn, is_column=True)
        #         actions[f"visual.deepstack_merger_list.{idx}.linear_fc1.bias"] = partial(vision_fn, is_column=True)
        #         actions[f"visual.deepstack_merger_list.{idx}.linear_fc2.weight"] = partial(vision_fn, is_column=False)
        #     return actions

        # mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers, config.prefix_name)
        # vision_depth = config.vision_config.get("depth", 0)
        # deepstack_count = len(config.vision_config.get("deepstack_visual_indexes", []))
        # vision_mappings = get_vision_parallel_split_mappings(vision_depth, deepstack_count)

        # mappings.update(vision_mappings)
        # return mappings
