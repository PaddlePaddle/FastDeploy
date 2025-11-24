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

import re
from functools import partial
from typing import Dict, Optional, Union

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
from fastdeploy.model_executor.models.qwen2 import Qwen2DecoderLayer


@support_graph_optimization
class Qwen2_5_VLModel(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie4_5_VLModel class.

        Args:

        """
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
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Qwen2DecoderLayer(
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
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
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
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ):
        hidden_states = input_embeddings

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
            )

        out = self.norm(hidden_states, residual)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="Qwen2_5_VLForConditionalGeneration",
    module_name="qwen2_5_vl.qwen2_5_vl",
    category=ModelCategory.MULTIMODAL,
    primary_use=ModelCategory.MULTIMODAL,
)
class Qwen2_5_VLForConditionalGeneration(ModelForCasualLM):
    """
    Qwen2_5_VLForConditionalGeneration
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(fd_config)
        # ----------- vision model ------------
        self.visual = self._init_vision_model(fd_config.model_config)
        # -----------  language model -------------
        self.model = Qwen2_5_VLModel(fd_config=fd_config)

        # Persistent buffers for CUDA graphs.
        if fd_config.graph_opt_config.use_cudagraph:
            self._decoder_input_embeddings = paddle.zeros(
                [fd_config.graph_opt_config.max_capture_size, fd_config.model_config.hidden_size],
                dtype=fd_config.model_config.dtype,
            )

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

    def _init_vision_model(self, model_config) -> nn.Layer:
        from fastdeploy.model_executor.models.qwen2_5_vl.dfnrope.modeling import (
            DFNRopeVisionTransformerPretrainedModel,
        )

        visual = DFNRopeVisionTransformerPretrainedModel(model_config, prefix_name="visual")
        visual = paddle.amp.decorate(models=visual, level="O2", dtype="bfloat16")
        visual.eval()
        return visual

    @classmethod
    def name(self):
        return "Qwen2_5_VLForConditionalGeneration"

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
            # 参数变量名与权重key不同的要做映射
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        for loaded_weight_name, loaded_weight in weights_iterator:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)
            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(self.model.embed_tokens.embeddings.weight.transpose([1, 0]))

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.model.load_state_dict(state_dict)
        self.visual.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(self.model.embed_tokens.embeddings.weight.transpose([1, 0]))
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:

        input_embeddings = self.model.get_input_embeddings(ids_remove_padding=ids_remove_padding)

        image_mask = ids_remove_padding == self.model.image_token_id
        image_token_num = image_mask.sum()

        video_mask = ids_remove_padding == self.model.video_token_id
        video_token_num = video_mask.sum()

        # Due to the fact that the framework only has image_features,
        # it currently does not support mixing images and videos
        # TODO(wangyafeng) Consider supporting the input of video_features in the future
        if image_token_num.item() > 0:
            input_embeddings[image_mask] = image_features.cast(self.model._dtype)
        if video_token_num.item() > 0:
            input_embeddings[video_mask] = image_features.cast(self.model._dtype)

        return input_embeddings

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ):
        input_embeddings = self.get_input_embeddings(
            ids_remove_padding=ids_remove_padding, image_features=image_features
        )

        if forward_meta.step_use_cudagraph:
            self._decoder_input_embeddings.copy_(input_embeddings, False)
            input_embeddings = self._decoder_input_embeddings

        hidden_states = self.model(
            input_embeddings=input_embeddings,
            ids_remove_padding=ids_remove_padding,
            image_features=image_features,
            forward_meta=forward_meta,
        )

        return hidden_states


class Qwen2_5_VLPretrainedModel(PretrainedModel):
    """
    Qwen2_PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "Qwen2_5_VLForConditionalGeneration"

    from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
    from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
    from fastdeploy.model_executor.models.utils import WeightMeta

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

    weight_vison = [
        # vision
        WeightMeta(
            f"visual.blocks.{{{layerid.LAYER_ID}}}.attn.proj.weight",
            False,
        ),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.up_proj.weight", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.up_proj.bias", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.gate_proj.weight", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.gate_proj.bias", True),
        WeightMeta(f"visual.blocks.{{{layerid.LAYER_ID}}}.mlp.down_proj.weight", False),
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
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("qwen2_5_vl inference model _get_tensor_parallel_mappings")
        from fastdeploy.model_executor.models.tp_utils import (
            build_expanded_keys,
            has_prefix,
            split_or_merge_func_v1,
        )

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        vision_fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.vision_config.get("num_heads"),
            num_key_value_heads=config.vision_config.get("num_heads"),
            head_dim=config.vision_config.get("hidden_size") // config.vision_config.get("num_heads"),
        )

        def get_tensor_parallel_split_mappings(
            num_layers: int,
            prefix_name: str,
        ):
            base_actions = {}
            for weight_name, is_column, extra in cls.weight_infos:
                params = {
                    "is_column": is_column,
                    **({extra.value: True} if extra else {}),
                }

                if "lm_head.weight" or "" in weight_name:
                    key = weight_name
                elif not has_prefix(prefix_name, weight_name):
                    key = f"{prefix_name}{weight_name}"
                else:
                    key = weight_name
                base_actions[key] = partial(fn, **params)
            final_actions = {}
            final_actions = build_expanded_keys(
                base_actions,
                num_layers,
            )
            return final_actions

        def get_vison_parallel_split_mappings(num_layers: int):
            base_actions = {}
            for weight_name, is_column, extra in cls.weight_vison:
                params = {
                    "is_column": is_column,
                    **({extra.value: True} if extra else {}),
                }
                base_actions[weight_name] = partial(vision_fn, **params)
            final_actions = {}
            final_actions = build_expanded_keys(
                base_actions,
                num_layers,
            )
            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers,
            config.prefix_name,
        )
        vision_mappings = get_vison_parallel_split_mappings(config.vision_config.get("depth"))

        return {**mappings, **vision_mappings}
