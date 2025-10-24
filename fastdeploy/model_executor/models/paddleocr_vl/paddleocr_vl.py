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

import re
from functools import partial
from typing import Dict, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_DecoderLayer
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)

from .projector import Projector
from .siglip import SiglipVisionModel


@support_graph_optimization
class PaddleOCRVLModel(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        super().__init__()

        self.config = fd_config.model_config
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"
        self._dtype = fd_config.model_config.torch_dtype

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=self._dtype,
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Ernie4_5_DecoderLayer(
                    fd_config=fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )
        for i, layer in enumerate(self.layers):
            layer.self_attn.attn = Attention(
                fd_config=fd_config,
                layer_id=i,
                prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}.self_attn",
                use_neox_rotary_style=True,
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
        forward_meta: ForwardMeta,
    ):
        hidden_states = input_embeddings
        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        hidden_states = hidden_states + residual

        out = self.norm(hidden_states)

        return out


@ModelRegistry.register_model_class(
    architecture="PaddleOCRVLForConditionalGeneration",
    module_name="paddleocr_vl.paddleocr_vl",
    category=ModelCategory.MULTIMODAL,
    primary_use=ModelCategory.MULTIMODAL,
)
class PaddleOCRVLForConditionalGeneration(ModelForCasualLM):
    def __init__(self, fd_config):
        super().__init__(fd_config)

        config = fd_config.model_config
        self.config = config
        self.mlp_AR = Projector(config, config.vision_config, prefix="mlp_AR")
        self.visual = SiglipVisionModel(config.vision_config, prefix="visual")
        self.model = PaddleOCRVLModel(fd_config)
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        # Persistent buffers for CUDA graphs.
        if envs.FD_ENABLE_MAX_PREFILL:
            max_length = fd_config.scheduler_config.max_num_seqs * fd_config.model_config.max_model_len
        else:
            max_length = fd_config.model_config.max_model_len
        self._input_embeddings = paddle.zeros(
            [max_length, fd_config.model_config.hidden_size],
            dtype=fd_config.model_config.dtype,
        )

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.

        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))
        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded_weight_name = (
                self.process_weights_before_loading_fn(loaded_weight_name)
                if getattr(self, "process_weights_before_loading_fn", None)
                else loaded_weight_name
            )
            if loaded_weight_name is None:
                continue
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
        self.projector.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    @property
    def projector(self):
        return self.mlp_AR

    @classmethod
    def name(self):
        return "PaddleOCRVLForConditionalGeneration"

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.vocab_size :] = -float("inf")

        return logits

    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_embeddings = self.model.get_input_embeddings(ids_remove_padding=ids_remove_padding)
        image_mask = ids_remove_padding == self.model.config.image_token_id
        image_token_num = image_mask.sum()

        if image_token_num > 0:
            input_embeddings[image_mask] = image_features.cast(self._dtype)
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
        self._input_embeddings.copy_(input_embeddings, False)

        hidden_states = self.model(
            input_embeddings=self._input_embeddings,
            forward_meta=forward_meta,
        )

        return hidden_states


class PaddleOCRVLPretrainedModel(PretrainedModel):

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "PaddleOCRVLForConditionalGeneration"

    from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
    from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
    from fastdeploy.model_executor.models.utils import WeightMeta

    weight_infos = [
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.TEXT_EXPERT_ID}}}.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.TEXT_EXPERT_ID}}}.down_proj.weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.IMG_EXPERT_ID}}}.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.IMG_EXPERT_ID}}}.down_proj.weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.weight",
            False,
        ),
        WeightMeta(".embed_tokens.weight", False),
        WeightMeta("lm_head.weight", True),
    ]

    weight_vison = [
        # resampler_model
        WeightMeta("ernie.resampler_model.spatial_linear.0.weight", False),
        WeightMeta("resampler_model.spatial_linear.0.weight", False),
        # vision
        WeightMeta(
            f"vision_model.blocks.{{{layerid.LAYER_ID}}}.attn.proj.weight",
            False,
        ),
        WeightMeta(f"vision_model.blocks.{{{layerid.LAYER_ID}}}.mlp.fc2.weight", False),
        WeightMeta(f"vision_model.blocks.{{{layerid.LAYER_ID}}}.mlp.fc1.weight", True),
        WeightMeta(f"vision_model.blocks.{{{layerid.LAYER_ID}}}.mlp.fc1.bias", True),
        WeightMeta(
            f"vision_model.blocks.{{{layerid.LAYER_ID}}}.attn.qkv.weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(
            f"vision_model.blocks.{{{layerid.LAYER_ID}}}.attn.qkv.bias",
            True,
            tsm.GQA,
        ),
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
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
            moe_num_experts: list[int],
            moe_layer_start_index: int,
            prefix_name: str,
        ):
            base_actions = {}
            for weight_name, is_column, extra in cls.weight_infos:
                params = {
                    "is_column": is_column,
                    **({extra.value: True} if extra else {}),
                }

                if "lm_head.weight" in weight_name or weight_name == "":
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
                (moe_layer_start_index if moe_layer_start_index > 0 else num_layers),
                text_num_experts=moe_num_experts[0],
                img_num_experts=moe_num_experts[1],
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

        moe_layer_start_index = -1
        if isinstance(config.moe_layer_start_index, list):
            moe_layer_start_index = min(config.moe_layer_start_index)
        elif isinstance(config.moe_layer_start_index, int):
            moe_layer_start_index = config.moe_layer_start_index

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers,
            config.moe_num_experts,
            moe_layer_start_index,
            config.prefix_name,
        )
        vision_mappings = get_vison_parallel_split_mappings(config.vision_config.get("depth"))

        return {**mappings, **vision_mappings}
