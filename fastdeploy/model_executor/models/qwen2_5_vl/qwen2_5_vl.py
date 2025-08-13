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

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import ReplicatedLinear
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2Model
)
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.multimodal.registry import MultimodalRegistry
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        extract_text_token_output,
        text_image_gather_scatter,
        text_image_index_out,
    )

from fastdeploy.model_executor.forward_meta import ForwardMeta

''' 没必要再创建qwen2_5_VL*类了，直接用qwen2model即可 '''
# class Qwen2_5_VLMLP(Qwen2MLP):
#     pass

# class Qwen2_5_VLAttention(Qwen2Attention):
#     pass

# class Qwen2_5_VLDecoderLayer(Qwen2DecoderLayer):
#     pass

# @support_graph_optimization
# class Qwen2_5_VLModel(Qwen2Model):
#     pass


@MultimodalRegistry.register_model()
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
        # -----------  resampler_model ------------   qwen2_5_VL 没有这部分操作
        # self.resampler_model = self._init_resampler_model_model(fd_config.model_config)
        # -----------  language model -------------
        # 在 vllm 里也是直接用的 qwen2_model
        # 未知：需要保证 FD 的 qwen2 和 vllm 的 qwen2 完全对齐
        self.model = Qwen2Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        # self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

    def _init_vision_model(self, model_config) -> nn.Layer:
        from fastdeploy.model_executor.models.qwen2_5_vl.dfnrope.modeling import (
            DFNRopeVisionTransformerPretrainedModel,
        )

        visual = DFNRopeVisionTransformerPretrainedModel(model_config, prefix_name="vision_model")
        visual = paddle.amp.decorate(models=visual, level="O2", dtype="bfloat16")
        visual.eval()
        return visual

    @classmethod
    def name(self):
        return "Qwen2_5_VLForConditionalGeneration"

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
        # self.resampler_model.load_state_dict(state_dict)
        # if self.tie_word_embeddings:
        #     self.lm_head.linear.weight.set_value(self.ernie.embed_tokens.embeddings.weight.transpose([1, 0]))
        # else:
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.ernie.layers[i].mlp.text_fused_moe(fake_hidden_states)
            self.ernie.layers[i].mlp.image_fused_moe(fake_hidden_states)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.model(
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

    # 这一部分不知道是在干什么？？？？？？？？？？？-----------------------------------------------
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
        logger.info("erine inference model _get_tensor_parallel_mappings")
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
