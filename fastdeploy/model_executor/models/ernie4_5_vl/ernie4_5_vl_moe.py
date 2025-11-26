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

import inspect
import re
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
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    cuda_graph_buffers,
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import ReplicatedLinear
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.ernie4_5_moe import (
    Ernie4_5_Attention,
    Ernie4_5_MLP,
)
from fastdeploy.model_executor.models.ernie4_5_vl.image_op import (
    text_image_gather_scatter,
    text_image_index_out,
)
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)


class Ernie4_5_VLMLP(Ernie4_5_MLP):
    pass


class Ernie4_5_VLAttention(Ernie4_5_Attention):
    pass


@dataclass
class VLMoEMeta:
    image_input: paddle.Tensor
    text_input: paddle.Tensor
    text_index: paddle.Tensor
    image_index: paddle.Tensor
    token_type_ids: paddle.Tensor
    image_token_num: paddle.Tensor
    num_image_patch_id: paddle.Tensor

    def __str__(self):
        return (
            f"VLMoEMeta(\n"
            f"  image_input: {self.image_input}, pointer: {self.image_input.data_ptr()}\n"
            f"  text_input: {self.text_input}, pointer: {self.text_input.data_ptr()}\n"
            f"  text_index: {self.text_index}, pointer: {self.text_index.data_ptr()}\n"
            f"  image_index: {self.image_index}, pointer: {self.image_index.data_ptr()}\n"
            f"  token_type_ids: {self.token_type_ids}, pointer: {self.token_type_ids.data_ptr()}\n\n"
            f")"
        )


class Ernie4_5_VLMoeBlock(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        moe_tag: str,
        expert_id_offset: int,
        gate_correction_bias=None,
    ) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, "moe_quant_type"):
            if moe_tag == "Image" and hasattr(fd_config.quant_config, "image_moe_quant_type"):
                moe_quant_type = fd_config.quant_config.image_moe_quant_type
            else:
                moe_quant_type = fd_config.quant_config.moe_quant_type

        if moe_quant_type == "tensor_wise_fp8" or (
            moe_quant_type == "block_wise_fp8"
            and (fd_config.model_config.is_quantized or fd_config.model_config.is_moe_quantized)
        ):
            weight_key_map = {
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4a8" or moe_quant_type == "w4afp8":
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        else:
            # wint4/wint8/bfloat16
            weight_key_map = {
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            }
        moe_intermediate_size = (
            fd_config.model_config.moe_intermediate_size[0]
            if moe_tag == "Text"
            else fd_config.model_config.moe_intermediate_size[1]
        )
        num_experts = (
            fd_config.model_config.moe_num_experts[0]
            if moe_tag == "Text"
            else fd_config.model_config.moe_num_experts[1]
        )
        self.experts = FusedMoE(
            fd_config=fd_config,
            reduce_results=False,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            expert_id_offset=expert_id_offset,
            top_k=fd_config.model_config.moe_k,
            layer_idx=layer_id,
            moe_tag=moe_tag,
            weight_key_map=weight_key_map,
            gate_correction_bias=gate_correction_bias,
        )

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
            weight_key="weight" if moe_tag == "Text" else "weight_1",
            model_format="",
        )

    def forward(self, hidden_states: paddle.Tensor):
        out = self.experts(hidden_states, self.gate)
        return out

    def load_state_dict(self, state_dict):
        self.experts.load_state_dict(state_dict)
        self.gate.load_state_dict(state_dict)


class Ernie4_5_VLMoE(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        moe_layer_start_index = fd_config.model_config.moe_layer_start_index
        if isinstance(moe_layer_start_index, int):
            text_moe_layer_start_index = moe_layer_start_index
            image_moe_layer_start_index = moe_layer_start_index
        else:
            text_moe_layer_start_index = moe_layer_start_index[0]
            image_moe_layer_start_index = moe_layer_start_index[1]

        moe_layer_end_index = fd_config.model_config.moe_layer_end_index
        if moe_layer_end_index is None:
            text_moe_layer_end_index = fd_config.model_config.num_hidden_layers
            image_moe_layer_end_index = fd_config.model_config.num_hidden_layers
        elif isinstance(moe_layer_end_index, int):
            text_moe_layer_end_index = moe_layer_end_index
            image_moe_layer_end_index = moe_layer_end_index
        else:
            text_moe_layer_end_index = moe_layer_end_index[0]
            image_moe_layer_end_index = moe_layer_end_index[1]

        assert text_moe_layer_start_index <= text_moe_layer_end_index
        if fd_config.model_config.moe_use_aux_free:
            self.gate_correction_bias = self.create_parameter(
                shape=[2, fd_config.model_config.moe_num_experts[0]],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            if not self.gate_correction_bias._is_initialized():
                self.gate_correction_bias.initialize()
        else:
            self.gate_correction_bias = None

        if layer_id >= text_moe_layer_start_index and layer_id <= text_moe_layer_end_index:
            self.text_fused_moe = Ernie4_5_VLMoeBlock(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}",
                moe_tag="Text",
                expert_id_offset=0,
                gate_correction_bias=self.gate_correction_bias[0] if fd_config.model_config.moe_use_aux_free else None,
            )
        else:
            self.text_fused_moe = Ernie4_5_VLMLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}",
                reduce_results=False,
            )

        assert image_moe_layer_start_index <= image_moe_layer_end_index
        if layer_id >= image_moe_layer_start_index and layer_id <= image_moe_layer_end_index:
            self.image_fused_moe = Ernie4_5_VLMoeBlock(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}",
                moe_tag="Image",
                expert_id_offset=fd_config.model_config.moe_num_experts[0],
                gate_correction_bias=self.gate_correction_bias[1] if fd_config.model_config.moe_use_aux_free else None,
            )
        else:
            self.image_fused_moe = Ernie4_5_VLMLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}",
                reduce_results=False,
            )

        self.num_shared_experts = fd_config.model_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            self.shared_experts = Ernie4_5_VLMLP(
                fd_config=fd_config,
                intermediate_size=self.num_shared_experts * fd_config.model_config.moe_intermediate_size[0],
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )

    def load_state_dict(self, state_dict):
        if self.gate_correction_bias is not None:
            gate_correction_bias_tensor = state_dict.pop(self.text_fused_moe.experts.gate_correction_bias_key)
            if self.gate_correction_bias.shape != gate_correction_bias_tensor.shape:
                gate_correction_bias_tensor = gate_correction_bias_tensor.reshape(self.gate_correction_bias.shape)
            self.gate_correction_bias.set_value(gate_correction_bias_tensor)
        self.text_fused_moe.load_state_dict(state_dict)
        self.image_fused_moe.load_state_dict(state_dict)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor, vl_moe_meta: VLMoEMeta):
        if self.num_shared_experts > 0:
            shared_experts_out = self.shared_experts(hidden_states)
        hidden_states, text_input, image_input = text_image_gather_scatter(
            hidden_states,
            vl_moe_meta.text_input,
            vl_moe_meta.image_input,
            vl_moe_meta.token_type_ids,
            vl_moe_meta.text_index,
            vl_moe_meta.image_index,
            True,
        )
        text_out = self.text_fused_moe(text_input)
        image_out = self.image_fused_moe(image_input)
        hidden_states, _, _ = text_image_gather_scatter(
            hidden_states,
            text_out,
            image_out,
            vl_moe_meta.token_type_ids,
            vl_moe_meta.text_index,
            vl_moe_meta.image_index,
            False,
        )
        if self.num_shared_experts > 0:
            hidden_states += shared_experts_out
        if self.tp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states


class Ernie4_5_VLDecoderLayer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        moe_layer_start_index = fd_config.model_config.moe_layer_start_index
        if isinstance(moe_layer_start_index, list):
            min_moe_layer_start_index = min(moe_layer_start_index)
        else:
            min_moe_layer_start_index = moe_layer_start_index

        max_moe_layer_end_index = fd_config.model_config.num_hidden_layers
        if fd_config.model_config.moe_layer_end_index is not None:
            moe_layer_end_index = fd_config.model_config.moe_layer_end_index
            if isinstance(moe_layer_start_index, list):
                max_moe_layer_end_index = max(moe_layer_end_index)
            else:
                max_moe_layer_end_index = moe_layer_end_index

        self.self_attn = Ernie4_5_VLAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        assert min_moe_layer_start_index <= max_moe_layer_end_index

        if (
            fd_config.model_config.moe_num_experts is not None
            and layer_id >= min_moe_layer_start_index
            and layer_id <= max_moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoE(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie4_5_VLMLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
        vl_moe_meta: VLMoEMeta = None,
    ):
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if isinstance(self.mlp, Ernie4_5_VLMoE):
            hidden_states = self.mlp(hidden_states, vl_moe_meta)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@cuda_graph_buffers(
    {
        "text_input": {
            "shape": ["model_config.max_model_len", "model_config.hidden_size"],
            "dtype": "model_config.dtype",
            "value": 1,
        },
        "image_input": {
            "shape": ["model_config.max_model_len", "model_config.hidden_size"],
            "dtype": "model_config.dtype",
            "value": 1,
        },
        "text_index": {
            "shape": ["model_config.max_model_len"],
            "dtype": "int32",
            "value": 0,
        },
        "image_index": {
            "shape": ["model_config.max_model_len"],
            "dtype": "int32",
            "value": 0,
        },
        "token_type_ids": {
            "shape": ["model_config.max_model_len"],
            "dtype": "int32",
            "value": -1,
        },
        "image_token_num": {
            "shape": [1],
            "dtype": "int64",
            "value": 0,
        },
    }
)
@support_graph_optimization
class Ernie4_5_VLModel(nn.Layer):
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
        self.im_patch_id = fd_config.model_config.im_patch_id
        self._dtype = fd_config.model_config.dtype
        fd_config.model_config.pretrained_config.prefix_name = "ernie"
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
                Ernie4_5_VLDecoderLayer(
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

    def prepare_vl_moe_meta(
        self,
        ids_remove_padding: paddle.Tensor,
    ) -> VLMoEMeta:

        image_mask = ids_remove_padding >= self.im_patch_id
        token_type_ids = image_mask.cast("int32")
        image_token_num = image_mask.sum()
        token_num = ids_remove_padding.shape[0]
        text_token_num = paddle.maximum((token_num - image_token_num), paddle.ones([], dtype="int64"))
        num_image_patch_id = ids_remove_padding == self.im_patch_id
        num_image_patch_id = num_image_patch_id.cast("int32").sum()

        # The scenario requiring padding is CUDA graph, thus we only need to pad the maximum capture size.
        self._cuda_graph_buffers["token_type_ids"][: self.fd_config.graph_opt_config.max_capture_size].fill_(-1)
        self._cuda_graph_buffers["token_type_ids"].copy_(token_type_ids, False)
        self._cuda_graph_buffers["image_token_num"].copy_(image_token_num, False)

        return VLMoEMeta(
            text_input=self._cuda_graph_buffers["text_input"][:text_token_num],
            image_input=self._cuda_graph_buffers["image_input"][:image_token_num],
            text_index=self._cuda_graph_buffers["text_index"][:token_num],
            image_index=self._cuda_graph_buffers["image_index"][:token_num],
            token_type_ids=self._cuda_graph_buffers["token_type_ids"][:token_num],
            image_token_num=self._cuda_graph_buffers["image_token_num"],
            num_image_patch_id=num_image_patch_id,
        )

    def get_input_embeddings(self, ids_remove_padding: paddle.Tensor) -> paddle.Tensor:
        return self.embed_tokens(ids_remove_padding=ids_remove_padding)

    def forward(
        self,
        input_embeddings: paddle.Tensor,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        vl_moe_meta: VLMoEMeta,
    ):
        text_image_index_out(vl_moe_meta.token_type_ids, vl_moe_meta.text_index, vl_moe_meta.image_index)

        hidden_states = input_embeddings
        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
                vl_moe_meta,
            )

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_VLMoeForConditionalGeneration",
    module_name="ernie4_5_vl.ernie4_5_vl_moe",
    category=ModelCategory.MULTIMODAL,
    primary_use=ModelCategory.MULTIMODAL,
)
class Ernie4_5_VLMoeForConditionalGeneration(ModelForCasualLM):
    """
    Ernie4_5_VLMoeForConditionalGeneration
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_VLMoeForConditionalGeneration, self).__init__(fd_config)
        # ----------- vision model ------------
        self.vision_model = self._init_vision_model(fd_config.model_config)
        # -----------  resampler_model ------------
        self.resampler_model = self._init_resampler_model_model(fd_config.model_config)
        # ernie
        self.ernie = Ernie4_5_VLModel(fd_config=fd_config)

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
        from fastdeploy.model_executor.models.ernie4_5_vl.dfnrope.modeling import (
            DFNRopeVisionTransformerPretrainedModel,
        )

        vision_model = DFNRopeVisionTransformerPretrainedModel(model_config, prefix_name="vision_model")
        vision_model = paddle.amp.decorate(models=vision_model, level="O2", dtype="bfloat16")
        vision_model.eval()
        return vision_model

    def _init_resampler_model_model(self, model_config) -> nn.Layer:
        from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import (
            VariableResolutionResamplerModel,
        )

        resampler_model = VariableResolutionResamplerModel(
            model_config.vision_config.hidden_size,
            model_config.hidden_size,
            model_config.spatial_conv_size,
            model_config.temporal_conv_size,
            config=model_config,
            prefix_name="resampler_model",
        )
        resampler_model = paddle.amp.decorate(models=resampler_model, level="O2", dtype="bfloat16")
        resampler_model.eval()
        return resampler_model

    @classmethod
    def name(self):
        return "Ernie4_5_VLMoeForConditionalGeneration"

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

        general_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("embed_tokens.embeddings", "embed_tokens", None, None),
            ("lm_head.linear", "lm_head", None, None),
            ("mlp.image_fused_moe.gate.weight", "mlp.gate.weight_1", None, "gate"),
            ("mlp.text_fused_moe.gate.weight", "mlp.gate.weight", None, "gate"),
            ("resampler_model", "ernie.resampler_model", None, None),
            ("vision_model", "ernie.vision_model", None, None),
            ("gate_correction_bias", "moe_statics.e_score_correction_bias", None, None),
            ("attn.cache_k_scale", "cachek_matmul.activation_scale", None, None),
            ("attn.cache_v_scale", "cachev_matmul.activation_scale", None, None),
            ("attn.cache_k_zp", "cachek_matmul.activation_zero_point", None, None),
            ("attn.cache_v_zp", "cachev_matmul.activation_zero_point", None, None),
            # for torch model
            ("resampler_model", "model.resampler_model", None, None),
            ("qkv_proj", "q_proj", None, "q"),
            ("qkv_proj", "k_proj", None, "k"),
            ("qkv_proj", "v_proj", None, "v"),
            ("up_gate_proj", "gate_proj", None, "gate"),
            ("up_gate_proj", "up_proj", None, "up"),
        ]

        text_expert_params_mapping = []
        if getattr(self.fd_config.model_config, "moe_num_experts", None) is not None:
            text_expert_params_mapping = FusedMoE.make_expert_params_mapping(
                num_experts=self.fd_config.model_config.moe_num_experts[0],
                ckpt_down_proj_name="down_proj",
                ckpt_gate_up_proj_name="up_gate_proj",
                ckpt_gate_proj_name="gate_proj",
                ckpt_up_proj_name="up_proj",
                param_gate_up_proj_name="text_fused_moe.experts.up_gate_proj_",
                param_down_proj_name="text_fused_moe.experts.down_proj_",
            )
            image_expert_params_mapping = FusedMoE.make_expert_params_mapping(
                num_experts=self.fd_config.model_config.moe_num_experts[1],
                ckpt_down_proj_name="down_proj",
                ckpt_gate_up_proj_name="up_gate_proj",
                ckpt_gate_proj_name="gate_proj",
                ckpt_up_proj_name="up_proj",
                param_gate_up_proj_name="image_fused_moe.experts.up_gate_proj_",
                param_down_proj_name="image_fused_moe.experts.down_proj_",
                experts_offset=self.fd_config.model_config.moe_num_experts[0],
            )

        all_param_mapping = general_params_mapping + text_expert_params_mapping + image_expert_params_mapping

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        expert_id = None
        shard_id = None
        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded_weight_name = (
                self.process_weights_before_loading_fn(loaded_weight_name)
                if getattr(self, "process_weights_before_loading_fn", None)
                else loaded_weight_name
            )
            if loaded_weight_name is None:
                continue
            for param_name, weight_name, exp_id, shard_id in all_param_mapping:
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name.startswith("model.") and self.fd_config.model_config.model_format == "torch":
                    model_param_name = model_param_name.replace("model.", "ernie.")

                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                expert_id = exp_id
                shard_id = shard_id
                break
            else:
                if loaded_weight_name not in params_dict.keys():
                    continue
                model_param_name = loaded_weight_name
                param = params_dict[model_param_name]

            # Get weight loader from parameter and set weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            sig = inspect.signature(weight_loader)

            if "expert_id" in sig.parameters:
                weight_loader(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            else:
                weight_loader(param, loaded_weight, shard_id)
            model_sublayer_name = re.sub(
                r"\.(up_gate_proj_weight|down_proj_weight|weight|cache_k_scale|cache_v_scale)$", "", model_param_name
            )
            process_weights_after_loading_fn(model_sublayer_name, param)
        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(self.ernie.embed_tokens.embeddings.weight.transpose([1, 0]))

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.ernie.load_state_dict(state_dict)
        self.vision_model.load_state_dict(state_dict)
        self.resampler_model.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.ernie.embed_tokens.embeddings.weight})
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
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

    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_token_num: int,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        input_embeddings = self.ernie.get_input_embeddings(ids_remove_padding=ids_remove_padding)
        if image_token_num > 0:
            input_embeddings[ids_remove_padding == self.ernie.im_patch_id] = image_features.cast(self.ernie._dtype)
        return input_embeddings

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: ForwardMeta,
    ):
        vl_moe_meta = self.ernie.prepare_vl_moe_meta(ids_remove_padding=ids_remove_padding)
        input_embeddings = self.get_input_embeddings(
            ids_remove_padding=ids_remove_padding,
            image_features=image_features,
            image_token_num=vl_moe_meta.num_image_patch_id.item(),
        )

        if forward_meta.step_use_cudagraph:
            self._decoder_input_embeddings.copy_(input_embeddings, False)
            input_embeddings = self._decoder_input_embeddings

        hidden_states = self.ernie(
            input_embeddings=input_embeddings,
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            vl_moe_meta=vl_moe_meta,
        )

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.ernie.clear_grpah_opt_backend(fd_config=self.fd_config)


class Ernie4_5_VLPretrainedModel(PretrainedModel):
    """
    Ernie4_5_MoePretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "Ernie4_5_VLMoeForConditionalGeneration"

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
        # quant tensorwise
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.quant_weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.TEXT_EXPERT_ID}}}.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.TEXT_EXPERT_ID}}}.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.IMG_EXPERT_ID}}}.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.IMG_EXPERT_ID}}}.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.quant_weight",
            False,
        ),
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
