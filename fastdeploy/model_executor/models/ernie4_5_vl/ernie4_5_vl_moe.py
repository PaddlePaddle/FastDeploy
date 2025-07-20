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
from fastdeploy.distributed.communication_op import tensor_model_parallel_all_reduce
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.ernie4_5_moe import (
    Ernie4_5_Attention,
    Ernie4_5_MLP,
)
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() and not current_platform.is_dcu():
    from fastdeploy.model_executor.ops.gpu import (
        extract_text_token_output,
        text_image_gather_scatter,
        text_image_index_out,
    )

from fastdeploy.model_executor.forward_meta import ForwardMeta


class Ernie4_5_VLMLP(Ernie4_5_MLP):
    pass


class Ernie4_5_VLAttention(Ernie4_5_Attention):
    pass


@dataclass
class VLMoEMeta:
    image_input: Optional[paddle.Tensor] = None
    text_input: Optional[paddle.Tensor] = None
    text_index: Optional[paddle.Tensor] = None
    image_index: Optional[paddle.Tensor] = None
    token_type_ids: Optional[paddle.Tensor] = None


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

        moe_quant_type = ""
        if hasattr(fd_config, "quant_config") and fd_config.quant_config is not None:
            moe_quant_type = getattr(fd_config.quant_config, "name", lambda: "")()

        if layer_id >= text_moe_layer_start_index and layer_id <= text_moe_layer_end_index:
            if moe_quant_type == "tensor_wise_fp8" or (
                moe_quant_type == "block_wise_fp8" and fd_config.model_config.is_quantized
            ):
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
                weight_key_map = {
                    "gate_weight_key": f"{prefix}.gate.weight",
                    "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                    "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                    "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
                }
            self.text_fused_moe = FusedMoE(
                fd_config=fd_config,
                reduce_results=False,
                moe_intermediate_size=fd_config.model_config.moe_intermediate_size[0],
                num_experts=fd_config.model_config.moe_num_experts[0],
                expert_id_offset=0,
                top_k=fd_config.model_config.moe_k,
                layer_idx=layer_id,
                moe_tag="Text",
                weight_key_map=weight_key_map,
            )
            self.text_fused_moe.extract_gate_correction_bias = self.extract_gate_correction_bias_text
        else:
            self.text_fused_moe = Ernie4_5_VLMLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}",
                reduce_results=False,
            )

        assert image_moe_layer_start_index <= image_moe_layer_end_index
        if layer_id >= image_moe_layer_start_index and layer_id <= image_moe_layer_end_index:
            if moe_quant_type == "tensor_wise_fp8" or (
                moe_quant_type == "block_wise_fp8" and fd_config.model_config.is_quantized
            ):
                weight_key_map = {
                    "gate_weight_key": f"{prefix}.gate.weight_1",
                    "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                    "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                    "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                    "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                    "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                    "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                    "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
                }
            else:
                weight_key_map = {
                    "gate_weight_key": f"{prefix}.gate.weight_1",
                    "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                    "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                    "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
                }
            self.image_fused_moe = FusedMoE(
                fd_config=fd_config,
                reduce_results=False,
                moe_intermediate_size=fd_config.model_config.moe_intermediate_size[1],
                num_experts=fd_config.model_config.moe_num_experts[1],
                expert_id_offset=fd_config.model_config.moe_num_experts[0],
                top_k=fd_config.model_config.moe_k,
                layer_idx=layer_id,
                moe_tag="Image",
                weight_key_map=weight_key_map,
            )
            self.image_fused_moe.extract_gate_correction_bias = self.extract_gate_correction_bias_image
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

    def extract_gate_correction_bias_text(self, gate_correction_bias_key, state_dict):
        """
        extract_gate_correction_bias function.
        """
        gate_correction_bias_tensor = get_tensor(state_dict[gate_correction_bias_key]).astype("float32")
        return gate_correction_bias_tensor[0].unsqueeze(0)

    def extract_gate_correction_bias_image(self, gate_correction_bias_key, state_dict):
        """
        extract_gate_correction_bias function.
        """
        gate_correction_bias_tensor = get_tensor(state_dict[gate_correction_bias_key]).astype("float32")
        return gate_correction_bias_tensor[1].unsqueeze(0)

    def load_state_dict(self, state_dict):
        self.text_fused_moe.load_state_dict(state_dict)
        self.image_fused_moe.load_state_dict(state_dict)
        if self.text_fused_moe.moe_use_gate_correction_bias:
            state_dict.pop(self.text_fused_moe.gate_correction_bias_key)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor, vl_moe_meta: VLMoEMeta):
        if self.num_shared_experts > 0:
            shared_experts_out = self.shared_experts(hidden_states)
        if vl_moe_meta.image_input is not None:
            text_image_gather_scatter(
                hidden_states,
                vl_moe_meta.text_input,
                vl_moe_meta.image_input,
                vl_moe_meta.token_type_ids,
                vl_moe_meta.text_index,
                vl_moe_meta.image_index,
                True,
            )
            text_out = self.text_fused_moe(vl_moe_meta.text_input)
            image_out = self.image_fused_moe(vl_moe_meta.image_input)
            text_image_gather_scatter(
                hidden_states,
                text_out,
                image_out,
                vl_moe_meta.token_type_ids,
                vl_moe_meta.text_index,
                vl_moe_meta.image_index,
                False,
            )
        else:
            hidden_states = self.text_fused_moe(hidden_states)
        if self.num_shared_experts > 0:
            hidden_states += shared_experts_out
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(hidden_states)
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
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

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

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        text_input = None
        image_input = None
        text_index = None
        image_index = None
        image_token_num = 0

        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        # -----------------------
        image_mask = ids_remove_padding == self.im_patch_id
        token_type_ids = image_mask.cast("int32")
        token_num = hidden_states.shape[0]
        image_token_num = paddle.count_nonzero(token_type_ids)
        text_token_num = paddle.maximum((token_num - image_token_num), paddle.ones([], dtype="int64"))
        if image_mask.any():
            hidden_states[image_mask] = image_features.cast(self._dtype)
            text_input = paddle.full(
                shape=[text_token_num, hidden_states.shape[1]],
                fill_value=1,
                dtype=self._dtype,
            )
            image_input = paddle.full(
                shape=[image_token_num, hidden_states.shape[1]],
                fill_value=1,
                dtype=self._dtype,
            )
            text_index = paddle.zeros_like(token_type_ids)
            image_index = paddle.zeros_like(token_type_ids)
            text_image_index_out(token_type_ids, text_index, image_index)

        vl_moe_meta = VLMoEMeta(
            text_input=text_input,
            image_input=image_input,
            text_index=text_index,
            image_index=image_index,
            token_type_ids=token_type_ids,
        )
        # -----------------------

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
                vl_moe_meta,
            )

        hidden_states = hidden_states + residual

        # -----------------------
        hidden_states = hidden_states.cast("float32")
        score_text = hidden_states

        if image_input is not None:
            token_type_ids = token_type_ids.reshape([-1])
            text_pos_shifted = token_type_ids[:token_num] == 0
            score_text = hidden_states[text_pos_shifted.reshape([-1])]
        max_seq_len, max_seq_len_index = paddle.topk(forward_meta.seq_lens_this_time.squeeze(-1), k=1)
        hidden_states = extract_text_token_output(
            max_seq_len,
            max_seq_len_index.cast("int32"),
            image_token_num.cast("int32"),
            forward_meta.seq_lens_this_time,
            forward_meta.cu_seqlens_q,
            score_text,
        ).cast(self._dtype)
        # -----------------------

        out = self.norm(hidden_states)

        return out


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
            self.lm_head.linear.weight.set_value(self.ernie.embed_tokens.embeddings.weight.transpose([1, 0]))
        else:
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

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.ernie(
            ids_remove_padding=ids_remove_padding,
            image_features=image_features,
            forward_meta=forward_meta,
        )

        return hidden_states


class Ernie4_5_VLPretrainedModel(PretrainedModel):
    """
    Ernie4_5_PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

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
