"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import Dict

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.model_loader import ModelRegistry
from fastdeploy.model_executor.models.ernie4_5_moe import \
    Ernie4_5_MoeForCausalLM
from fastdeploy.model_executor.models.qwen2 import Qwen2PretrainedModel
from fastdeploy.model_executor.models.qwen3 import Qwen3PretrainedModel
from fastdeploy.model_executor.models.qwen3moe import Qwen3MoePretrainedModel

RL_MODEL_CLASSES = {
    "Ernie4_5_MoeForCausalLMRL": Ernie4_5_MoeForCausalLM,
    "Qwen2ForCausalLMRL": Qwen2PretrainedModel,
    "Qwen3ForCausalLMRL": Qwen3PretrainedModel,
    "Qwen3MoeForCausalLMRL": Qwen3MoePretrainedModel,
}


class RollOutModel(nn.Layer):
    """Main model class for rollout operations, supports multimodal components for train."""

    def __init__(self, fd_config: FDConfig):
        """Initialize with FastDeploy configuration."""
        super(RollOutModel, self).__init__()
        self.fd_config = fd_config
        self._init_models()

    def _init_models(self):
        """Initialize all model components including multimodal if needed."""
        self.is_vl = "VL" in self.fd_config.model_config.architectures[0]
        self.rollout_model = self._load_primary_model()
        self.rollout_models = [self.rollout_model]

        if self.is_vl:
            self._init_multimodal_models()
            self.rollout_models.extend(
                [self.vision_model, self.resampler_model])

    def _init_multimodal_models(self):
        """Initialize vision and resampler components for multimodal models."""
        # TODO:(gaoziyuan) Implement actual initialization
        self.vision_model = nn.Layer()
        self.resampler_model = nn.Layer()

    def _load_primary_model(self):
        """Load main model from loader based on config."""
        if "VL" in self.fd_config.model_config.architectures[0]:
            logger.error("Loaded Vision Language model, not support now")

        context = paddle.LazyGuard()
        architectures = f"{self.fd_config.model_config.architectures[0]}RL"
        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(self.fd_config)

        model.eval()
        return model

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Get parameter name mappings between rollout and training models."""
        mappings = {}
        for model in self.rollout_models:
            mappings.update(
                getattr(model, "get_name_mappings_to_training", lambda: {})())
        return mappings

    @paddle.no_grad()
    def state_dict(self):
        """state_dict"""
        all_params = {}
        for model in self.rollout_models:
            for name, param in model.state_dict().items():
                logger.debug(
                    f"Model param: {name}, shape={param.shape}, dtype={param.dtype}"
                )
                all_params[name] = param
        return all_params


class Ernie4_5_MoeForCausalLMRL(Ernie4_5_MoeForCausalLM):
    """
    Ernie4_5_MoeForCausalLMRL
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_MoeForCausalLMRL, self).__init__(fd_config)

    @classmethod
    def name(self):
        """name"""
        return "Ernie4_5_MoeForCausalLMRL"

    def get_name_mappings_to_training(self):
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        have_bias = self.fd_config.model_config.get("have_norm_bias", False)
        # Prepare placeholders
        place_holders = ["weight"] + (["bias"] if have_bias else [])

        # Initialize mapping dictionary
        infer_to_train = {}

        # Static mappings (non-layer specific)
        static_mappings = {
            "model.embeddings.word_embeddings.weight":
            "ernie.embed_tokens.weight",
            "model.norm.ln_weight": "ernie.norm.weight",
            "lm_head.out_linear.weight": "lm_head.weight"
        }
        if self.fd_config.model_config.get("weight_sharing", False):
            # Support tie_word_embeddings
            logger.debug("enable tie_word_embeddings")
            static_mappings.pop("lm_head.out_linear.weight")
        infer_to_train.update(static_mappings)
        infer_base_name = "model.hidden_layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx, is_moe_layer=False):
            # Handle special case for layer 0's input layernorm
            for ph in place_holders:
                infer_key = f"{infer_base_name}.{layer_idx}.input_layernorm.ln_{ph}"
                train_key = f"ernie.layers.{layer_idx}.input_layernorm.{ph}"
                infer_to_train[infer_key] = train_key

            # Common attention mappings
            for ph in place_holders:
                infer_to_train[f"{infer_base_name}.{layer_idx}.self_attn.qkv_proj.linear_{ph}"] = \
                    f"ernie.layers.{layer_idx}.self_attn.qkv_proj.{ph}"

                infer_to_train[f"{infer_base_name}.{layer_idx}.self_attn.o_proj.linear_{ph}"] = \
                    f"ernie.layers.{layer_idx}.self_attn.o_proj.{ph}"

            # Post-attention layernorm
            for ph in place_holders:
                infer_to_train[f"{infer_base_name}.{layer_idx}.post_attention_layernorm.ln_{ph}"] = \
                    f"ernie.layers.{layer_idx}.post_attention_layernorm.{ph}"

            if not is_moe_layer:
                # Dense FFN mappings
                for ph in place_holders:
                    infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.gate_up_proj.linear_{ph}"] = \
                        f"ernie.layers.{layer_idx}.mlp.up_gate_proj.{ph}"

                    infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.down_proj.linear_{ph}"] = \
                        f"ernie.layers.{layer_idx}.mlp.down_proj.{ph}"
            else:
                # MoE specific mappings
                infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.fused_moe.gate_weight"] = \
                    f"ernie.layers.{layer_idx}.mlp.gate.weight"

                if self.fd_config.moe_config.moe_use_aux_free:
                    infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.fused_moe.gate_correction_bias"] = \
                        f"ernie.layers.{layer_idx}.mlp.moe_statics.e_score_correction_bias"

                # Support shared experts
                if self.fd_config.model_config.get(
                        "moe_num_shared_experts") > 0:
                    infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.shared_experts.gate_up_proj.linear_weight"] = \
                        f"ernie.layers.{layer_idx}.mlp.shared_experts.up_gate_proj.weight"
                    infer_to_train[f"{infer_base_name}.{layer_idx}.mlp.shared_experts.down_proj.linear_weight"] = \
                        f"ernie.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"

                # MoE experts mappings
                for expert_idx in range(self.fd_config.moe_config.num_experts):
                    for ph in place_holders:
                        # FFN1 (up_gate_proj)
                        ffn1_key = f"{infer_base_name}.{layer_idx}.mlp.fused_moe.moe_ffn1_weight"
                        if ffn1_key not in infer_to_train:
                            infer_to_train[ffn1_key] = []
                        infer_to_train[ffn1_key].append(
                            f"ernie.layers.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                        )

                        # FFN2 (down_proj)
                        ffn2_key = f"{infer_base_name}.{layer_idx}.mlp.fused_moe.moe_ffn2_weight"
                        if ffn2_key not in infer_to_train:
                            infer_to_train[ffn2_key] = []
                        infer_to_train[ffn2_key].append(
                            f"ernie.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                        )

        # Process non-MoE layers
        for layer_idx in range(
                self.fd_config.moe_config.moe_layer_start_index):
            _add_layer_mappings(layer_idx, is_moe_layer=False)

        # Process MoE layers
        for layer_idx in range(self.fd_config.moe_config.moe_layer_start_index,
                               self.fd_config.model_config.num_layers):
            _add_layer_mappings(layer_idx, is_moe_layer=True)

        return infer_to_train
