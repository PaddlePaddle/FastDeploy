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
from fastdeploy.model_executor.models.ernie4_5_vl.ernie4_5_vl_moe import \
    Ernie4_5_VLMoeForConditionalGeneration
from fastdeploy.model_executor.models.qwen2 import Qwen2ForCausalLM
from fastdeploy.model_executor.models.qwen3 import Qwen3ForCausalLM
from fastdeploy.model_executor.models.qwen3moe import Qwen3MoeForCausalLM
from fastdeploy.rl.rollout_config import RolloutModelConfig


class RolloutModel(nn.Layer):
    """Main model class for rollout operations, supports multimodal components for train."""

    def __init__(self, rollout_model_config: RolloutModelConfig):
        """Initialize with FastDeploy configuration."""
        super(RolloutModel, self).__init__()
        self.fd_config = rollout_model_config.initialize()
        self.rollout_model = self._init_model()

    def _init_model(self) -> nn.Layer:
        """Load model from loader based on config."""
        context = paddle.LazyGuard()
        architectures = f"{self.fd_config.model_config.architectures[0]}RL"
        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(self.fd_config)
        model.eval()
        return model

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Get parameter name mappings between rollout and training models."""
        return getattr(self.rollout_model, "get_name_mappings_to_training", lambda: {})()

    @paddle.no_grad()
    def state_dict(self):
        """state_dict"""
        return self.rollout_model.state_dict()


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
    def name(self) -> str:
        """name"""
        return "Ernie4_5_MoeForCausalLMRL"

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        infer_to_train = {}

        base_name = "ernie"
        # Static mappings (non-layer specific)
        static_mappings = {
            f"{base_name}.embed_tokens.embeddings.weight":
            f"{base_name}.embed_tokens.weight",
            "lm_head.linear.weight": "lm_head.weight"
        }
        if getattr(self.fd_config.model_config, "tie_word_embeddings", False):
            # Support tie_word_embeddings
            logger.debug("enable tie_word_embeddings")
            static_mappings.pop("lm_head.linear.weight")
        infer_to_train.update(static_mappings)

        base_name = base_name + ".layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int):
            # MoE specific mappings
            infer_to_train[f"{base_name}.{layer_idx}.mlp.fused_moe.gate_weight"] = \
                f"{base_name}.{layer_idx}.mlp.gate.weight"

            if self.fd_config.model_config.moe_use_aux_free:
                infer_to_train[f"{base_name}.{layer_idx}.mlp.fused_moe.gate_correction_bias"] = \
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"

            # MoE experts mappings
            for expert_idx in range(self.fd_config.model_config.moe_num_experts):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.fused_moe.up_gate_proj_weight"
                    if up_gate_proj_key not in infer_to_train:
                        infer_to_train[up_gate_proj_key] = []
                    infer_to_train[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.fused_moe.down_proj_weight"
                    if down_proj_key not in infer_to_train:
                        infer_to_train[down_proj_key] = []
                    infer_to_train[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        assert isinstance(self.fd_config.model_config.moe_layer_start_index, int)
        # Process MoE layers
        for layer_idx in range(self.fd_config.model_config.moe_layer_start_index,
                               self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        return infer_to_train


class Ernie4_5_VLMoeForConditionalGenerationRL(Ernie4_5_VLMoeForConditionalGeneration):
    """
    Ernie4_5_VLMoeForConditionalGenerationRL
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_VLMoeForConditionalGenerationRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Ernie4_5_VLMoeForConditionalGenerationRL"

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        infer_to_train = {}

        base_name = "ernie"
        # Static mappings (non-layer specific)
        static_mappings = {
            f"{base_name}.embed_tokens.embeddings.weight":
            f"{base_name}.embed_tokens.weight",
            "lm_head.linear.weight": "lm_head.weight"
        }
        if getattr(self.fd_config.model_config, "tie_word_embeddings", False):
            # Support tie_word_embeddings
            logger.debug("enable tie_word_embeddings")
            static_mappings.pop("lm_head.linear.weight")
        infer_to_train.update(static_mappings)

        base_name = base_name + ".layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int, moe_tag: str):
            # MoE specific mappings
            infer_to_train[f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.gate_weight"] = f"{base_name}.{layer_idx}.mlp.gate.weight" if moe_tag == "text" else f"{base_name}.{layer_idx}.mlp.gate.weight_1"

            if self.fd_config.model_config.moe_use_aux_free:
                infer_to_train[f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.gate_correction_bias"] = \
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"

            # MoE experts mappings
            assert isinstance(self.fd_config.model_config.moe_num_experts, list)
            if moe_tag == "text":
                expert_idx_start = 0
                expert_idx_end = self.fd_config.model_config.moe_num_experts[0]
            else:
                expert_idx_start = self.fd_config.model_config.moe_num_experts[0]
                expert_idx_end = self.fd_config.model_config.moe_num_experts[1]

            for expert_idx in range(expert_idx_start, expert_idx_end):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.up_gate_proj_weight"
                    if up_gate_proj_key not in infer_to_train:
                        infer_to_train[up_gate_proj_key] = []
                    infer_to_train[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.down_proj_weight"
                    if down_proj_key not in infer_to_train:
                        infer_to_train[down_proj_key] = []
                    infer_to_train[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        moe_layer_start_index = self.fd_config.model_config.moe_layer_start_index
        if isinstance(moe_layer_start_index, int):
            text_moe_layer_start_index = moe_layer_start_index
            image_moe_layer_start_index = moe_layer_start_index
        else:
            text_moe_layer_start_index = moe_layer_start_index[0]
            image_moe_layer_start_index = moe_layer_start_index[1]

        moe_layer_end_index = self.fd_config.model_config.moe_layer_end_index
        if moe_layer_end_index is None:
            text_moe_layer_end_index = self.fd_config.model_config.num_hidden_layers
            image_moe_layer_end_index = self.fd_config.model_config.num_hidden_layers
        elif isinstance(moe_layer_end_index, int):
            text_moe_layer_end_index = moe_layer_end_index
            image_moe_layer_end_index = moe_layer_end_index
        else:
            text_moe_layer_end_index = moe_layer_end_index[0]
            image_moe_layer_end_index = moe_layer_end_index[1]
        # Process MoE layers
        for layer_idx in range(text_moe_layer_start_index, text_moe_layer_end_index):
            _add_layer_mappings(layer_idx, "text")
        for layer_idx in range(image_moe_layer_start_index, image_moe_layer_end_index):
            _add_layer_mappings(layer_idx, "image")

        return infer_to_train


class Qwen2ForCausalLMRL(Qwen2ForCausalLM):
    """
    Qwen2ForCausalLMRL
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen2ForCausalLMRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Qwen2ForCausalLMRL"

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        infer_to_train = {}

        base_name = "qwen2"
        # Static mappings (non-layer specific)
        static_mappings = {
            f"{base_name}.embed_tokens.embeddings.weight":
            f"{base_name}.embed_tokens.weight",
            "lm_head.linear.weight": "lm_head.weight"
        }
        infer_to_train.update(static_mappings)

        base_name = base_name + ".layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx):
            # FFN mappings
            for ph in place_holders:
                infer_to_train[f"{base_name}.{layer_idx}.mlp.up_gate_proj.{ph}"] = \
                    f"{base_name}.{layer_idx}.mlp.gate_up_fused_proj.{ph}"

        for layer_idx in range(
                self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        return infer_to_train


class Qwen3MoeForCausalLMRL(Qwen3MoeForCausalLM):
    """
    Qwen3MoeForCausalLMRL
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen3MoeForCausalLMRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Qwen3MoeForCausalLMRL"

    def get_name_mappings_to_training(self) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        infer_to_train = {}

        base_name = "model"
        # Static mappings (non-layer specific)
        static_mappings = {
            f"{base_name}.embed_tokens.embeddings.weight":
            f"{base_name}.embed_tokens.weight",
            "lm_head.linear.weight": "lm_head.weight"
        }
        infer_to_train.update(static_mappings)

        base_name = base_name + ".layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int):
            # MoE specific mappings
            infer_to_train[f"{base_name}.{layer_idx}.mlp.gate_weight"] = \
                f"{base_name}.{layer_idx}.mlp.gate.weight"

            if self.fd_config.moe_config.moe_use_aux_free:
                infer_to_train[f"{base_name}.{layer_idx}.mlp.fused_moe.gate_correction_bias"] = \
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"

            # MoE experts mappings
            for expert_idx in range(self.fd_config.moe_config.num_experts):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.up_gate_proj_weight"
                    if up_gate_proj_key not in infer_to_train:
                        infer_to_train[up_gate_proj_key] = []
                    infer_to_train[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.down_proj_weight"
                    if down_proj_key not in infer_to_train:
                        infer_to_train[down_proj_key] = []
                    infer_to_train[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        # Process MoE layers
        for layer_idx in range(self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        return infer_to_train


class Qwen3ForCausalLMRL(Qwen3ForCausalLM):
    """
    Qwen3ForCausalLMRL
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen3ForCausalLMRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Qwen3ForCausalLMRL"
