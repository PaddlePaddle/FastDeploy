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

import copy
from typing import Dict

import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.ernie4_5_moe import (
    Ernie4_5_MoeForCausalLM,
    Ernie4_5_MoePretrainedModel,
)
from fastdeploy.model_executor.models.ernie4_5_vl.ernie4_5_vl_moe import (
    Ernie4_5_VLMoeForConditionalGeneration,
    Ernie4_5_VLPretrainedModel,
)
from fastdeploy.model_executor.models.glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoePretrainedModel,
)
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.models.qwen2 import (
    Qwen2ForCausalLM,
    Qwen2PretrainedModel,
)
from fastdeploy.model_executor.models.qwen2_5_vl.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPretrainedModel,
)
from fastdeploy.model_executor.models.qwen3 import (
    Qwen3ForCausalLM,
    Qwen3PretrainedModel,
)
from fastdeploy.model_executor.models.qwen3moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoePretrainedModel,
)
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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Get parameter name mappings between rollout and training models."""
        return getattr(self.rollout_model, "get_name_mappings_to_training", lambda: {})(trainer_degree)

    def get_quantization_infer_keys(self) -> Dict[str, str]:
        """Get parameter name mappings between rollout and training models."""
        return getattr(self.rollout_model, "get_quantization_infer_keys", lambda: {})()

    @paddle.no_grad()
    def state_dict(self):
        """state_dict"""
        return self.rollout_model.state_dict()


class BaseRLModel(nn.Layer):
    """Base class for RL models with common functionality"""

    def __init__(
        self,
    ):
        super(BaseRLModel, self).__init__()
        self.infer_to_train_mapping = {}
        self.fd_config = None
        self._mappings_built = False

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def _update_base_mappings(self, base_name: str) -> None:
        """Common static mappings"""
        static_mappings = {
            f"{base_name}.embed_tokens.embeddings.weight": f"{base_name}.embed_tokens.weight",
            "lm_head.linear.weight": "lm_head.weight",
        }
        self.infer_to_train_mapping.update(static_mappings)

    def _complete_missing_mappings(self) -> None:
        """
        Complete the mapping dictionary with keys that have identical names in inference and training.
        """
        for key in self.state_dict().keys():
            if key not in self.infer_to_train_mapping and "_scale" not in key:
                # Skip weight scale parameters in mapping. Train and infer have same key.
                self.infer_to_train_mapping[key] = key

    def get_quantization_infer_keys(self) -> list[str]:
        """Get quantization infer keys"""
        quant_weight_key = []
        if self.fd_config.quant_config.name() == "wint8":
            """RL only support weight_only_int8 now"""
            for key in self.state_dict().keys():
                if "scale" in key:
                    quant_weight_key.append(key.replace(".weight_scale", ".weight"))
        else:
            raise ValueError("Only 'wint8' quantization is supported in RL roullout.")
        return quant_weight_key


class Ernie4_5_MoeForCausalLMRL(Ernie4_5_MoeForCausalLM, BaseRLModel):
    """
    Ernie4_5_MoeForCausalLMRL
    """

    _get_tensor_parallel_mappings = Ernie4_5_MoePretrainedModel._get_tensor_parallel_mappings

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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(do not delete!)."""
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True

        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("ernie")

        base_name = "ernie.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int):
            # MoE specific mappings
            self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.gate.weight"] = (
                f"{base_name}.{layer_idx}.mlp.gate.weight"
            )

            if self.fd_config.model_config.moe_use_aux_free:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.experts.gate_correction_bias"] = (
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"
                )

            # MoE experts mappings
            for expert_idx in range(self.fd_config.model_config.moe_num_experts):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.experts.up_gate_proj_weight"
                    if up_gate_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[up_gate_proj_key] = []
                    self.infer_to_train_mapping[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.experts.down_proj_weight"
                    if down_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[down_proj_key] = []
                    self.infer_to_train_mapping[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        assert isinstance(self.fd_config.model_config.moe_layer_start_index, int)
        # Process MoE layers
        for layer_idx in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Ernie4_5_VLMoeForConditionalGenerationRL(Ernie4_5_VLMoeForConditionalGeneration, BaseRLModel):
    """
    Ernie4_5_VLMoeForConditionalGenerationRL
    """

    _get_tensor_parallel_mappings = Ernie4_5_VLPretrainedModel._get_tensor_parallel_mappings

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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(do not delete!)."""
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("ernie")

        base_name = "ernie.layers"

        # Helper function to add layer mappings
        def _add_expert_mappings(layer_idx: int, moe_tag: str, expert_start: int):
            # MoE specific mappings
            gate_suffix = "" if moe_tag == "text" else "_1"
            self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.gate.weight"] = (
                f"{base_name}.{layer_idx}.mlp.gate.weight{gate_suffix}"
            )

            if self.fd_config.model_config.moe_use_aux_free:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.gate_correction_bias"] = (
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"
                )

            # Initialize defaultdict for expert weights
            from collections import defaultdict
            from itertools import chain

            def _generate_ranges(start, end, step=16, take=8):
                """生成 [start, start+take), [start+step, start+step+take), ... 直到 end"""
                return chain(*(range(i, min(i + take, end)) for i in range(start, end, step)))  # 防止越界

            expert_mappings = defaultdict(list)
            for expert_idx in _generate_ranges(
                expert_start,
                total_moe_num,
                expert_num_per_rank * 2,
                expert_num_per_rank,
            ):
                for ph in place_holders:
                    expert_mappings[
                        f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.experts.up_gate_proj_weight"
                    ].append(f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}")
                    expert_mappings[
                        f"{base_name}.{layer_idx}.mlp.{moe_tag}_fused_moe.experts.down_proj_weight"
                    ].append(f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}")
            self.infer_to_train_mapping.update(expert_mappings)

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

        assert isinstance(self.fd_config.model_config.moe_num_experts, list)
        total_moe_num = sum(self.fd_config.model_config.moe_num_experts)
        if not trainer_degree:
            trainer_degree = self.fd_config.parallel_config.tensor_parallel_size
        expert_num_per_rank = self.fd_config.model_config.moe_num_experts[0] // trainer_degree
        # Process MoE layers
        for layer_idx in range(text_moe_layer_start_index, text_moe_layer_end_index):
            _add_expert_mappings(layer_idx, "text", expert_start=0)
        for layer_idx in range(image_moe_layer_start_index, image_moe_layer_end_index):
            _add_expert_mappings(layer_idx, "image", expert_start=expert_num_per_rank)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Qwen2ForCausalLMRL(Qwen2ForCausalLM, BaseRLModel):
    """
    Qwen2ForCausalLMRL
    """

    _get_tensor_parallel_mappings = Qwen2PretrainedModel._get_tensor_parallel_mappings

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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(do not delete!)."""
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("qwen2")
        base_name = "qwen2.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx):
            # FFN mappings
            for ph in place_holders:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.up_gate_proj.{ph}"] = (
                    f"{base_name}.{layer_idx}.mlp.gate_up_fused_proj.{ph}"
                )

        for layer_idx in range(self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Qwen3MoeForCausalLMRL(Qwen3MoeForCausalLM, BaseRLModel):
    """
    Qwen3MoeForCausalLMRL
    """

    _get_tensor_parallel_mappings = Qwen3MoePretrainedModel._get_tensor_parallel_mappings

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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(do not delete!)."""
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("model")

        base_name = "model.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int):
            # MoE specific mappings
            self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.gate.weight"] = (
                f"{base_name}.{layer_idx}.mlp.gate.weight"
            )

            if self.fd_config.model_config.moe_use_aux_free:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.experts.gate_correction_bias"] = (
                    f"{base_name}.{layer_idx}.mlp.moe_statics.e_score_correction_bias"
                )

            # MoE experts mappings
            for expert_idx in range(self.fd_config.model_config.num_experts):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.experts.up_gate_proj_weight"
                    if up_gate_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[up_gate_proj_key] = []
                    self.infer_to_train_mapping[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.experts.down_proj_weight"
                    if down_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[down_proj_key] = []
                    self.infer_to_train_mapping[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        # Process MoE layers
        for layer_idx in range(self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Qwen3ForCausalLMRL(Qwen3ForCausalLM, BaseRLModel):
    """
    Qwen3ForCausalLMRL
    """

    _get_tensor_parallel_mappings = Qwen3PretrainedModel._get_tensor_parallel_mappings

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

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("model")
        base_name = "model.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx):
            # FFN mappings
            for ph in place_holders:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.up_gate_proj.{ph}"] = (
                    f"{base_name}.{layer_idx}.mlp.gate_up_fused_proj.{ph}"
                )

        for layer_idx in range(self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Qwen2_5_VLForConditionalGenerationRL(Qwen2_5_VLForConditionalGeneration, BaseRLModel):
    """
    Qwen2_5_VLForConditionalGenerationRL
    """

    _get_tensor_parallel_mappings = Qwen2_5_VLPretrainedModel._get_tensor_parallel_mappings

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen2_5_VLForConditionalGenerationRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Qwen2_5_VLForConditionalGenerationRL"

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("model")
        base_name = "model.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx):
            # FFN mappings
            for ph in place_holders:
                self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.up_gate_proj.{ph}"] = (
                    f"{base_name}.{layer_idx}.mlp.gate_up_fused_proj.{ph}"
                )

        for layer_idx in range(self.fd_config.model_config.num_hidden_layers):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()

        return self.infer_to_train_mapping


class Glm4MoeForCausalLMRL(Glm4MoeForCausalLM, BaseRLModel):
    """
    Glm4MoeForCausalLMRL
    """

    _get_tensor_parallel_mappings = Glm4MoePretrainedModel._get_tensor_parallel_mappings

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Glm4MoeForCausalLMRL, self).__init__(fd_config)

    @classmethod
    def name(self) -> str:
        """name"""
        return "Glm4MoeForCausalLMRL"

    def get_name_mappings_to_training(self, trainer_degree=None) -> Dict[str, str]:
        """Generate mapping between inference and training parameter for RL(donot delete!)."""
        if self._mappings_built:
            return self.infer_to_train_mapping

        self.infer_to_train_mapping = {}
        self._mappings_built = True
        # Prepare placeholders
        place_holders = ["weight"]

        # Initialize mapping dictionary
        self._update_base_mappings("model")

        base_name = "model.layers"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx: int):
            # MoE specific mappings
            self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.gate.weight"] = (
                f"{base_name}.{layer_idx}.mlp.gate.weight"
            )

            self.infer_to_train_mapping[f"{base_name}.{layer_idx}.mlp.gate.e_score_correction_bias"] = (
                f"{base_name}.{layer_idx}.mlp.gate.e_score_correction_bias"
            )

            # MoE experts mappings
            for expert_idx in range(self.fd_config.model_config.n_routed_experts):
                for ph in place_holders:
                    # up_gate_proj (up_gate_proj)
                    up_gate_proj_key = f"{base_name}.{layer_idx}.mlp.experts.up_gate_proj_weight"
                    if up_gate_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[up_gate_proj_key] = []
                    self.infer_to_train_mapping[up_gate_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                    )

                    # down_proj (down_proj)
                    down_proj_key = f"{base_name}.{layer_idx}.mlp.experts.down_proj_weight"
                    if down_proj_key not in self.infer_to_train_mapping:
                        self.infer_to_train_mapping[down_proj_key] = []
                    self.infer_to_train_mapping[down_proj_key].append(
                        f"{base_name}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                    )

        # Process MoE layers
        for layer_idx in range(
            self.fd_config.model_config.first_k_dense_replace,
            self.fd_config.model_config.num_hidden_layers,
        ):
            _add_layer_mappings(layer_idx)

        self._complete_missing_mappings()
        infer_to_train_mapping_copy = copy.deepcopy(self.infer_to_train_mapping)
        for key in infer_to_train_mapping_copy.keys():
            if "mlp.experts.gate_correction_bias" in key:
                self.infer_to_train_mapping.pop(key)

        return self.infer_to_train_mapping
