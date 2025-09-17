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

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel

from fastdeploy.config import (
    ModelConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)

from .interfaces_base import (
    determine_model_category,
    get_default_pooling_type,
    is_multimodal_model,
    is_pooling_model,
    is_text_generation_model,
)


class ModelCategory(Enum):
    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"


_TEXT_GENERATION_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3moe", "Qwen3MoeForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v3", "DeepseekV3ForCausalLM"),
    "Ernie4_5_MTPForCausalLM": ("ernie4_5_mtp", "Ernie4_5_MTPForCausalLM"),
    "Ernie4_5_ForCausalLM": ("ernie4_5_moe", "Ernie4_5_ForCausalLM"),
    "Ernie4_5_MoeForCausalLM": ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM"),
}

_MULTIMODAL_MODELS = {
    "Qwen2_5_VLForConditionalGeneration": ("qwen2_5_vl.qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    "Ernie4_5_VLMoeForConditionalGeneration": (
        "ernie4_5_vl.ernie4_5_vl_moe",
        "Ernie4_5_VLMoeForConditionalGeneration",
    ),
}

_EMBEDDING_MODELS = {
    "BertModel": ("bert", "BertEmbeddingModel"),
    "Qwen2Model": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "Qwen2ForProcessRewardModel": ("qwen2_rm", "Qwen2ForProcessRewardModel"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
}

_ALL_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_MULTIMODAL_MODELS,
    **_EMBEDDING_MODELS,
}


@dataclass(frozen=True)
class ModelInfo:

    architecture: str
    category: ModelCategory
    is_text_generation: bool
    is_multimodal: bool
    is_pooling: bool
    module_path: str
    default_pooling_type: str

    @staticmethod
    def from_model_cls(model_cls: Type[nn.Layer], module_path: str = "") -> "ModelInfo":
        return ModelInfo(
            architecture=model_cls.__name__,
            category=determine_model_category(model_cls.__name__),
            is_text_generation=is_text_generation_model(model_cls),
            is_multimodal=is_multimodal_model(model_cls.__name__),
            is_pooling=is_pooling_model(model_cls),
            default_pooling_type=get_default_pooling_type(model_cls),
            module_path=module_path,
        )


class BaseRegisteredModel(ABC):
    """Base class for registered models"""

    @abstractmethod
    def load_model_cls(self) -> Type[nn.Layer]:
        raise NotImplementedError

    @abstractmethod
    def inspect_model_cls(self) -> ModelInfo:
        raise NotImplementedError


@dataclass(frozen=True)
class LazyRegisteredModel(BaseRegisteredModel):
    """Lazy loaded model"""

    module_name: str
    class_name: str

    def load_model_cls(self) -> Type[nn.Layer]:
        try:
            full_module = f"fastdeploy.model_executor.models.{self.module_name}"
            module = importlib.import_module(full_module)
            return getattr(module, self.class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load {self.class_name}: {e}")

    def inspect_model_cls(self) -> ModelInfo:
        model_cls = self.load_model_cls()
        return ModelInfo.from_model_cls(model_cls, self.module_name)


@dataclass(frozen=True)
class RegisteredModel(BaseRegisteredModel):

    model_cls: Type[nn.Layer]

    def load_model_cls(self) -> Type[nn.Layer]:
        return self.model_cls

    def inspect_model_cls(self) -> ModelInfo:
        return ModelInfo.from_model_cls(self.model_cls)


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: BaseRegisteredModel,
) -> Optional[ModelInfo]:
    try:
        return model.inspect_model_cls()
    except Exception:
        print("Error in inspecting model architecture '%s'", model_arch)
        return None


class ModelRegistry:

    _arch_to_model_cls = {}
    _arch_to_pretrained_model_cls = {}

    def __init__(self):
        self.models: Dict[str, BaseRegisteredModel] = {}
        self.pretrained_models: Dict[str, Type[PretrainedModel]] = {}
        self._registered_models: Dict[str, BaseRegisteredModel] = {}
        self._register_predefined_models()

    def _register_predefined_models(self):
        for arch, (module_name, class_name) in _ALL_MODELS.items():
            model = LazyRegisteredModel(module_name, class_name)
            self.models[arch] = model
            self._registered_models[arch] = model

    @lru_cache(maxsize=128)
    def _try_load_model_cls(self, architecture: str) -> Optional[Type[nn.Layer]]:
        if architecture not in self.models:
            return None
        try:
            return self.models[architecture].load_model_cls()
        except Exception as e:
            print(f"Failed to load model {architecture}: {e}")
            return None

    @lru_cache(maxsize=128)
    def _try_inspect_model_cls(self, model_arch: str) -> Optional[ModelInfo]:
        if model_arch not in self.models:
            return None
        try:
            return self.models[model_arch].inspect_model_cls()
        except Exception as e:
            print(f"Failed to inspect model {model_arch}: {e}")
            return None

    def _normalize_arch(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> str:
        if architecture in self.models:
            return architecture

        # This may be called in order to resolve runner_type and convert_type
        # in the first place, in which case we consider the default match
        match = try_match_architecture_defaults(
            architecture,
            runner_type=getattr(model_config, "runner_type", None),
            convert_type=getattr(model_config, "convert_type", None),
        )
        if match:
            suffix, _ = match

            # Get the name of the base model to convert
            for repl_suffix, _ in iter_architecture_defaults():
                base_arch = architecture.replace(suffix, repl_suffix)
                if base_arch in self.models:
                    return base_arch

        return architecture

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failedare not supported. "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def inspect_model_cls(
        self, architectures: Union[str, List[str]], model_config: ModelConfig = None
    ) -> Tuple[ModelInfo, str]:
        if isinstance(architectures, str):
            architectures = [architectures]

        if not architectures:
            raise ValueError("No model architectures are specified")

        for arch in architectures:
            normalized_arch = self._normalize_arch(arch, model_config)
            model_info = self._try_inspect_model_cls(normalized_arch)
            if model_info is not None:
                return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    @classmethod
    def register_model_class(cls, model_class):
        """register model class"""
        if issubclass(model_class, ModelForCasualLM) and model_class is not ModelForCasualLM:
            cls._arch_to_model_cls[model_class.name()] = model_class
        return model_class

    @classmethod
    def register_pretrained_model(cls, pretrained_model):
        """register pretrained model class"""
        if (
            issubclass(pretrained_model, PretrainedModel)
            and pretrained_model is not PretrainedModel
            and hasattr(pretrained_model, "arch_name")
        ):
            cls._arch_to_pretrained_model_cls[pretrained_model.arch_name()] = pretrained_model

        return pretrained_model

    @classmethod
    def get_class(cls, name):
        """get model class"""
        if name not in cls._arch_to_model_cls:
            raise ValueError(f"Model '{name}' is not registered!")
        return cls._arch_to_model_cls[name]

    @classmethod
    def get_pretrain_cls(cls, architectures: str):
        """get_pretrain_cls"""
        return cls._arch_to_pretrained_model_cls[architectures]

    @classmethod
    def get_supported_archs(cls):
        assert len(cls._arch_to_model_cls) >= len(
            cls._arch_to_pretrained_model_cls
        ), "model class num is more than pretrained model registry num"
        return [key for key in cls._arch_to_model_cls.keys()]

    def resolve_model_cls(self, architectures: Union[str, List[str]]) -> Tuple[Type[nn.Layer], str]:
        """Resolve model class"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return model_cls, arch

        raise ValueError(f"Cannot find supported model: {architectures}")

    def is_multimodal_model(self, architectures: Union[str, List[str]], model_config: ModelConfig = None) -> bool:
        """Check if it's a multimodal model"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_multimodal
        return False

    def is_text_generation_model(self, architectures: Union[str, List[str]], model_config: ModelConfig = None) -> bool:
        """Check if it's a text generation model"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_text_generation
        return False

    def is_pooling_model(self, architectures: Union[str, List[str]], model_config: ModelConfig = None) -> bool:
        """Check if it's a pooling model"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_pooling
        return False


class ModelForCasualLM(nn.Layer, ABC):
    """
    Base class for LM
    """

    def __init__(self, configs):
        """
        Args:
            configs (dict): Configurations including parameters such as max_dec_len, min_dec_len, decode_strategy,
                vocab_size, use_topp_sampling, etc.
        """
        super(ModelForCasualLM, self).__init__()
        self.fd_config = configs

    @abstractmethod
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.
        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input_ids=None,
        pos_emb=None,
        **model_kwargs,
    ):
        """
        Defines the forward pass of the model for generating text.
        Args:
            input_ids (Tensor, optional): The input token ids to the model.
            pos_emb (Tensor, optional): position Embeddings for model.
            **model_kwargs: Additional keyword arguments for the model.
        Returns:
            Tensor or list of Tensors: Generated tokens or decoded outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_logits(self, hidden_state, **logits_prosessor_kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(self):
        raise NotImplementedError
