"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
        @@ -12,31 +11,265 @@
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag, auto
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
from fastdeploy.model_executor.models.interfaces_base import get_default_pooling_type


class ModelCategory(IntFlag):
    TEXT_GENERATION = auto()
    MULTIMODAL = auto()
    EMBEDDING = auto()
    REASONING = auto()
    REWARD = auto()


@dataclass(frozen=True)
class ModelInfo:
    architecture: str
    category: ModelCategory
    is_text_generation: bool
    is_multimodal: bool
    is_reasoning: bool
    is_pooling: bool
    module_path: str
    default_pooling_type: str

    @staticmethod
    def from_model_cls(
        model_cls: Type[nn.Layer], module_path: str = "", category: ModelCategory = None
    ) -> "ModelInfo":
        return ModelInfo(
            architecture=model_cls.__name__,
            category=category,
            is_text_generation=ModelCategory.TEXT_GENERATION in category,
            is_multimodal=ModelCategory.MULTIMODAL in category,
            is_reasoning=ModelCategory.REASONING in category,
            is_pooling=ModelCategory.EMBEDDING in category,
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
    module_path: str
    class_name: str
    category: ModelCategory

    def load_model_cls(self) -> Type[nn.Layer]:
        try:
            full_module = f"{self.module_path}.{self.module_name}"
            module = importlib.import_module(full_module)
            return getattr(module, self.class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load {self.class_name}: {e}")

    def inspect_model_cls(self) -> ModelInfo:
        model_cls = self.load_model_cls()
        return ModelInfo.from_model_cls(model_cls, self.module_name, self.category)


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
    _enhanced_models: Dict[str, Dict] = {}

    def __init__(self):
        self.models: Dict[str, BaseRegisteredModel] = {}
        self.pretrained_models: Dict[str, Type[PretrainedModel]] = {}
        self._registered_models: Dict[str, BaseRegisteredModel] = {}
        self._register_enhanced_models()

    def _register_enhanced_models(self):
        for arch, model_info in self._enhanced_models.items():
            model = LazyRegisteredModel(
                module_name=model_info["module_name"],
                module_path=model_info["module_path"],
                class_name=model_info["class_name"],
                category=model_info["category"],
            )
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

    def _normalize_arch(self, architecture: str, model_config: ModelConfig) -> str:
        if architecture in self.models:
            return architecture

        match = try_match_architecture_defaults(
            architecture,
            runner_type=getattr(model_config, "runner_type", None),
            convert_type=getattr(model_config, "convert_type", None),
        )
        if match:
            suffix, _ = match
            for repl_suffix, _ in iter_architecture_defaults():
                base_arch = architecture.replace(suffix, repl_suffix)
                if base_arch in self.models:
                    return base_arch

        return architecture

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed to be inspected. "
                "Please check the logs for more details."
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
    def register_model_class(
        cls,
        model_class=None,
        *,
        architecture: str = None,
        module_name: str = None,
        module_path: str = "fastdeploy.model_executor.models",
        category: Union[ModelCategory, List[ModelCategory]] = ModelCategory.TEXT_GENERATION,
        primary_use: ModelCategory = None,
    ):
        """
        Enhanced model class registration supporting both traditional and decorator-style registration.

        Can be used as:
        1. Traditional decorator: @ModelRegistry.register_model_class
        2. Enhanced decorator with metadata: @ModelRegistry.register_model_class(architecture="...", module_path="...")

        Args:
            model_class: The model class (when used as simple decorator)
            architecture (str): Unique identifier for the model architecture
            module_name (str): Relative path to the module containing the model
            module_path (str): Absolute path to the module containing the model
            category: Model category or list of categories
            primary_use: Primary category for multi-category models
        """

        def _register(model_cls):
            # Traditional registration for ModelForCasualLM subclasses
            cls._arch_to_model_cls[model_cls.name()] = model_cls

            # Enhanced decorator-style registration
            if architecture and module_name:
                categories = category if isinstance(category, list) else [category]

                # Register main entry
                arch_key = architecture
                cls._enhanced_models[arch_key] = {
                    "class_name": model_cls.__name__,
                    "module_name": module_name,
                    "module_path": module_path,
                    "category": primary_use or categories[0],
                    "class": model_cls,
                }

                # Register category-specific entries for multi-category models
                if len(categories) > 1:
                    for cat in categories:
                        key = f"{arch_key}_{cat.value}"
                        cls._enhanced_models[key] = {
                            "class_name": model_cls.__name__,
                            "module_name": module_name,
                            "module_path": module_path,
                            "category": cat,
                            "primary_use": primary_use or categories[0],
                            "class": model_cls,
                        }
            return model_cls

        if model_class is not None:
            return _register(model_class)
        else:
            return _register

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
        traditional_archs = list(cls._arch_to_model_cls.keys())
        enhanced_archs = list(cls._enhanced_models.keys())
        return traditional_archs + enhanced_archs

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

    def is_reasoning_model(self, architectures: Union[str, List[str]], model_config: ModelConfig = None) -> bool:
        """Check if it's a reasoning model"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_reasoning
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
