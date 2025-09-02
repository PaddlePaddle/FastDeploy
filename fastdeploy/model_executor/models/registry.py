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
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

# 避免循环导入
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

from paddle import nn
from paddleformers.transformers import PretrainedModel

if TYPE_CHECKING:
    pass

from .interfaces_base import (
    determine_model_category,
    is_multimodal_model,
    is_pooling_model,
    is_text_generation_model,
)


class ModelCategory(Enum):
    """模型类别"""

    TEXT_GENERATION = "text_generation"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"


# 预定义的模型映射
_TEXT_GENERATION_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
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
    # 示例嵌入模型
    "BertForSequenceClassification": ("bert", "BertForSequenceClassification"),
    "Qwen2ForSequenceClassification": ("qwen2", "Qwen2ForSequenceClassification"),
}

_ALL_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_MULTIMODAL_MODELS,
    **_EMBEDDING_MODELS,
}


@dataclass(frozen=True)
class ModelInfo:
    """模型信息"""

    architecture: str
    category: ModelCategory
    is_text_generation: bool
    is_multimodal: bool
    is_pooling: bool
    module_path: str

    @staticmethod
    def from_model_cls(model_cls: Type[nn.Layer], module_path: str = "") -> "ModelInfo":
        """从模型类创建 ModelInfo 实例"""
        return ModelInfo(
            architecture=model_cls.__name__,
            category=determine_model_category(model_cls.__name__),
            is_text_generation=is_text_generation_model(model_cls),
            is_multimodal=is_multimodal_model(model_cls.__name__),
            is_pooling=is_pooling_model(model_cls),
            module_path=module_path,
        )


class BaseRegisteredModel(ABC):
    """注册模型基类"""

    @abstractmethod
    def load_model_cls(self) -> Type[nn.Layer]:
        raise NotImplementedError

    @abstractmethod
    def inspect_model_cls(self) -> ModelInfo:
        raise NotImplementedError


@dataclass(frozen=True)
class LazyRegisteredModel(BaseRegisteredModel):
    """懒加载模型"""

    module_name: str
    class_name: str

    def load_model_cls(self) -> Type[nn.Layer]:
        try:
            full_module = f"fastdeploy.model_executor.models.{self.module_name}"
            module = importlib.import_module(full_module)
            return getattr(module, self.class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"无法加载模型 {self.class_name}: {e}")

    def inspect_model_cls(self) -> ModelInfo:
        model_cls = self.load_model_cls()
        return ModelInfo.from_model_cls(model_cls, self.module_name)


@dataclass(frozen=True)
class RegisteredModel(BaseRegisteredModel):
    """已加载的模型"""

    model_cls: Type[nn.Layer]

    def load_model_cls(self) -> Type[nn.Layer]:
        return self.model_cls

    def inspect_model_cls(self) -> ModelInfo:
        return ModelInfo.from_model_cls(self.model_cls)


class ModelRegistry:
    """模型注册中心"""

    def __init__(self):
        self.models: Dict[str, BaseRegisteredModel] = {}
        self.pretrained_models: Dict[str, Type[PretrainedModel]] = {}
        self._register_predefined_models()

    def _register_predefined_models(self):
        """注册预定义模型"""
        for arch, (module_name, class_name) in _ALL_MODELS.items():
            self.models[arch] = LazyRegisteredModel(module_name, class_name)

    @lru_cache(maxsize=128)
    def _try_load_model_cls(self, architecture: str) -> Optional[Type[nn.Layer]]:
        if architecture not in self.models:
            return None
        try:
            return self.models[architecture].load_model_cls()
        except Exception as e:
            print(f"加载模型 {architecture} 失败: {e}")
            return None

    @lru_cache(maxsize=128)
    def _try_inspect_model_cls(self, architecture: str) -> Optional[ModelInfo]:
        if architecture not in self.models:
            return None
        try:
            return self.models[architecture].inspect_model_cls()
        except Exception as e:
            print(f"检查模型 {architecture} 失败: {e}")
            return None

    def register_model_class(self, model_class):
        """兼容旧的注册方法"""
        print(f"注册模型类: {model_class}")
        from .model_base import ModelForCasualLM

        if (
            inspect.isclass(model_class)
            and issubclass(model_class, ModelForCasualLM)
            and model_class is not ModelForCasualLM
        ):
            arch_name = model_class.__name__
            if hasattr(model_class, "name"):
                arch_name = model_class.name()
            self.models[arch_name] = RegisteredModel(model_class)
        return model_class

    def register_pretrained_model(self, pretrained_model):
        """注册预训练模型"""
        if (
            inspect.isclass(pretrained_model)
            and issubclass(pretrained_model, PretrainedModel)
            and pretrained_model is not PretrainedModel
            and hasattr(pretrained_model, "arch_name")
        ):
            self.pretrained_models[pretrained_model.arch_name()] = pretrained_model
        return pretrained_model

    def get_class(self, name):
        """获取模型类"""
        model_cls = self._try_load_model_cls(name)
        if model_cls is None:
            raise ValueError(f"模型 '{name}' 未注册!")
        print("model_cls", model_cls)
        return model_cls

    def get_pretrain_cls(self, architecture: str):
        """获取预训练模型类"""
        return self.pretrained_models.get(architecture)

    def get_supported_archs(self):
        """获取支持的架构"""
        # print("支持的架构", list(self.models.keys()))
        return list(self.models.keys())

    def resolve_model_cls(self, architectures: Union[str, List[str]]) -> Tuple[Type[nn.Layer], str]:
        """解析模型类"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return model_cls, arch

        raise ValueError(f"找不到支持的模型: {architectures}")

    def is_multimodal_model(self, architectures: Union[str, List[str]]) -> bool:
        """检查是否为多模态模型"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_multimodal
        return False

    def is_text_generation_model(self, architectures: Union[str, List[str]]) -> bool:
        """检查是否为文本生成模型"""
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_text_generation
        return False

    def is_pooling_model(self, architectures: Union[str, List[str]]) -> bool:
        if isinstance(architectures, str):
            architectures = [architectures]

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return model_info.is_pooling
        return False


# 全局注册中心实例
model_registry = ModelRegistry()
