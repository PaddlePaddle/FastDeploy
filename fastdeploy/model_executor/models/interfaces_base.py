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

from typing import Type

from paddle import nn

from .model_base import ModelForCasualLM


def is_text_generation_model(model_cls: Type[nn.Layer]) -> bool:
    """检查模型是否为文本生成模型"""
    return issubclass(model_cls, ModelForCasualLM)


def is_pooling_model(model_cls: Type[nn.Layer]) -> bool:
    """检查模型是否为嵌入（池化）模型"""
    class_name = model_cls.__name__
    pooling_indicators = ["Embedding", "ForSequenceClassification"]
    return (
        any(indicator in class_name for indicator in pooling_indicators)
        or hasattr(model_cls, "is_embedding_model")
        and model_cls.is_embedding_model
    )


def is_multimodal_model(class_name: str) -> bool:
    """判断是否为多模态模型"""
    multimodal_indicators = ["VL", "Vision", "ConditionalGeneration"]
    return any(indicator in class_name for indicator in multimodal_indicators)


def determine_model_category(class_name: str):
    """确定模型类别"""
    from .registry import ModelCategory

    if any(pattern in class_name for pattern in ["VL", "Vision", "ConditionalGeneration"]):
        return ModelCategory.MULTIMODAL
    elif any(pattern in class_name for pattern in ["Embedding", "ForSequenceClassification"]):
        return ModelCategory.EMBEDDING
    return ModelCategory.TEXT_GENERATION
