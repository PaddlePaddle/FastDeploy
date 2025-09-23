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


def is_text_generation_model(model_cls: Type[nn.Layer]) -> bool:
    from .model_base import ModelForCasualLM

    return issubclass(model_cls, ModelForCasualLM)


def is_pooling_model(model_cls: Type[nn.Layer]) -> bool:
    class_name = model_cls.__name__
    pooling_indicators = ["Embedding", "ForSequenceClassification"]
    return (
        any(indicator in class_name for indicator in pooling_indicators)
        or hasattr(model_cls, "is_embedding_model")
        and model_cls.is_embedding_model
    )


def is_multimodal_model(class_name: str) -> bool:
    multimodal_indicators = ["VL", "Vision", "ConditionalGeneration"]
    return any(indicator in class_name for indicator in multimodal_indicators)


def determine_model_category(class_name: str):
    from fastdeploy.model_executor.models.model_base import ModelCategory

    if any(pattern in class_name for pattern in ["VL", "Vision", "ConditionalGeneration"]):
        return ModelCategory.MULTIMODAL
    elif any(pattern in class_name for pattern in ["Embedding", "ForSequenceClassification"]):
        return ModelCategory.EMBEDDING
    return ModelCategory.TEXT_GENERATION


def get_default_pooling_type(model_cls: Type[nn.Layer] = None) -> str:
    if model_cls is not None:
        return getattr(model_cls, "default_pooling_type", "LAST")
    return "LAST"
