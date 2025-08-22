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

from typing import Callable


class MultimodalRegistry:
    """
    A registry for multimodal models
    """

    mm_models: set[str] = set()

    @classmethod
    def register_model(cls, name: str = "") -> Callable:
        """
        Register model with the given name, class name is used if name is not provided.
        """

        def _register(model):
            nonlocal name
            if len(name) == 0:
                name = model.__name__
            if name in cls.mm_models:
                raise ValueError(f"multimodal model {name} is already registered")
            cls.mm_models.add(name)
            return model

        return _register

    @classmethod
    def contains_model(cls, name: str) -> bool:
        """
        Check if the given name exists in registry.
        """
        return name in cls.mm_models
