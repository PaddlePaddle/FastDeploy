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

from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)

from .base import PaddleFormersModelBase
from .causallm import CausalLMMixin

__all__ = [
    "PaddleFormersForCausalLM",
]


# ============ Text Generation Models ============
@ModelRegistry.register_model_class(
    architecture="PaddleFormersForCausalLM",
    module_name="paddleformers",
    category=ModelCategory.TEXT_GENERATION,
)
class PaddleFormersForCausalLM(CausalLMMixin, PaddleFormersModelBase, ModelForCasualLM):
    @classmethod
    def name(cls):
        return "PaddleFormersForCausalLM"
