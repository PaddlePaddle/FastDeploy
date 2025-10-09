"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    pass

from fastdeploy.model_executor.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepSeekV3PretrainedModel,
)


@ModelRegistry.register_model_class(
    architecture="DeepseekV32ForCausalLM",
    module_name="deepseek_v32",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """
    DeepseekV32ForCausalLM
    """

    @classmethod
    def name(cls):
        return "DeepseekV32ForCausalLM"


class DeepSeekV32PretrainedModel(DeepSeekV3PretrainedModel):
    """
    DeepSeekV32PretrainedModel
    """

    @classmethod
    def arch_name(self):
        return "DeepseekV32ForCausalLM"
