"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers import PretrainedModel

from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() or current_platform.is_maca():
    pass

from fastdeploy.model_executor.models.deepseek_v3 import DeepseekV32ForCausalLM


@ModelRegistry.register_model_class(
    architecture="GlmMoeDsaForCausalLM",
    module_name="glm_moe_dsa",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    """
    GlmMoeDsaForCausalLM - GLM-5 with MLA/DSA attention
    """

    @classmethod
    def name(cls):
        """ """
        return "GlmMoeDsaForCausalLM"


class GlmMoeDsaPretrainedModel(PretrainedModel):
    """
    GlmMoeDsaPretrainedModel
    """

    @classmethod
    def arch_name(self):
        return "GlmMoeDsaForCausalLM"
