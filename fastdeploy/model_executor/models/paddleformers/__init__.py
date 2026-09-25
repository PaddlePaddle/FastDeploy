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
from fastdeploy.model_executor.utils import is_paddlefleet_available

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

    def __call__(self, inputs=None, forward_meta=None, **kwargs):
        # Some model runners (e.g. xpu_model_runner.execute_model) call the model
        # positionally, while support_graph_optimization installs a kwargs-only
        # __call__. Bridge the two calling conventions.
        if isinstance(inputs, dict):
            kwargs.update(inputs)
        elif inputs is not None:
            kwargs["ids_remove_padding"] = inputs
        if forward_meta is not None:
            kwargs["forward_meta"] = forward_meta
        return super().__call__(**kwargs)


if is_paddlefleet_available():
    from .base_fleet import PaddleFleetModelBase

    __all__ += ["PaddleFleetForCausalLM"]

    @ModelRegistry.register_model_class(
        architecture="PaddleFleetForCausalLM",
        module_name="paddleformers",
        category=ModelCategory.TEXT_GENERATION,
    )
    class PaddleFleetForCausalLM(PaddleFleetModelBase, ModelForCasualLM):
        @classmethod
        def name(cls):
            return "PaddleFleetForCausalLM"
