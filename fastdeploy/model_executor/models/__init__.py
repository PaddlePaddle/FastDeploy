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
import importlib
import inspect
import os

from .model_base import ModelForCasualLM, ModelRegistry

inference_runner_supported_models = ["Qwen2ForCausalLM"]


def auto_models_registry():
    """
    auto registry all models in this folder
    """
    for module_file in os.listdir(os.path.dirname(__file__)):
        if module_file.endswith('.py') and module_file != '__init__.py':
            module_name = module_file[:-3]
            try:
                module = importlib.import_module(
                    f'fastdeploy.model_executor.models.{module_name}')
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr) and issubclass(
                            attr,
                            ModelForCasualLM) and attr is not ModelForCasualLM:
                        ModelRegistry.register(attr)
            except ImportError:
                raise ImportError(f"{module_name=} import error")


auto_models_registry()
