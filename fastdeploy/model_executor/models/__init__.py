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
from pathlib import Path

from paddleformers.transformers import PretrainedModel

from fastdeploy.plugins.model_register import load_model_register_plugins

from .model_base import ModelForCasualLM, ModelRegistry


def _find_py_files(root_dir):
    root_path = Path(root_dir)
    py_files = []
    for py_file in root_path.rglob("*.py"):
        rel_path = py_file.relative_to(root_dir)
        if "__init__" in str(py_file):
            continue
        dotted_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        py_files.append(dotted_path)
    return py_files


def auto_models_registry(dir_path, register_path="fastdeploy.model_executor.models"):
    """
    auto registry all models in this folder
    """
    for module_file in _find_py_files(dir_path):
        try:
            module = importlib.import_module(f"{register_path}.{module_file}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                if inspect.isclass(attr) and issubclass(attr, ModelForCasualLM) and attr is not ModelForCasualLM:
                    ModelRegistry.register_model_class(attr)

                if (
                    inspect.isclass(attr)
                    and issubclass(attr, PretrainedModel)
                    and attr is not PretrainedModel
                    and hasattr(attr, "arch_name")
                ):
                    ModelRegistry.register_pretrained_model(attr)

        except ImportError:
            raise ImportError(f"{module_file=} import error")


auto_models_registry(os.path.dirname(__file__))

load_model_register_plugins()
