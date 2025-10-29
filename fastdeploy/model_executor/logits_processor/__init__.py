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

from importlib import import_module

from .base import LogitsProcessor
from .builtin import LogitBiasLogitsProcessor


def load_class(spec: str):
    """
    Load a class from a string spec.

    If the spec is in the form 'package.module:ClassName', loads ClassName from the specified module.
    If the spec does not contain a colon, it is treated as the name of a builtin class from
    'fastdeploy.model_executor.logits_processor'.

    Args:
        spec (str): The class specifier string.

    Returns:
        type: The loaded class object.

    Raises:
        ValueError: If the spec is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
    """
    try:
        if ":" in spec:
            module_path, class_name = spec.split(":", 1)
        else:
            module_path = "fastdeploy.model_executor.logits_processor"
            class_name = spec
        module = import_module(module_path)
        obj = getattr(module, class_name)
        return obj
    except ValueError as e:
        raise ValueError(f"Invalid spec {spec!r}; expected 'module:ClassName'.") from e
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}") from e
    except AttributeError as e:
        raise AttributeError(f"Module {module_path} does not have attribute {class_name}") from e


def build_logits_processors(fd_config):
    logit_procs = []
    for fqcn in fd_config.structured_outputs_config.logits_processors or []:
        logit_procs.append(load_class(fqcn)(fd_config))
    return logit_procs


__all__ = [
    "build_logits_processors",
    "LogitsProcessor",
    "LogitBiasLogitsProcessor",
]
