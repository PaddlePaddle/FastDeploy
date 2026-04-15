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

"""Registry for multimodal encoding strategy classes."""

from typing import Dict, Type


class EncodingRegistry:
    """Maps model_type strings to encoding strategy classes.

    Encoding classes register themselves via the ``register`` decorator
    at import time.  ``MultiModalProcessor`` queries this registry by
    *model_type* instead of using string-based dynamic imports.
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, *model_types: str):
        """Decorator that registers an encoding class for one or more model types."""

        def decorator(enc_cls):
            for mt in model_types:
                if mt in cls._registry:
                    raise ValueError(
                        f"Encoding for '{mt}' already registered "
                        f"as {cls._registry[mt].__name__}, "
                        f"cannot re-register as {enc_cls.__name__}"
                    )
                cls._registry[mt] = enc_cls
            return enc_cls

        return decorator

    @classmethod
    def get(cls, model_type: str) -> Type:
        """Look up the encoding class for a given *model_type*."""
        if model_type not in cls._registry:
            raise ValueError(
                f"No encoding registered for '{model_type}'. " f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[model_type]
