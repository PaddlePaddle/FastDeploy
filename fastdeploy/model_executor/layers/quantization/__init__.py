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
quantization module
"""
from typing import Dict, List, Type

from .quant_base import QuantConfigBase

QUANTIZATION_METHODS: List[str] = [
    "weight_only",
    "block_wise",
    "w4afp8",
    "w8a8",
    "wfp8afp8",
]


def get_quantization_config(quantization: str) -> Type[QuantConfigBase]:
    """
    Get the quantization config class by the quantization name.
    """
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    from .block_wise import BlockWiseConfig
    from .w4afp8 import W4AFP8Config
    from .w8a8 import W8A8Config
    from .weight_only import WeightOnlyConfig
    from .wfp8afp8 import WFP8AFP8Config
    from .kv_cache import KvCacheQuantConfig
    
    method_to_config: Dict[str, Type[QuantConfigBase]] = {
        "weight_only": WeightOnlyConfig,
        "block_wise": BlockWiseConfig,
        "w4afp8": W4AFP8Config,
        "w8a8": W8A8Config,
        "wfp8afp8": WFP8AFP8Config,
        "kvcache": KvCacheQuantConfig
    }

    return method_to_config[quantization]
