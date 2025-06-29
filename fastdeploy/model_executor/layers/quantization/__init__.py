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
    "wint2",
    "wint4",
    "wint8",
    "weight_only",
    "block_wise_fp8",
    "w4afp8",
    "w8a8",
    "w4a8",
    "wfp8afp8",
    "mix_quant",
    "tensor_wise_fp8",
    "kvcache",
]


def get_quantization_config(quantization: str) -> Type[QuantConfigBase]:
    """
    Get the quantization config class by the quantization name.
    """
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    from .block_wise_fp8 import BlockWiseFP8Config
    from .kv_cache import KvCacheQuantConfig
    from .mix_quant import MixQuantConfig
    from .tensor_wise_fp8 import TensorWiseFP8Config
    from .w4a8 import W4A8Config
    from .w4afp8 import W4AFP8Config
    from .w8a8 import W8A8Config
    from .weight_only import WeightOnlyConfig, WINT4Config, WINT8Config
    from .wfp8afp8 import WFP8AFP8Config
    from .wint2 import WINT2Config

    method_to_config: Dict[str, Type[QuantConfigBase]] = {
        "wint2": WINT2Config,
        "wint4": WINT4Config,
        "wint8": WINT8Config,
        "weight_only": WeightOnlyConfig,
        "block_wise_fp8": BlockWiseFP8Config,
        "w4afp8": W4AFP8Config,
        "w8a8": W8A8Config,
        "w4a8": W4A8Config,
        "wfp8afp8": WFP8AFP8Config,
        "tensor_wise_fp8": TensorWiseFP8Config,
        "kvcache": KvCacheQuantConfig,
        "mix_quant": MixQuantConfig,
    }

    return method_to_config[quantization]
