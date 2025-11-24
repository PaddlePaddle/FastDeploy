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

from fastdeploy.utils import parse_quantization

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


def parse_quant_config(args, model_config, is_ernie, is_v1_loader):
    if args.quantization is not None and isinstance(args.quantization, str):
        args.quantization = parse_quantization(args.quantization)
    # 1.model_config.is_quantized
    # TODO(bukejiyu)  model_config.is_quantized is v0 only need to be removed in future
    if model_config.model_format == "torch":
        quantization_config = model_config.quantization_config
        if quantization_config is not None:
            model_config.is_quantized = True
    else:
        quantization_config = model_config.quantization_config
        if not model_config.is_quantized:
            if quantization_config is not None:
                if "is_quantized" in quantization_config:
                    model_config.is_quantized = quantization_config["is_quantized"]
                elif "is_moe_quantized" in quantization_config:
                    model_config.is_moe_quantized = quantization_config["is_moe_quantized"]
                elif "kv_cache_quant_type" not in quantization_config:
                    model_config.is_quantized = True
                    if "is_moe_quantized" not in quantization_config:
                        model_config.is_quantized = True
                    else:
                        model_config.is_moe_quantized = True
            if quantization_config is not None and quantization_config.get("quantization", None) is None:
                raise ValueError(
                    "quantization_config should have a key named 'quantization' for specify quant config."
                )

    quant_config_name = None

    if quantization_config is not None:
        quant_config_name = _get_offline_quant_config_name(
            quantization_config, model_config.model_format == "torch", is_v1_loader
        )
    elif args.quantization is not None:
        quantization_config = {}
        try:
            quantization_config.update(args.quantization)
            quant_config_name = quantization_config["quantization"]
        except:
            quant_config_name = args.quantization["quantization"]
            quantization_config["quantization"] = quant_config_name
        # Special handling for Ernie models
        if quant_config_name == "wint4" and is_ernie:
            quantization_config["dense_quant_type"] = "wint8"
            quantization_config["moe_quant_type"] = "wint4"
            quantization_config["quantization"] = "mix_quant"
            quant_config_name = "mix_quant"
    else:
        quant_config_name = None
    if quant_config_name is None:
        quant_config = None
    else:
        if not quantization_config.get("is_quantized"):
            quantization_config["is_quantized"] = model_config.is_quantized
        if args.dynamic_load_weight and quantization_config is not None:
            quantization_config["is_quantized"] = True
        quant_cls = get_quantization_config(quant_config_name)
        quant_config = quant_cls.from_config(quantization_config)
    return quant_config


def _get_offline_quant_config_name(quantization_config, is_torch_weight, is_v1_loader):
    if is_torch_weight:
        # only support block_wise_fp8 now
        quant_method = quantization_config.get("quant_method")
        has_block_size = "weight_block_size" in quantization_config
        if quant_method == "fp8" and has_block_size:
            quant_config_name = "block_wise_fp8"
        else:
            raise ValueError("Torch weight offline quantization only supports block-wise FP8.")
    else:
        quant_config_name = quantization_config["quantization"]
    return quant_config_name


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
