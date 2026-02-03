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

from typing import Optional

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.platforms import current_platform

from . import get_quantization_config
from .quant_base import QuantConfigBase, QuantMethodBase

# FP8 quantization types that require SM89+
_FP8_QUANT_TYPES = ["block_wise_fp8", "w4afp8", "wfp8afp8", "tensor_wise_fp8"]


def _check_fp8_support_and_fallback(quant_type: str) -> str:
    """
    Check if FP8 quantization type is supported on current hardware.
    Returns the fallback type if not supported.

    V100 (SM70) and A100 (SM80) do NOT support FP8 quantization.
    """
    if quant_type not in _FP8_QUANT_TYPES:
        return quant_type

    if not current_platform.is_cuda():
        return quant_type

    from paddleformers.utils.log import logger

    from fastdeploy.platforms.cuda import CUDAPlatform

    if CUDAPlatform.supports_fp8():
        return quant_type

    sm_version = CUDAPlatform.get_sm_version()

    # Provide fallback for FP8 quantization types
    if quant_type == "block_wise_fp8":
        logger.warning(
            f"FP8 quantization (block_wise_fp8) is not supported on SM{sm_version} "
            f"(requires SM{CUDAPlatform.SM_FP8_MIN}+). "
            f"Falling back to wint8 for dense layers."
        )
        return "wint8"
    elif quant_type == "w4afp8":
        logger.warning(
            f"W4AFP8 quantization is not supported on SM{sm_version} "
            f"(requires SM{CUDAPlatform.SM_FP8_MIN}+). "
            f"Falling back to wint4 for MoE layers."
        )
        return "wint4"
    elif quant_type == "wfp8afp8":
        logger.warning(
            f"WFP8AFP8 quantization is not supported on SM{sm_version} "
            f"(requires SM{CUDAPlatform.SM_FP8_MIN}+). "
            f"Falling back to wint8."
        )
        return "wint8"
    elif quant_type == "tensor_wise_fp8":
        logger.warning(
            f"Tensor-wise FP8 quantization is not supported on SM{sm_version} "
            f"(requires SM{CUDAPlatform.SM_FP8_MIN}+). "
            f"Falling back to wint8."
        )
        return "wint8"

    return quant_type


class MixQuantConfig(QuantConfigBase):
    """
    Quantization config for layers that has different quantization methods.
    """

    def __init__(
        self,
        dense_quant_type: str,
        moe_quant_type: str,
        kv_cache_quant_type: str = None,
        image_moe_quant_type: str = None,
        is_channel_wise: bool = False,
        has_zero_point: bool = False,
        is_permuted: bool = True,
        is_quantized: bool = False,
        hadamard_block_size: int = 128,
        moe_dynamic_quant: bool = False,
        is_moe_quantized: bool = False,
    ) -> None:
        super().__init__()
        self.dense_quant_type = dense_quant_type
        self.moe_quant_type = moe_quant_type
        self.kv_cache_quant_type = kv_cache_quant_type
        if image_moe_quant_type is None:
            self.image_moe_quant_type = moe_quant_type
        else:
            self.image_moe_quant_type = image_moe_quant_type
        self.is_channel_wise = is_channel_wise
        self.has_zero_point = has_zero_point
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0
        self.is_permuted = is_permuted
        self.is_checkpoint_bf16 = not is_quantized
        self.is_quantized = is_quantized
        self.hadamard_block_size = hadamard_block_size
        self.moe_dynamic_quant = moe_dynamic_quant
        self.is_moe_quantized = is_moe_quantized

    def name(self) -> str:
        return "mix_quant"

    @classmethod
    def from_config(cls, config: dict) -> "MixQuantConfig":
        return cls(
            config.get("dense_quant_type", None),
            config.get("moe_quant_type", None),
            config.get("kv_cache_quant_type", None),
            config.get("image_moe_quant_type", None),
            config.get("is_channel_wise", False),
            config.get("has_zero_point", False),
            config.get("is_permuted", True),
            config.get("is_quantized", False),
            config.get("hadamard_block_size", 128),
            config.get("moe_dynamic_quant", False),
            config.get("is_moe_quantized", False),
        )

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            if layer.moe_tag == "Image":
                if self.image_moe_quant_type is not None:
                    # Check and fallback FP8 quant types for SM70 compatibility
                    actual_quant_type = _check_fp8_support_and_fallback(self.image_moe_quant_type)
                    return (
                        get_quantization_config(actual_quant_type)
                        .from_config(
                            {
                                "is_permuted": self.is_permuted,
                                "is_quantized": not self.is_checkpoint_bf16,
                                "hadamard_block_size": self.hadamard_block_size,
                            }
                        )
                        .get_quant_method(layer)
                    )
                else:
                    return None
            else:
                if self.moe_quant_type is not None:
                    # Check and fallback FP8 quant types for SM70 compatibility
                    actual_quant_type = _check_fp8_support_and_fallback(self.moe_quant_type)
                    return (
                        get_quantization_config(actual_quant_type)
                        .from_config(
                            {
                                "is_permuted": self.is_permuted,
                                "is_quantized": not self.is_checkpoint_bf16 or self.is_moe_quantized,
                                "hadamard_block_size": self.hadamard_block_size,
                            }
                        )
                        .get_quant_method(layer)
                    )
                else:
                    return None
        elif isinstance(layer, Attention):
            if self.kv_cache_quant_type is not None:
                return (
                    get_quantization_config("kvcache")
                    .from_config(self.kv_cache_quant_type, self.is_channel_wise, self.has_zero_point)
                    .get_quant_method(layer)
                )
            else:
                return None
        else:
            if self.dense_quant_type is not None:
                # Check and fallback FP8 quant types for SM70 compatibility
                actual_quant_type = _check_fp8_support_and_fallback(self.dense_quant_type)
                return (
                    get_quantization_config(actual_quant_type)
                    .from_config({"is_quantized": not self.is_checkpoint_bf16})
                    .get_quant_method(layer)
                )
            else:
                return None
