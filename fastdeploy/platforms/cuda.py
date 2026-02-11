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

import functools
import traceback

import paddle

from fastdeploy.utils import console_logger as logger

from .base import Platform, _Backend


class CUDAPlatform(Platform):
    """
    cuda platform class
    """

    device_name = "gpu"

    # SM architecture thresholds
    SM_BF16_MIN = 80  # BF16 requires SM80+ (Ampere)
    SM_FP8_MIN = 89  # FP8 requires SM89+ (Ada Lovelace)
    SM_ASYNC_COPY_MIN = 80  # cp.async requires SM80+ (Ampere)
    SM_MARLIN_MIN = 80  # Marlin GEMM requires SM80+ (Ampere)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_sm_version(cls) -> int:
        """
        Get the SM version of the current CUDA device.
        Returns the compute capability as an integer (e.g., 70 for V100, 80 for A100).
        """
        try:
            prop = paddle.device.cuda.get_device_properties()
            return prop.major * 10 + prop.minor
        except Exception:
            return 0

    @classmethod
    def supports_bf16(cls) -> bool:
        """
        Check if the current GPU supports BF16 (bfloat16).
        BF16 requires SM80+ (Ampere architecture or newer).
        V100 (SM70) does NOT support BF16.
        """
        return cls.get_sm_version() >= cls.SM_BF16_MIN

    @classmethod
    def supports_fp8(cls) -> bool:
        """
        Check if the current GPU supports FP8 quantization.
        FP8 requires SM89+ (Ada Lovelace architecture or newer).
        V100 (SM70) and A100 (SM80) do NOT support FP8.
        """
        return cls.get_sm_version() >= cls.SM_FP8_MIN

    @classmethod
    def supports_async_copy(cls) -> bool:
        """
        Check if the current GPU supports cp.async instructions.
        cp.async requires SM80+ (Ampere architecture or newer).
        V100 (SM70) does NOT support cp.async.
        This affects Append Attention and MLA Attention backends.
        """
        return cls.get_sm_version() >= cls.SM_ASYNC_COPY_MIN

    @classmethod
    def supports_marlin(cls) -> bool:
        """
        Check if the current GPU supports Marlin GEMM kernels.
        Marlin requires SM80+ (Ampere architecture or newer).
        V100 (SM70) does NOT support Marlin.
        """
        return cls.get_sm_version() >= cls.SM_MARLIN_MIN

    @classmethod
    def get_recommended_dtype(cls, requested_dtype: str) -> str:
        """
        Get the recommended dtype based on hardware capabilities.
        Automatically downgrades BF16 to FP16 on unsupported hardware.

        Args:
            requested_dtype: The requested dtype (e.g., "bfloat16", "float16")

        Returns:
            The recommended dtype that is supported by the hardware.
        """
        sm_version = cls.get_sm_version()
        if requested_dtype in ("bfloat16", "bf16"):
            if not cls.supports_bf16():
                logger.warning(
                    f"BF16 is not supported on SM{sm_version} (requires SM{cls.SM_BF16_MIN}+). "
                    f"Automatically falling back to FP16."
                )
                return "float16"
        return requested_dtype

    @classmethod
    def available(self):
        """
        Check whether CUDA is available.
        """
        try:
            assert len(paddle.static.cuda_places()) > 0
            return True
        except Exception as e:
            logger.warning(
                "You are using GPU version PaddlePaddle, but there is no GPU "
                "detected on your machine. Maybe CUDA devices is not set properly."
                f"\n Original Error is {e}, "
                f"{str(traceback.format_exc())}"
            )
            return False

    @classmethod
    def get_attention_backend_cls(cls, selected_backend: _Backend):
        """
        get_attention_backend_cls with automatic fallback for SM70 (V100)
        """
        sm_version = cls.get_sm_version()

        # Check for SM70 (V100) compatibility and apply fallbacks
        if not cls.supports_async_copy():
            # APPEND_ATTN, MLA_ATTN, and FLASH_ATTN all require SM80+ (cp.async or dependent ops)
            # V100 must use NATIVE_ATTN which is the only fully compatible backend
            if selected_backend in (_Backend.APPEND_ATTN, _Backend.MLA_ATTN, _Backend.FLASH_ATTN):
                logger.warning(
                    f"{selected_backend} backend requires SM{cls.SM_ASYNC_COPY_MIN}+ "
                    f"(cp.async instructions or dependent ops), "
                    f"but current GPU is SM{sm_version}. "
                    f"Automatically falling back to NATIVE_ATTN backend."
                )
                selected_backend = _Backend.NATIVE_ATTN

        if selected_backend == _Backend.NATIVE_ATTN:
            logger.info("Using NATIVE ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.PaddleNativeAttnBackend"
        elif selected_backend == _Backend.APPEND_ATTN:
            logger.info("Using APPEND ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.AppendAttentionBackend"
        elif selected_backend == _Backend.MLA_ATTN:
            logger.info("Using MLA ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.MLAAttentionBackend"
        elif selected_backend == _Backend.DSA_ATTN:
            logger.info("Using DSA ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.DSAAttentionBackend"
        elif selected_backend == _Backend.FLASH_ATTN:
            logger.info("Using FLASH ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.FlashAttentionBackend"
        elif selected_backend == _Backend.PLAS_ATTN:
            logger.info("Using PLAS ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.PlasAttentionBackend"
        elif selected_backend == _Backend.FLASH_MASK_ATTN:
            logger.info("Using FLASH MASK ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.FlashMaskAttentionBackend"
        else:
            raise ValueError(
                "Invalid attention backend you specified.\n"
                "Now only support [NATIVE_ATTN, MLA_ATTN, APPEND_ATTN] in cuda place."
            )
