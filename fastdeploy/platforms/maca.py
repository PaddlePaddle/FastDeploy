"""
# Copyright (c) 2025 MetaX-tech Authors. All Rights Reserved.
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

"""
maca platform file
"""
import traceback

import paddle
from paddleformers.utils.log import logger

from .base import Platform, _Backend


class MACAPlatform(Platform):
    """
    maca platform class
    """

    device_name = "metax_gpu"

    @classmethod
    def available(self):
        """
        Check whether MACA is available.
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
        get_attention_backend_cls
        """
        if selected_backend == _Backend.NATIVE_ATTN:
            logger.info("Using NATIVE ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.PaddleNativeAttnBackend"
        elif selected_backend == _Backend.APPEND_ATTN:
            logger.info("Using FLASH ATTN backend to instead of attend attention.")
            return "fastdeploy.model_executor.layers.backends.metax.attention.flash_attn_backend.FlashAttentionBackend"
        elif selected_backend == _Backend.MLA_ATTN:
            logger.info("Using MLA ATTN backend.")
            return "fastdeploy.model_executor.layers.backends.metax.attention.mla_attn_metax_backend.MetaxMLAAttentionBackend"
        else:
            raise ValueError(
                "Invalid attention backend you specified.\n"
                "Now only support [NATIVE_ATTN, MLA_ATTN, APPEND_ATTN] in cuda place."
            )
