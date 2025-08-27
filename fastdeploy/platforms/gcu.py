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

import traceback

import paddle

from fastdeploy.utils import console_logger as logger

from .base import Platform, _Backend


class GCUPlatform(Platform):
    """
    gcu platform class
    """

    device_name = "gcu"

    @classmethod
    def available(self):
        """
        Check whether GCU is available.
        """
        try:
            assert paddle.base.core.get_custom_device_count("gcu") > 0
            return True
        except Exception as e:
            logger.warning(
                "You are using GCUPlatform, but there is no GCU "
                "detected on your machine. Maybe GCU devices is not set properly."
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
            logger.info("Using GCU mem_efficient ATTN backend.")
            return "fastdeploy.model_executor.layers.backends.gcu.attention.mem_efficient_attn_backend.GCUMemEfficientAttnBackend"
        elif selected_backend == _Backend.APPEND_ATTN:
            logger.info("Using GCU ATTN backend.")
            return "fastdeploy.model_executor.layers.backends.gcu.attention.flash_attn_backend.GCUFlashAttnBackend"
        else:
            raise ValueError(
                "Invalid attention backend you specified.\n"
                "Now only support [NATIVE_ATTN, APPEND_ATTN] in gcu place."
            )
