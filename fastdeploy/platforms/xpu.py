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
import traceback

import paddle

from fastdeploy.utils import console_logger as logger

from .base import Platform, _Backend


class XPUPlatform(Platform):
    """
    xpu platform class
    """

    device_name = "xpu"

    @classmethod
    def available(self):
        """
        Check whether XPU is available.
        """
        try:
            assert paddle.is_compiled_with_xpu()
            assert len(paddle.static.xpu_places()) > 0
            return True
        except Exception as e:
            logger.warning(
                "You are using XPU version PaddlePaddle, but there is no XPU "
                "detected on your machine. Maybe CUDA devices is not set properly."
                f"\n Original Error is {e}, "
                f"{str(traceback.format_exc())}"
            )
            return False

    @classmethod
    def get_attention_backend_cls(cls, selected_backend):
        """
        get_attention_backend_cls
        """
        # TODO: 等支持配置 attention engine 之后再改回去
        return "fastdeploy.model_executor.layers.attention.XPUAttentionBackend"
        if selected_backend == _Backend.NATIVE_ATTN:
            return "fastdeploy.model_executor.layers.attention.XPUAttentionBackend"
        else:
            logger.warning("Other backends are not supported for now for XPU.")
