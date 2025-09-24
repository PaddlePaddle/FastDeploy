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
import paddle

from fastdeploy.utils import console_logger as logger

from .base import Platform, _Backend


class INTEL_HPUPlatform(Platform):
    device_name = "intel_hpu"

    @classmethod
    def available(self):
        """
        Check whether Intel HPU is available.
        """
        try:
            assert paddle.base.core.get_custom_device_count("intel_hpu") > 0
            return True
        except Exception as e:
            logger.warning(
                "You are using Intel HPU platform, but there is no Intel HPU "
                "detected on your machine. Maybe Intel HPU devices is not set properly."
                f"\n Original Error is {e}"
            )
            return False

    @classmethod
    def get_attention_backend_cls(cls, selected_backend):
        """
        get_attention_backend_cls
        """
        if selected_backend == _Backend.NATIVE_ATTN:
            logger.info("Using NATIVE ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.PaddleNativeAttnBackend"
        elif selected_backend == _Backend.HPU_ATTN:
            logger.info("Using HPU ATTN backend.")
            return "fastdeploy.model_executor.layers.backends.intel_hpu.attention.HPUAttentionBackend"
        else:
            logger.warning("Other backends are not supported for now.")
