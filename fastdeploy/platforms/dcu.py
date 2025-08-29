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
dcu platform file
"""
import traceback

import paddle
from paddleformers.utils.log import logger

from .base import Platform, _Backend


class DCUPlatform(Platform):
    """
    dcu platform class
    """

    device_name = "dcu"

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
    def get_attention_backend_cls(cls, selected_backend):
        """
        get_attention_backend_cls
        """
        if selected_backend == _Backend.NATIVE_ATTN:
            logger.info("Using NATIVE ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.PaddleNativeAttnBackend"
        elif selected_backend == _Backend.BLOCK_ATTN:
            logger.info("Using BLOCK ATTN backend.")
            return "fastdeploy.model_executor.layers.attention.BlockAttentionBackend"
        else:
            logger.warning("Other backends are not supported for now.")
