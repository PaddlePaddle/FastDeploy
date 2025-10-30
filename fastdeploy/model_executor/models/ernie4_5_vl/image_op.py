"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from fastdeploy.platforms import current_platform
from fastdeploy.utils import llm_logger as logger

if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import (
        text_image_gather_scatter,
        text_image_index_out,
    )
elif current_platform.is_xpu():
    from fastdeploy.model_executor.ops.xpu import (
        text_image_gather_scatter,
        text_image_index_out,
    )
elif current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        text_image_gather_scatter,
        text_image_index_out,
    )
else:
    text_image_gather_scatter = None
    text_image_index_out = None
    logger.warning("Unsupported platform, image ops only support CUDA, XPU, MACA and Iluvatar")

__all__ = ["text_image_gather_scatter", "text_image_index_out"]
