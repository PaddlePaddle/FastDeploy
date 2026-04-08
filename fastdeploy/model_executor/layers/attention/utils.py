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

import os

from fastdeploy.config import FDConfig
from fastdeploy.platforms import current_platform


def init_rank_and_device_id(fd_config: FDConfig):
    """ """
    rank = (
        fd_config.parallel_config.local_data_parallel_id * fd_config.parallel_config.tensor_parallel_size
        + fd_config.parallel_config.tensor_parallel_rank
    )

    if current_platform.is_xpu():
        cuda_visible_devices = os.getenv("XPU_VISIBLE_DEVICES", None)
    else:  # default cuda
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)

    if cuda_visible_devices is None:
        device_id = rank
    else:
        cuda_visible_devices = cuda_visible_devices.split(",")
        rank_index = rank % len(cuda_visible_devices)
        device_id = cuda_visible_devices[rank_index]

    return rank, device_id
