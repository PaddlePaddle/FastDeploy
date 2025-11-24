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

import paddle

from fastdeploy.platforms import current_platform


def open_shm_and_get_meta_signal(
    rank: int = 0,
    device_id: int = 0,
    keep_pd_step_flag: bool = False,
) -> paddle.Tensor:
    """
    open_shm_and_get_meta_signal
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import open_shm_and_get_meta_signal

        out = open_shm_and_get_meta_signal(rank, device_id, keep_pd_step_flag)
        return out
    elif current_platform.is_xpu():
        from fastdeploy.model_executor.ops.xpu import open_shm_and_get_meta_signal

        out = open_shm_and_get_meta_signal(rank, device_id, keep_pd_step_flag)
        return out
    else:
        raise NotImplementedError
