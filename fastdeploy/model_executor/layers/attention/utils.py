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
from typing import TYPE_CHECKING

from fastdeploy.config import FDConfig
from fastdeploy.platforms import current_platform

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import paddle


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


def split_decodes_and_prefills(
    forward_meta: "ForwardMeta",
):
    """
    Prefill and decode phases are split to allow the use of specialized attention kernels.
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens
    """
    num_running_requests = forward_meta.seq_lens_this_time.shape[0]
    seq_lens_encoder_running = forward_meta.seq_lens_encoder[:num_running_requests]
    actual_token_nums = paddle.sum(forward_meta.seq_lens_this_time).item()

    if seq_lens_encoder_running[0].item() > 0:
        return 0, num_running_requests, 0, actual_token_nums

    if seq_lens_encoder_running[-1].item() == 0:
        return num_running_requests, 0, actual_token_nums, 0

    is_prefill = seq_lens_encoder_running > 0
    first_prefill = is_prefill.int().argmax().item()
    num_decodes = first_prefill
    num_prefills = num_running_requests - num_decodes
    num_decode_tokens = paddle.sum(forward_meta.seq_lens_this_time[:first_prefill]).item()
    num_prefill_tokens = actual_token_nums - num_decode_tokens
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens
