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

import contextlib
from dataclasses import dataclass

import paddle
import pynvml

from fastdeploy.platforms import current_platform


@dataclass
class PaddleMemoryInfo:
    # Max memory reserved by Paddle
    max_reserved: int = 0
    # Max memory allocated by Paddle
    max_allocated: int = 0
    # Current memory reserved by Paddle
    current_reserved: int = 0
    # Current memory allocated by Paddle
    current_allocated: int = 0


class GPUMemoryChecker:
    def __init__(
        self,
        device: int = 0,  # logic device id
        device_id: int = 0,  # physical device id
        print_debug_info: bool = True,
    ):
        self.gpu_memory_info = None
        self.paddle_memory_info = None
        self.device = device
        self.device_id = device_id
        self.print_debug_info = print_debug_info

        if current_platform.is_iluvatar():
            self.gpu_memory_handle = None
        else:
            pynvml.nvmlInit()
            self.gpu_memory_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

    def __del__(self):
        """ """
        if self.gpu_memory_handle is None:
            pass
        else:
            pynvml.nvmlShutdown()

    def _print_memory_info(
        self,
        debug_title: str = "",
    ):
        """Print debug info"""
        print(
            f"\n{debug_title}:",
            f"\n\tDevice Total memory: {self.gpu_memory_info.total}",
            f"\n\tDevice Used memory: {self.gpu_memory_info.used}",
            f"\n\tDevice Free memory: {self.gpu_memory_info.free}",
            f"\n\tPaddle max memory Reserved: {self.paddle_memory_info.max_reserved}",
            f"\n\tPaddle max memory Allocated: {self.paddle_memory_info.max_allocated}",
            f"\n\tPaddle memory Reserved: {self.paddle_memory_info.current_reserved}",
            f"\n\tPaddle memory Allocated: {self.paddle_memory_info.current_reserved}",
        )

    def get_gpu_memory_info(self):
        """Get Device memory information"""
        current_meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_memory_handle)

        return current_meminfo

    def get_paddle_memory_info(self) -> PaddleMemoryInfo:
        """Get GPU memory information managed by Paddle"""
        current_paddle_memory_info = PaddleMemoryInfo()
        current_paddle_memory_info.max_reserved = paddle.device.cuda.max_memory_reserved(self.device)
        current_paddle_memory_info.max_allocated = paddle.device.cuda.max_memory_allocated(self.device)
        current_paddle_memory_info.reserved = paddle.device.cuda.memory_reserved(self.device)
        current_paddle_memory_info.allocated = paddle.device.cuda.memory_allocated(self.device)

        return current_paddle_memory_info

    def _check_memory(self):
        """Check current device memory usage with pre checkpoint"""
        current_gpu_memory_info = self.get_gpu_memory_info()
        current_paddle_memory_info = self.get_paddle_memory_info()

        if self.gpu_memory_info is not None and self.paddle_memory_info is not None:
            assert (
                current_paddle_memory_info.max_reserved <= self.paddle_memory_info.max_reserved
            ), f"Memory Check Failed! Current checkpoint Padddle memory usage ({current_paddle_memory_info.max_reserved}) must be less than or equal to the previous one ({self.paddle_memory_info.max_reserved})."
            assert (
                current_gpu_memory_info.used <= self.gpu_memory_info.used
            ), f"Memory Check Failed! Current checkpoint GPU memory usage ({current_gpu_memory_info.used}) must be less than or equal to the previous one ({self.gpu_memory_info.used})."

        self.gpu_memory_info = current_gpu_memory_info
        self.paddle_memory_info = current_paddle_memory_info

    def add_check_point(
        self,
        debug_title: str = "",
    ):
        """Add checkpoints for GPU memory usage"""
        self._check_memory()
        if self.print_debug_info:
            self._print_memory_info(debug_title)


def create_guard(default_value):
    _state = default_value

    @contextlib.contextmanager
    def state_guard(current_state):
        nonlocal _state
        old_state = _state
        _state = current_state
        try:
            yield
        finally:
            _state = old_state

    def get_state():
        return _state

    return state_guard, get_state


sot_warmup_guard, in_sot_warmup_mode = create_guard(False)
profile_run_guard, in_profile_run_mode = create_guard(False)
