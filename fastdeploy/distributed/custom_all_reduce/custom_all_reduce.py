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


import atexit
import ctypes
from contextlib import contextmanager
from typing import List, Optional

import paddle
import paddle.distributed as dist
from paddle.distributed.communication.group import Group

from fastdeploy.distributed.custom_all_reduce import cuda_wrapper
from fastdeploy.model_executor.ops.gpu import (
    all_reduce,
    clear_ipc_handles,
    dispose,
    get_graph_buffer_ipc_meta,
    init_custom_all_reduce,
    meta_size,
    register_buffer,
    register_graph_buffers,
)

try:
    meta_size()
    custom_ar = True
except Exception:
    custom_ar = False

_instances = []


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self, group: Group, max_size: int = 8192 * 1024) -> None:
        """
        Args:
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.capturing = False
        self.group = group

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            return

        if world_size < 2:
            return

        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(group, meta_size() + max_size)

        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(group, max_size)

        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = paddle.empty([8 * 1024 * 1024], dtype=paddle.uint8)

        self.max_size = max_size
        self.world_size = world_size
        self.full_nvlink = True
        self._ptr = init_custom_all_reduce(self.meta_ptrs, self.rank_data, rank, self.full_nvlink)
        register_buffer(self._ptr, self.buffer_ptrs)

        _instances.append(self)

    @staticmethod
    def create_shared_buffer(group: Group, size_in_bytes: int) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = cuda_wrapper.CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        rank = dist.get_rank(group=group)
        handles = []
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(group: Group, pointers: List[int], rank: Optional[int] = None) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        lib = cuda_wrapper.CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        
    # def should_custom_ar(self, inp: paddle.Tensor):
    #     if self.capturing:
    #         return True
    #     inp_size = inp.shape[0] * inp.shape[1] * inp.element_size()
    #     # custom allreduce requires input byte size to be multiples of 16
    #     if inp_size % 16 != 0:
    #         return False
    #     # for 4 or more non NVLink-capable GPUs, custom allreduce provides
    #     # little performance improvement over NCCL.
    #     if self.world_size == 2 or self.full_nvlink:
    #         return inp_size < self.max_size
    #     return False

    # def should_custom_ar(self, inp: paddle.Tensor):
    #     if self.capturing:
    #         return True
    #     # inp_size = inp.shape[0] * inp.shape[1] * inp.element_size()
    #     # --- [修复] ---
    #     # 1. 使用 paddle.numel() 计算总元素数，保证对任意维度张量都正确
    #     num_elements = paddle.numel(inp)
    #     inp_size = num_elements * inp.element_size()

    #     # 2. 增加一个最小尺寸阈值。对于非常小的张量，直接使用原生all_reduce更稳定且性能差异不大。
    #     #    这里的 256 字节是一个经验值，可以调整。
    #     #    我们的 variance (16字节) 会在这里返回 False。
    #     MIN_SIZE_THRESHOLD = 256 
    #     if inp_size < MIN_SIZE_THRESHOLD:
    #         return False
    #     # --- [结束修复] ---
    #     # custom allreduce requires input byte size to be multiples of 16
    #     if inp_size % 16 != 0:
    #         return False
    #     # for 4 or more non NVLink-capable GPUs, custom allreduce provides
    #     # little performance improvement over NCCL.
    #     if self.world_size == 2 or self.full_nvlink:
    #         return inp_size < self.max_size
    #     return 
    
    def should_custom_ar(self, inp: paddle.Tensor):
        # --- [PR 建议修改] ---
        
        # 1. 直接获取 numel() 的 Python int 值，避免在 if 判断中隐式同步
        #    .shape 是一个 CPU 属性，访问它是图安全的
        #    .numel() 方法可能返回 tensor，而 .shape.numel() 总是返回 int
        if not isinstance(inp.shape, list):
            # In dygraph mode, shape is a list, in static mode, it might be something else
            # To be safe, we check.
            # In the context of FD, it should always be dygraph.
            # A more robust way might be needed if static graph is considered.
             try:
                num_elements = inp.shape.numel()
             except:
                # Fallback for static graph or other cases, might be unsafe in graph capture
                num_elements = paddle.numel(inp).item()
        else:
            # For list shape
            import math
            num_elements = math.prod(inp.shape)
        
        inp_size = num_elements * inp.element_size()

        # 2. 约束 a: 字节数必须是 16 的倍数
        if inp_size % 16 != 0:
            return False
            
        # 3. 约束 b: "input length to be multiple of 4" (假设指元素数量)
        #    这是为了修复 "multiple of 4" 的 RuntimeError
        if num_elements % 4 != 0:
            return False

        # 4. 约束 c: 增加一个最小尺寸阈值，过滤掉小张量
        #    这可以同时解决 variance 的数值问题和形状限制问题
        MIN_SIZE_THRESHOLD = 256 
        if inp_size < MIN_SIZE_THRESHOLD:
            return False
            
        # 5. 约束 d: 不能超过最大尺寸限制
        if inp_size >= self.max_size:
            return False

        # 6. 硬件和 world_size 的约束
        if self.world_size > 2 and not self.full_nvlink:
            return False

        # 7. 如果所有检查都通过了，才返回 True。
        #    这个修改移除了 `if self.capturing: return True` 的逻辑漏洞。
        return True
        # --- [结束修改] ---

    def all_reduce(
        self,
        inp: paddle.Tensor,
        out: paddle.Tensor = None,
        registered: bool = False,
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = paddle.empty_like(inp)
        if registered:
            all_reduce(inp, out, self._ptr, 0, 0)
        else:
            all_reduce(inp, out, self._ptr, self.buffer_ptrs[self.rank], self.max_size)
        return out

    def start_capture(self):
        """
        set CUDA graph flag: True.
        """
        self.capturing = True

    def stop_capture(self):
        """
        set CUDA graph flag: False and register the graph buffers.
        """
        self.capturing = False
        self.register_graph_buffers()

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self.capturing = True
            yield
        finally:
            self.capturing = False
            self.register_graph_buffers()

    def register_graph_buffers(self):
        """
        Register the graph buffers collected CUDA graph during capture.
        """
        handle, offset = get_graph_buffer_ipc_meta(self._ptr)
        all_datas = []
        all_data = [handle, offset]

        dist.all_gather_object(all_datas, all_data, group=self.group)

        handles = [d[0] for d in all_datas]
        offsets = [d[1] for d in all_datas]
        register_graph_buffers(self._ptr, handles, offsets)

    def custom_all_reduce(self, input: paddle.Tensor) -> Optional[paddle.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        if self.capturing:
            lib = cuda_wrapper.CudaRTLibrary()
            stream = paddle.device.current_stream()
            stream_capturing = lib.cudaStreamIsCapturing(stream)
            if stream_capturing.value == 1:
                # 1 is cudaStreamCaptureStatusActive: The stream is capturing.
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return paddle.empty_like(input)
        else:
            return self.all_reduce(input, registered=False)

    def clear_ipc_handles(self):
        clear_ipc_handles(self._ptr)

    def close(self):
        if self._ptr:
            dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.group, self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.group, self.buffer_ptrs, rank=self.rank)


def _cleanup_instances():
    for instance in _instances:
        instance.close()


atexit.register(_cleanup_instances)
