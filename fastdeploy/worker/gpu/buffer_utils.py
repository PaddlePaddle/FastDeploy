"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

from collections.abc import Iterable
from functools import partial

import numpy as np
import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.gpu import (
    copy_array_to_tensor,
    get_cuda_view_from_cpu_tensor,
    numpy_view_of_cpu_tensor,
)

paddle_to_numpy_dtype = {
    paddle.float32: np.float32,
    paddle.float64: np.float64,
    paddle.float16: np.float16,
    paddle.int64: np.int64,
    paddle.int32: np.int32,
    paddle.int16: np.int16,
    paddle.int8: np.int8,
    paddle.uint8: np.uint8,
    paddle.bool: np.bool_,
}


def async_to_tensor(x: int | float | bool | list | np.ndarray | paddle.Tensor, dtype=None) -> paddle.Tensor:
    if isinstance(x, (int, float, bool)):
        return paddle.full([1], fill_value=x, dtype=dtype)
    elif isinstance(x, list):
        x_np = np.array(x)
        x_tensor = paddle.empty(x_np.shape, dtype=str(x_np.dtype))
        return copy_array_to_tensor(x_np, x_tensor)
    elif isinstance(x, np.ndarray):
        x_tensor = paddle.empty(x.shape, dtype=str(x.dtype))
        return copy_array_to_tensor(x, x_tensor)
    elif isinstance(x, paddle.Tensor):
        return x
    else:
        raise ValueError("async_to_tensor unsupported type: {}".format(type(x)))


def async_set_value(tgt: paddle.Tensor, src: int | float | bool | list | np.ndarray | paddle.Tensor) -> None:
    if not tgt.place.is_gpu_place():
        raise ValueError("async_set_value tgt place must be paddle.CUDAPlace")
    if isinstance(src, (int, float, bool)):
        # if src is not paddle.Tensor, convert it to paddle.Tensor first
        src = paddle.full(tgt.shape, fill_value=src, dtype=tgt.dtype)
    elif isinstance(src, (list, np.ndarray)):
        # if src is np.array, copy_array_to_tensor will be called
        dtype_str = str(tgt.dtype).split(".")[1]
        if isinstance(src, list):
            src = np.array(src, dtype=dtype_str if dtype_str != "bfloat16" else "float32")
        if str(src.dtype) != dtype_str:
            srt_tensor = paddle.empty(src.shape, dtype=str(src.dtype))
            src = copy_array_to_tensor(src, srt_tensor)
        else:
            return copy_array_to_tensor(src, tgt)
    elif isinstance(src, paddle.Tensor):
        pass
    else:
        raise ValueError("async_set_value unsupported src type: {}".format(type(src)))
    if src.shape != tgt.shape:
        src = src.reshape(tgt.shape)
    if src.dtype != tgt.dtype:
        src = src.cast(tgt.dtype)
    if src.place != tgt.place:
        src = src.to(tgt.place)
    tgt.copy_(src, blocking=False)


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(
        self,
        shape: int | list[int] | tuple[int],
        dtype: paddle.dtype,
        init_value: bool | int | float = 0,
        pin_memory: bool = True,
        with_numpy: bool = True,
        device: str = "cuda",
    ) -> None:
        # Only pin memory for CUDA devices (check if device starts with "cuda")
        # TODO: Temporarily disable pin memory due to Paddle CPU place limitation
        actual_pin_memory = False
        self.cpu = paddle.full(shape, init_value, dtype=dtype, device="cpu", pin_memory=actual_pin_memory)
        self.gpu = paddle.full(shape, init_value, dtype=dtype)
        self.np: np.ndarray
        if with_numpy:
            if dtype == paddle.bfloat16:
                raise ValueError(
                    "Bfloat16 paddle tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = numpy_view_of_cpu_tensor(self.cpu)

    def copy_to_gpu(self, n: int | None = None) -> paddle.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: int | None = None) -> paddle.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


class UvaBuffer:
    def __init__(self, shape: list[int], dtype: paddle.dtype, init_value: bool | int | float = 0):
        self.cpu = paddle.full(shape, init_value, dtype=dtype, device="cpu").pin_memory()
        self.np = numpy_view_of_cpu_tensor(self.cpu)
        self.uva = get_cuda_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    def __init__(
        self,
        shape: int | list[int] | tuple[int],
        dtype: paddle.dtype,
        init_value: bool | int | float = 0,
        max_concurrency: int = 2,
    ):
        self.shape = shape
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(shape, dtype, init_value) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: paddle.Tensor | np.ndarray | list) -> paddle.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, paddle.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: paddle.Tensor | np.ndarray,
        out: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, blocking=False)


class UvaBackedTensor:
    def __init__(
        self,
        shape: int | list[int] | tuple[int],
        dtype: paddle.dtype,
        init_value: bool | int | float = 0,
        max_concurrency: int = 2,
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = paddle.full(shape, init_value, dtype=dtype, device="cpu").pin_memory()
        self.np = numpy_view_of_cpu_tensor(self.cpu)

        # Buffers for concurrency
        self.pool = UvaBufferPool(shape, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> paddle.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        shape: int | list[int] | tuple[int],
        dtype: paddle.dtype,
        init_value: bool | int | float = 0,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.num_rows = self.shape[0]
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = paddle.full(shape, init_value, dtype=dtype)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(shape, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=paddle.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=paddle.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=paddle.int32)

    def stage_write(self, index: int, start: int, x: Iterable[int] | Iterable[float]) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)  # 目标行索引
        self._staged_write_starts.append(start)  # 行内起始位置
        self._staged_write_contents.extend(x)  # 要写入的数据
        self._staged_write_cu_lens.append(len(self._staged_write_contents))  # 累积长度

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = paddle.empty(len(self._staged_write_contents), dtype=self.dtype)
        async_set_value(write_contents, self._staged_write_contents)

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask)
