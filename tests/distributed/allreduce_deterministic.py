#!/usr/bin/env python
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
All-Reduce Deterministic Test with Real Communication

Tests:
1. Custom All-Reduce is deterministic for supported dtypes (float32, float16, bfloat16)
2. Non-16 byte aligned tensors raise RuntimeError in deterministic mode
3. Unsupported dtypes (int32) raise AssertionError in deterministic mode

Run:
    python -m paddle.distributed.launch --gpus=0,1,2,3 tests/distributed/allreduce_deterministic.py
"""

import os

import paddle
import paddle.distributed as dist
import pytest

pytestmark = pytest.mark.gpu

from fastdeploy import envs
from fastdeploy.distributed import communication
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce

SUPPORTED_DTYPES = [paddle.float32, paddle.float16, paddle.bfloat16]
TENSOR_SIZE = 2048
NUM_RUNS = 20


def _create_tensor(size: int, dtype: paddle.dtype, rank: int) -> paddle.Tensor:
    """Create a test tensor with appropriate dtype and scaling."""
    if dtype == paddle.int32:
        return paddle.randint(-100, 100, shape=[size, 1], dtype=dtype) * (rank + 1)
    return paddle.randn([size, 1], dtype=dtype) * (rank + 1)


def _check_results_identical(results: list) -> bool:
    """Check if all results are identical."""
    if not results:
        return True
    return all((results[0] == r).all() for r in results[1:])


def _init_custom_allreduce(world_size: int):
    """Initialize custom all-reduce for testing."""
    mp_group = dist.new_group(ranks=list(range(world_size)))
    communication.use_custom_allreduce(mp_group, 8192 * 1024)
    return mp_group


def _enable_deterministic_mode():
    """Enable deterministic mode via environment variable."""
    os.environ["FD_DETERMINISTIC_MODE"] = "1"
    assert envs.FD_DETERMINISTIC_MODE, f"FD_DETERMINISTIC_MODE should be True but got {envs.FD_DETERMINISTIC_MODE}"


def test_custom_allreduce_deterministic(rank, world_size, dtype):
    """Custom all-reduce should be deterministic."""
    _mp_group = _init_custom_allreduce(world_size)  # noqa: F841
    results = []

    for _ in range(NUM_RUNS):
        paddle.seed(42 + rank)
        x = _create_tensor(TENSOR_SIZE, dtype, rank)
        result = tensor_model_parallel_all_reduce(x)
        results.append(result.numpy().copy())
        dist.barrier()

    communication.custom_ar_clear_ipc_handles()
    return _check_results_identical(results)


def test_non_16_aligned_raises_error(rank, world_size):
    """Non-16 byte aligned tensors should raise RuntimeError in deterministic mode."""
    _enable_deterministic_mode()
    mp_group = _init_custom_allreduce(world_size)
    # 1026 * 4 = 4104 bytes (NOT multiple of 16)
    x = paddle.to_tensor([1.0] * 1026, dtype=paddle.float32).reshape([1026, 1])

    try:
        with pytest.raises(RuntimeError, match="DETERMINISTIC_MODE.*multiple of 16"):
            tensor_model_parallel_all_reduce(x, group_=mp_group)
    finally:
        communication.custom_ar_clear_ipc_handles()


def test_unsupported_dtype_raises_error(rank, world_size):
    """Unsupported dtypes should raise AssertionError in deterministic mode."""
    _enable_deterministic_mode()
    mp_group = _init_custom_allreduce(world_size)
    x = _create_tensor(TENSOR_SIZE, paddle.int32, rank)

    try:
        with pytest.raises(AssertionError, match="DETERMINISTIC_MODE.*not supported"):
            tensor_model_parallel_all_reduce(x, group_=mp_group)
    finally:
        communication.custom_ar_clear_ipc_handles()


def main():
    if not dist.is_initialized():
        paddle.distributed.init_parallel_env()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size >= 2, f"Test requires at least 2 GPUs, got {world_size}"

    print(f"All-Reduce Deterministic Test (world_size={world_size}, runs={NUM_RUNS})")

    # Error path tests
    test_non_16_aligned_raises_error(rank, world_size)
    print("PASS: non-16 byte aligned tensor raises RuntimeError")
    dist.barrier()

    test_unsupported_dtype_raises_error(rank, world_size)
    print("PASS: unsupported dtype (int32) raises AssertionError")
    dist.barrier()

    # Determinism tests for supported dtypes
    for dtype in SUPPORTED_DTYPES:
        assert test_custom_allreduce_deterministic(
            rank, world_size, dtype
        ), f"Custom all-reduce is NOT deterministic for {dtype}"
        print(f"PASS: custom all-reduce deterministic for {dtype}")
        dist.barrier()

    print("All tests passed.")


if __name__ == "__main__":
    main()
