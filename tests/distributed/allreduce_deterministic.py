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
1. Custom All-Reduce is deterministic for supported dtypes (float32, float16, bfloat16) and
2. Common nccl allreduce is not deterministic.
3. Non-16 byte aligned tensors raise RuntimeError in deterministic mode
4. Unsupported dtypes (int32) raise AssertionError in deterministic mode

Run with 2 GPUs:
    python -m paddle.distributed.launch --gpus=0,1,2,3 tests/distributed/allreduce_deterministic.py
"""

import os
from dataclasses import dataclass

import paddle
import paddle.distributed as dist
import pytest

pytestmark = pytest.mark.gpu

from fastdeploy import envs
from fastdeploy.distributed import communication
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce


@dataclass
class DataTypeInfo:
    name: str
    dtype: paddle.dtype
    element_size: int
    supported: bool


DATA_TYPES = [
    DataTypeInfo("float32", paddle.float32, 4, True),
    DataTypeInfo("float16", paddle.float16, 2, True),
    DataTypeInfo("bfloat16", paddle.bfloat16, 2, True),
    DataTypeInfo("int32", paddle.int32, 4, False),
]

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


def test_nccl_allreduce_non_deterministic(rank, world_size, dtype):
    """NCCL all-reduce may produce non-deterministic results."""
    results = []

    for i in range(NUM_RUNS):
        paddle.seed(i * 100 + rank)
        if dtype == paddle.int32:
            x = paddle.randint(-100, 100, shape=[TENSOR_SIZE], dtype=dtype) * (rank + 1)
        else:
            x = paddle.randn([TENSOR_SIZE], dtype=dtype) * (rank + 1)

        dist.all_reduce(x)
        results.append(x.numpy().copy())
        dist.barrier()

    return _check_results_identical(results)


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
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    if not envs.FD_DETERMINISTIC_MODE:
        raise AssertionError(f"FD_DETERMINISTIC_MODE should be True but got {envs.FD_DETERMINISTIC_MODE}")

    mp_group = _init_custom_allreduce(world_size)
    # 1026 * 4 = 4104 bytes (NOT multiple of 16)
    x = paddle.to_tensor([1.0] * 1026, dtype=paddle.float32).reshape([1026, 1])

    try:
        with pytest.raises(RuntimeError, match="DETERMINISTIC_MODE.*multiple of 16"):
            tensor_model_parallel_all_reduce(x, group_=mp_group)
        return True
    finally:
        communication.custom_ar_clear_ipc_handles()


def test_unsupported_dtype_raises_error(rank, world_size):
    """Unsupported dtypes should raise AssertionError in deterministic mode."""
    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    if not envs.FD_DETERMINISTIC_MODE:
        raise AssertionError(f"FD_DETERMINISTIC_MODE should be True but got {envs.FD_DETERMINISTIC_MODE}")

    mp_group = _init_custom_allreduce(world_size)
    x = _create_tensor(TENSOR_SIZE, paddle.int32, rank)

    try:
        with pytest.raises(AssertionError, match="DETERMINISTIC_MODE.*not supported"):
            tensor_model_parallel_all_reduce(x, group_=mp_group)
        return True
    finally:
        communication.custom_ar_clear_ipc_handles()


def _run_single_dtype_test(rank: int, world_size: int, dtype_info: DataTypeInfo) -> dict:
    """Run tests for a single data type."""
    if not dtype_info.supported:
        return {"custom_deterministic": None, "nccl_deterministic": None}

    print(f"\n{'='*70}")
    print(f"Testing {dtype_info.name}")
    print(f"  Element Size: {dtype_info.element_size} bytes")
    print(f"  Total Bytes: {TENSOR_SIZE * dtype_info.element_size}")
    print(f"{'='*70}")

    dist.barrier()

    # Custom All-Reduce test
    print(f"\n--- Custom All-Reduce Test ({dtype_info.name}) ---")
    try:
        custom_same = test_custom_allreduce_deterministic(rank, world_size, dtype_info.dtype)
        status = "✅ PASS" if custom_same else "❌ FAIL"
        print(f"  Deterministic: {custom_same} {status}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        custom_same = False

    dist.barrier()

    # NCCL test
    print(f"\n--- NCCL All-Reduce Test ({dtype_info.name}) ---")
    try:
        nccl_same = test_nccl_allreduce_non_deterministic(rank, world_size, dtype_info.dtype)
        if not nccl_same:
            print("  Non-deterministic: ✅ PASS")
        else:
            print("  Identical: ⚠️  Note (may occur with small data)")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        nccl_same = False

    dist.barrier()

    return {"custom_deterministic": custom_same, "nccl_deterministic": nccl_same}


def main():
    if not dist.is_initialized():
        paddle.distributed.init_parallel_env()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size >= 2, f"Test requires at least 2 GPUs, got {world_size}"

    print(f"\n{'='*70}")
    print("All-Reduce Deterministic Test")
    print(f"{'='*70}")
    print(f"  World Size: {world_size}")
    print(f"  Runs per test: {NUM_RUNS}")
    print(f"{'='*70}")

    # Test 1: Non-16 byte aligned error
    print("\n--- Test 1: Non-16 Byte Aligned Tensor ---")
    test_non_16_aligned_raises_error(rank, world_size)
    dist.barrier()

    # Test 2: Unsupported dtype error
    print("\n--- Test 2: Unsupported dtype (int32) ---")
    test_unsupported_dtype_raises_error(rank, world_size)
    dist.barrier()

    # Test 3: Supported dtypes determinism
    results = {}
    for dtype_info in DATA_TYPES:
        results[dtype_info.name] = _run_single_dtype_test(rank, world_size, dtype_info)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Data Type':<15} | {'Custom AR Deterministic':<25} | {'NCCL Deterministic':<20}")
    print("-" * 70)

    for dtype_info in DATA_TYPES:
        result = results[dtype_info.name]
        if not dtype_info.supported:
            custom_status = "❌ N/A (not supported)"
            nccl_status = "❌ N/A (not supported)"
        else:
            custom_status = "✅ YES" if result["custom_deterministic"] else "❌ NO"
            nccl_status = "❌ NO" if not result["nccl_deterministic"] else "⚠️  YES"
        print(f"{dtype_info.name:<15} | {custom_status:<25} | {nccl_status:<20}")

    # Overall result
    supported_results = [dt.name for dt in DATA_TYPES if dt.supported]
    all_custom_pass = all(results[dt]["custom_deterministic"] for dt in supported_results)

    print(f"{'='*70}")
    if all_custom_pass:
        print("✅ Custom All-Reduce is deterministic for all supported types!")
    else:
        print("❌ Custom All-Reduce failed determinism test!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
