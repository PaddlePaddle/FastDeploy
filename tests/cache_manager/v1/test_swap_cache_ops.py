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

Unit tests for swap_cache_all_layers operator.

Tests cover:
- Data correctness verification (MD5 checksum before and after transfer)
- Transfer speed benchmark
- Both CPU->GPU (load) and GPU->CPU (evict) modes
"""

import ctypes
import hashlib
import random
import statistics
import unittest
from dataclasses import dataclass

import numpy as np
import paddle

# Import the ops under test
from fastdeploy.cache_manager.ops import cuda_host_alloc, swap_cache_all_layers


@dataclass
class TestConfig:
    """Test configuration for KV cache transfer."""

    num_layers: int = 4
    num_heads: int = 16
    head_dim: int = 128
    block_size: int = 64
    total_block_num: int = 128
    dtype: paddle.dtype = paddle.bfloat16

    @property
    def kv_shape(self):
        """KV cache shape: [total_block_num, num_heads, block_size, head_dim]"""
        return (self.total_block_num, self.num_heads, self.block_size, self.head_dim)

    @property
    def kv_cache_dim(self):
        """Single block K or V cache dimension size."""
        return self.head_dim * self.num_heads * self.block_size

    @property
    def element_size(self):
        """Size of each element in bytes."""
        dummy = paddle.zeros([], dtype=self.dtype)
        return dummy.element_size()

    @property
    def block_bytes(self):
        """Single block K or V size in bytes."""
        return self.kv_cache_dim * self.element_size

    @property
    def layer_bytes(self):
        """Single layer K+V total size in bytes."""
        return self.block_bytes * self.total_block_num * 2


def compute_md5(data: np.ndarray) -> str:
    """Compute MD5 checksum of numpy array data.

    Note: For bfloat16 data, we need to handle the fact that numpy
    doesn't have native bfloat16 support. We convert to uint16 to get
    the raw bytes for MD5 computation.
    """
    if data.dtype == np.float32:
        # Already float32, use directly
        return hashlib.md5(data.tobytes()).hexdigest()
    elif data.dtype == np.uint16 or str(data.dtype) == "bfloat16":
        # bfloat16 stored as uint16 in numpy, use raw bytes
        return hashlib.md5(data.tobytes()).hexdigest()
    else:
        # For other dtypes, convert to float32 for consistent comparison
        return hashlib.md5(data.astype(np.float32).tobytes()).hexdigest()


def init_test_data(
    config: TestConfig,
    num_blocks_to_transfer: int,
    use_random: bool = False,
    shuffle_blocks: bool = False,
    seed: int = 42,
):
    """
    Initialize test data for transfer.

    Args:
        config: Test configuration for KV cache transfer.
        num_blocks_to_transfer: Number of blocks to transfer.
        use_random: If True, use random tensor values instead of constant per-layer values.
        shuffle_blocks: If True, use randomly sampled non-consecutive block IDs.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (gpu_k_tensors, gpu_v_tensors, k_ptrs, v_ptrs, src_k_data, src_v_data, md5_sums)
    """
    device = "cuda"
    rng = random.Random(seed)

    if shuffle_blocks:
        # Non-consecutive GPU block IDs: randomly sample from the full GPU block pool
        # CPU block IDs must stay in [0, num_blocks_to_transfer) as CPU pinned memory
        # is allocated for exactly num_blocks_to_transfer contiguous slots.
        all_ids = list(range(config.total_block_num))
        gpu_block_ids = sorted(rng.sample(all_ids, num_blocks_to_transfer))
        cpu_block_ids = list(range(num_blocks_to_transfer))
    else:
        # Consecutive: 0, 1, 2, ..., num_blocks_to_transfer-1
        gpu_block_ids = list(range(num_blocks_to_transfer))
        cpu_block_ids = list(range(num_blocks_to_transfer))

    gpu_k_tensors = []
    gpu_v_tensors = []
    k_ptrs = []
    v_ptrs = []
    src_k_data = []
    src_v_data = []
    md5_sums = []

    bytes_per_block = config.kv_cache_dim * config.element_size

    for layer_idx in range(config.num_layers):
        if use_random:
            # Random values: use float32 seed-based generation then cast to target dtype
            paddle.seed(seed + layer_idx)
            src_k = paddle.randn(config.kv_shape, dtype=paddle.float32).cast(config.dtype)
            src_v = paddle.randn(config.kv_shape, dtype=paddle.float32).cast(config.dtype)
        else:
            # Constant values per layer for easier visual verification
            src_k = paddle.ones(config.kv_shape, dtype=config.dtype) * (layer_idx + 1)
            src_v = paddle.ones(config.kv_shape, dtype=config.dtype) * (layer_idx + 2)
        src_k_data.append(src_k)
        src_v_data.append(src_v)

        # Compute MD5 for verification (only for the cpu_block_ids blocks in source)
        # cpu_block_ids indicates which source blocks get copied into CPU pinned memory
        k_np = np.array(src_k)[cpu_block_ids]
        v_np = np.array(src_v)[cpu_block_ids]
        md5_sums.append((compute_md5(k_np), compute_md5(v_np)))

        # GPU tensors (destination for H2D, source for D2H)
        dst_k = paddle.zeros(config.kv_shape, dtype=config.dtype).to(device)
        dst_v = paddle.zeros(config.kv_shape, dtype=config.dtype).to(device)
        gpu_k_tensors.append(dst_k)
        gpu_v_tensors.append(dst_v)

        # Allocate CPU pinned memory
        k_ptr = cuda_host_alloc(bytes_per_block * num_blocks_to_transfer)
        v_ptr = cuda_host_alloc(bytes_per_block * num_blocks_to_transfer)

        # Fill CPU memory: pack the cpu_block_ids blocks contiguously
        k_np_full = np.array(src_k)
        v_np_full = np.array(src_v)
        k_np_flat = k_np_full[cpu_block_ids].flatten()
        v_np_flat = v_np_full[cpu_block_ids].flatten()
        ctypes.memmove(k_ptr, k_np_flat.ctypes.data, bytes_per_block * num_blocks_to_transfer)
        ctypes.memmove(v_ptr, v_np_flat.ctypes.data, bytes_per_block * num_blocks_to_transfer)

        k_ptrs.append(k_ptr)
        v_ptrs.append(v_ptr)

    total_transfer_bytes = num_blocks_to_transfer * config.block_bytes * config.num_layers * 2

    return (
        gpu_k_tensors,
        gpu_v_tensors,
        k_ptrs,
        v_ptrs,
        src_k_data,
        src_v_data,
        md5_sums,
        total_transfer_bytes,
        gpu_block_ids,
        cpu_block_ids,
    )


def verify_transfer_correctness(
    gpu_tensors,
    src_data_list,
    md5_sums,
    num_blocks_to_check,
    config: TestConfig,
    atol=1e-2,
    rtol=1e-2,
    gpu_block_ids=None,
    src_block_ids=None,
):
    """
    Verify transfer correctness by comparing data and MD5 checksums.

    Args:
        gpu_block_ids: indices of blocks on GPU that were written (H2D destination).
                       If None, defaults to 0..num_blocks_to_check-1 (consecutive).
        src_block_ids: indices into src_data_list tensors that correspond to the
                       source blocks (i.e. what was in CPU memory).
                       If None, defaults to 0..num_blocks_to_check-1 (consecutive).

    Returns:
        Tuple of (md5_passed, data_passed)
    """
    if gpu_block_ids is None:
        gpu_block_ids = list(range(num_blocks_to_check))
    if src_block_ids is None:
        src_block_ids = list(range(num_blocks_to_check))

    md5_passed = True
    data_passed = True

    for layer_idx in range(config.num_layers):
        gpu_data = gpu_tensors[layer_idx].cpu().numpy()
        # Only check the transferred blocks (by gpu_block_ids)
        gpu_data = gpu_data[gpu_block_ids]
        src_np = np.array(src_data_list[layer_idx])[src_block_ids]

        # Check MD5 checksum
        actual_md5 = compute_md5(gpu_data)
        expected_md5 = md5_sums[layer_idx]
        if actual_md5 != expected_md5:
            md5_passed = False

        # Check numerical correctness
        if not np.allclose(gpu_data, src_np, rtol=rtol, atol=atol):
            data_passed = False

    return md5_passed, data_passed


def benchmark_transfer(
    op_func,
    gpu_k_tensors,
    gpu_v_tensors,
    k_ptrs,
    v_ptrs,
    num_blocks,
    gpu_block_ids,
    cpu_block_ids,
    device_id,
    mode,
    num_warmup=2,
    num_iterations=5,
):
    """
    Benchmark transfer operation.

    Returns:
        Tuple of (avg_time_ms, all_times_ms)
    """
    # Warmup
    for _ in range(num_warmup):
        op_func(
            gpu_k_tensors,
            k_ptrs,
            num_blocks,
            gpu_block_ids,
            cpu_block_ids,
            device_id,
            mode,
        )
        op_func(
            gpu_v_tensors,
            v_ptrs,
            num_blocks,
            gpu_block_ids,
            cpu_block_ids,
            device_id,
            mode,
        )
    paddle.device.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = paddle.device.cuda.Event(enable_timing=True)
        end = paddle.device.cuda.Event(enable_timing=True)

        start.record()
        op_func(
            gpu_k_tensors,
            k_ptrs,
            num_blocks,
            gpu_block_ids,
            cpu_block_ids,
            device_id,
            mode,
        )
        op_func(
            gpu_v_tensors,
            v_ptrs,
            num_blocks,
            gpu_block_ids,
            cpu_block_ids,
            device_id,
            mode,
        )
        end.record()
        paddle.device.cuda.synchronize()

        times.append(start.elapsed_time(end))

    avg_time = statistics.mean(times)
    return avg_time, times


class TestSwapCacheAllLayersCorrectness(unittest.TestCase):
    """Test correctness of swap_cache_all_layers operator."""

    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("Swap cache ops test temporarily skipped")
        """Set up test environment."""
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available, skipping GPU tests")

    def setUp(self):
        """Set up each test."""
        self.config = TestConfig(
            num_layers=64,
            num_heads=16,
            head_dim=128,
            block_size=64,
            total_block_num=256,
        )
        self.device_id = 0
        self.num_blocks = 256  # Number of blocks to transfer in each test

    def test_h2d_transfer_correctness(self):
        """Test Host->Device (load) transfer correctness with MD5 verification."""
        (
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            src_k_data,
            src_v_data,
            md5_sums,
            _,
            gpu_block_ids,
            cpu_block_ids,
        ) = init_test_data(self.config, self.num_blocks)

        # Perform H2D transfer
        swap_cache_all_layers(
            gpu_k_tensors,
            k_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,  # Host->Device
        )
        swap_cache_all_layers(
            gpu_v_tensors,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
        )
        paddle.device.cuda.synchronize()

        # Verify correctness
        k_md5_ok, k_data_ok = verify_transfer_correctness(
            gpu_k_tensors, src_k_data, [m[0] for m in md5_sums], self.num_blocks, self.config
        )
        v_md5_ok, v_data_ok = verify_transfer_correctness(
            gpu_v_tensors, src_v_data, [m[1] for m in md5_sums], self.num_blocks, self.config
        )

        self.assertTrue(k_md5_ok, "K cache MD5 mismatch after H2D transfer")
        self.assertTrue(v_md5_ok, "V cache MD5 mismatch after H2D transfer")
        self.assertTrue(k_data_ok, "K cache data mismatch after H2D transfer")
        self.assertTrue(v_data_ok, "V cache data mismatch after H2D transfer")

    def test_d2h_transfer_correctness(self):
        """Test Device->Host (evict) transfer correctness."""
        (
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            src_k_data,
            src_v_data,
            md5_sums,
            _,
            gpu_block_ids,
            cpu_block_ids,
        ) = init_test_data(self.config, self.num_blocks)

        # First H2D to fill GPU
        swap_cache_all_layers(
            gpu_k_tensors,
            k_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
        )
        swap_cache_all_layers(
            gpu_v_tensors,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
        )
        paddle.device.cuda.synchronize()

        # Clear CPU memory (use uint16 to match bfloat16 storage)
        bytes_per_block = self.config.kv_cache_dim * self.config.element_size
        zero_data = np.zeros(self.num_blocks * self.config.kv_cache_dim, dtype=np.uint16)
        for k_ptr, v_ptr in zip(k_ptrs, v_ptrs):
            ctypes.memmove(k_ptr, zero_data.ctypes.data, bytes_per_block * self.num_blocks)
            ctypes.memmove(v_ptr, zero_data.ctypes.data, bytes_per_block * self.num_blocks)

        # Perform D2H transfer
        swap_cache_all_layers(
            gpu_k_tensors,
            k_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=0,  # Device->Host
        )
        swap_cache_all_layers(
            gpu_v_tensors,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=0,
        )
        paddle.device.cuda.synchronize()

        # Verify data in CPU memory
        bytes_per_layer = bytes_per_block * self.num_blocks
        k_md5_ok = True
        v_md5_ok = True

        for layer_idx in range(self.config.num_layers):
            # Read back from CPU memory (use uint16 to match bfloat16 storage)
            k_np = np.zeros(self.num_blocks * self.config.kv_cache_dim, dtype=np.uint16)
            v_np = np.zeros(self.num_blocks * self.config.kv_cache_dim, dtype=np.uint16)
            ctypes.memmove(k_np.ctypes.data, k_ptrs[layer_idx], bytes_per_layer)
            ctypes.memmove(v_np.ctypes.data, v_ptrs[layer_idx], bytes_per_layer)

            # Reshape to compare
            k_np = k_np.reshape(self.num_blocks, self.config.num_heads, self.config.block_size, self.config.head_dim)
            v_np = v_np.reshape(self.num_blocks, self.config.num_heads, self.config.block_size, self.config.head_dim)

            # Check MD5
            if compute_md5(k_np) != md5_sums[layer_idx][0]:
                k_md5_ok = False
            if compute_md5(v_np) != md5_sums[layer_idx][1]:
                v_md5_ok = False

        self.assertTrue(k_md5_ok, "K cache MD5 mismatch after D2H transfer")
        self.assertTrue(v_md5_ok, "V cache MD5 mismatch after D2H transfer")


class TestSwapCacheAllLayersPerformance(unittest.TestCase):
    """Test performance of swap_cache_all_layers operator."""

    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("Swap cache ops test temporarily skipped")

    def setUp(self):
        """Set up each test."""
        self.config = TestConfig(
            num_layers=64,
            num_heads=16,
            head_dim=128,
            block_size=64,
            total_block_num=256,
        )
        self.device_id = 0
        self.num_blocks = 256

    def test_h2d_bandwidth(self):
        """Test H2D transfer bandwidth."""
        (
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            _,
            _,
            _,
            total_bytes,
            gpu_block_ids,
            cpu_block_ids,
        ) = init_test_data(self.config, self.num_blocks)

        avg_time, _ = benchmark_transfer(
            swap_cache_all_layers,
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
            num_warmup=2,
            num_iterations=5,
        )

        bandwidth_gbps = (total_bytes / (1024**3)) / (avg_time / 1000)

        print("\n swap_cache_all_layers H2D Performance:")
        print(f"  Data size: {total_bytes / (1024**3):.2f} GB")
        print(f"  Avg time: {avg_time:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")

        # Sanity check: bandwidth should be > 1 GB/s
        self.assertGreater(bandwidth_gbps, 1.0)

    def test_d2h_bandwidth(self):
        """Test D2H transfer bandwidth."""
        (
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            _,
            _,
            _,
            total_bytes,
            gpu_block_ids,
            cpu_block_ids,
        ) = init_test_data(self.config, self.num_blocks)

        # First H2D to fill GPU
        swap_cache_all_layers(
            gpu_k_tensors,
            k_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
        )
        swap_cache_all_layers(
            gpu_v_tensors,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=1,
        )
        paddle.device.cuda.synchronize()

        avg_time, _ = benchmark_transfer(
            swap_cache_all_layers,
            gpu_k_tensors,
            gpu_v_tensors,
            k_ptrs,
            v_ptrs,
            self.config.total_block_num,
            gpu_block_ids,
            cpu_block_ids,
            self.device_id,
            mode=0,
            num_warmup=2,
            num_iterations=5,
        )

        bandwidth_gbps = (total_bytes / (1024**3)) / (avg_time / 1000)

        print("\n swap_cache_all_layers D2H Performance:")
        print(f"  Data size: {total_bytes / (1024**3):.2f} GB")
        print(f"  Avg time: {avg_time:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")

        self.assertGreater(bandwidth_gbps, 1.0)


@unittest.skip("Swap cache ops test temporarily skipped")
class TestSwapCacheRandomBlockIndices(unittest.TestCase):
    """
    Test swap operations with random, varying block indices per round.

    Simulates real-world cache eviction/loading patterns:
    - Each round picks a different random subset of blocks
    - Block count varies per round (e.g. 4~64 out of 128 total)
    - Verifies both swapped blocks (MD5 + allclose) and non-swapped blocks
    - Tests swap_cache_all_layers
    """

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available, skipping GPU tests")

    def setUp(self):
        self.config = TestConfig(
            num_layers=64,
            num_heads=16,
            head_dim=128,
            block_size=64,
            total_block_num=256,
        )
        self.device_id = 0
        self.num_rounds = 10
        self.min_blocks = 32
        self.max_blocks = 128
        self.seed = 2025

    def _init_all_gpu_blocks(self):
        """Initialize ALL GPU blocks with unique random data. Returns ground truth numpy arrays."""
        config = self.config
        gpu_k, gpu_v, gt_k, gt_v = [], [], [], []
        for li in range(config.num_layers):
            paddle.seed(self.seed + li * 1000)
            k = paddle.randn(config.kv_shape, dtype=paddle.float32).cast(config.dtype)
            v = paddle.randn(config.kv_shape, dtype=paddle.float32).cast(config.dtype)
            gt_k.append(np.array(k).copy())
            gt_v.append(np.array(v).copy())
            gpu_k.append(k.to("cuda"))
            gpu_v.append(v.to("cuda"))
        paddle.device.cuda.synchronize()
        return gpu_k, gpu_v, gt_k, gt_v

    def _snapshot_non_swap_blocks(self, gpu_k, gpu_v, swap_ids, rng):
        """Snapshot a few non-swapped blocks for later corruption check."""
        non_swap = [i for i in range(self.config.total_block_num) if i not in set(swap_ids)]
        check_ids = sorted(rng.sample(non_swap, min(5, len(non_swap))))
        snapshots = {}
        for name, tensors in [("k", gpu_k), ("v", gpu_v)]:
            for li in range(self.config.num_layers):
                data = tensors[li].cpu().numpy()
                for bid in check_ids:
                    snapshots[(name, li, bid)] = data[bid].copy()
        return snapshots

    def _zero_gpu_blocks(self, gpu_k, gpu_v, block_ids):
        """Zero out specific blocks on GPU via numpy round-trip."""
        for t in gpu_k + gpu_v:
            arr = t.cpu().numpy().copy()
            for bid in block_ids:
                arr[bid] = 0
            t.copy_(paddle.to_tensor(arr, place=t.place))
        paddle.device.cuda.synchronize()

    def _verify_cpu_against_gt(self, k_ptrs, v_ptrs, gt_k, gt_v, swap_ids, num_blocks, label):
        """Read CPU pinned memory and compare MD5 with ground truth."""
        config = self.config
        bytes_per_block = config.kv_cache_dim * config.element_size
        total_bytes = bytes_per_block * num_blocks
        for li in range(config.num_layers):
            for ptrs, gt_list, kv_name in [(k_ptrs, gt_k, "K"), (v_ptrs, gt_v, "V")]:
                buf = np.zeros(num_blocks * config.kv_cache_dim, dtype=np.uint16)
                ctypes.memmove(buf.ctypes.data, ptrs[li], total_bytes)
                buf = buf.reshape(num_blocks, config.num_heads, config.block_size, config.head_dim)
                expected = gt_list[li][swap_ids]
                self.assertEqual(
                    compute_md5(buf),
                    compute_md5(expected),
                    f"{label} Layer {li} {kv_name}: MD5 mismatch in CPU memory after D2H",
                )

    def _verify_gpu_against_gt(self, gpu_k, gpu_v, gt_k, gt_v, swap_ids, label):
        """Read GPU tensors and compare with ground truth at swap_ids."""
        for li in range(self.config.num_layers):
            for tensors, gt_list, kv_name in [(gpu_k, gt_k, "K"), (gpu_v, gt_v, "V")]:
                actual = tensors[li].cpu().numpy()[swap_ids]
                expected = gt_list[li][swap_ids]
                self.assertEqual(
                    compute_md5(actual),
                    compute_md5(expected),
                    f"{label} Layer {li} {kv_name}: MD5 mismatch on GPU after H2D",
                )
                self.assertTrue(
                    np.allclose(actual, expected, rtol=1e-2, atol=1e-2),
                    f"{label} Layer {li} {kv_name}: data mismatch on GPU after H2D",
                )

    def _verify_non_swap_unchanged(self, gpu_k, gpu_v, snapshots, label):
        """Verify that non-swapped blocks were not corrupted by swap operations."""
        for (name, li, bid), expected_data in snapshots.items():
            tensors = gpu_k if name == "k" else gpu_v
            actual = tensors[li].cpu().numpy()[bid]
            self.assertTrue(
                np.array_equal(actual, expected_data),
                f"{label} {name.upper()} layer {li} block {bid}: non-swapped block corrupted!",
            )

    def _run_multi_round(self, op_func, op_name):
        """
        Core multi-round test logic:
        Each round picks a different random subset of blocks, does D2H then H2D,
        and verifies: CPU correctness after D2H, GPU correctness after H2D,
        and non-swapped blocks are not corrupted.
        """
        rng = random.Random(self.seed)
        config = self.config
        bytes_per_block = config.kv_cache_dim * config.element_size

        gpu_k, gpu_v, gt_k, gt_v = self._init_all_gpu_blocks()

        for round_idx in range(self.num_rounds):
            num_swap = rng.randint(self.min_blocks, self.max_blocks)
            swap_ids = sorted(rng.sample(range(config.total_block_num), num_swap))
            cpu_ids = list(range(num_swap))
            label = f"[{op_name} Round {round_idx + 1}/{self.num_rounds}, {num_swap} blocks]"

            print(f"\n{label}")
            print(f"  swap_ids (first 8): {swap_ids[:8]}...")

            # Snapshot non-swapped blocks before swap
            snapshots = self._snapshot_non_swap_blocks(gpu_k, gpu_v, swap_ids, rng)

            # Allocate CPU pinned memory for this round
            k_ptrs, v_ptrs = [], []
            for li in range(config.num_layers):
                k_ptrs.append(cuda_host_alloc(bytes_per_block * num_swap))
                v_ptrs.append(cuda_host_alloc(bytes_per_block * num_swap))

            # === D2H: evict GPU -> CPU ===
            op_func(gpu_k, k_ptrs, num_swap, swap_ids, cpu_ids, self.device_id, mode=0)
            op_func(gpu_v, v_ptrs, num_swap, swap_ids, cpu_ids, self.device_id, mode=0)
            paddle.device.cuda.synchronize()
            self._verify_cpu_against_gt(k_ptrs, v_ptrs, gt_k, gt_v, swap_ids, num_swap, f"{label} D2H")
            print("  D2H CPU verify: PASS")

            # Zero swapped blocks on GPU to ensure H2D must write correct data
            self._zero_gpu_blocks(gpu_k, gpu_v, swap_ids)

            # === H2D: load CPU -> GPU ===
            op_func(gpu_k, k_ptrs, num_swap, swap_ids, cpu_ids, self.device_id, mode=1)
            op_func(gpu_v, v_ptrs, num_swap, swap_ids, cpu_ids, self.device_id, mode=1)
            paddle.device.cuda.synchronize()
            self._verify_gpu_against_gt(gpu_k, gpu_v, gt_k, gt_v, swap_ids, f"{label} H2D")
            print("  H2D GPU verify: PASS")

            # Verify non-swapped blocks were not corrupted
            self._verify_non_swap_unchanged(gpu_k, gpu_v, snapshots, label)
            print("  Non-swap corruption check: PASS")

        print(f"\nAll {self.num_rounds} rounds passed ({op_name}).")

    def test_random_indices_multi_round_non_batch(self):
        """Multi-round swap with varying random block indices using non-batch operator."""
        self._run_multi_round(swap_cache_all_layers, "non-batch")


if __name__ == "__main__":
    paddle.device.set_device("cuda:0")
    unittest.main()
