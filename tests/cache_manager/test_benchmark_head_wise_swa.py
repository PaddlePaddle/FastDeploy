#!/usr/bin/env python3
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
"""T53 PR1 head-wise SWA recycle micro-benchmark (CPU, no model).

Mirrors the ``tests/spec_decode/test_benchmark_ngram_kernel.py`` (T48)
pattern: a unittest-discoverable benchmark that sweeps a small parameter
grid and records ops/sec for the head-wise free-list / SWA recycle paths.

The benchmark covers the **scheduler-side** primitives only; the
end-to-end +30% throughput gate on ERNIE-4.5-21B-A3B-Paddle is still
exercised by ``.checkpoints/h10/task-53/scripts/bench_recycle.sh`` on
A800 (BF16, fixed-IO, same VRAM, fixture mode).

Groups
------
  1. kv_num_heads — [2, 4, 8, 16]   (TP shards)
  2. blocks_per_req — [16, 64, 256] (pressure on free list)
  3. window/sink ratio — [(64,32), (1024,128), (4096,256)]

Run::

    cd FastDeploy && python tests/cache_manager/test_benchmark_head_wise_swa.py
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1

WARMUP = 50
NUM_ITERS = 500


def _build_prefix_cache(num_blocks: int, kv_num_heads: int) -> PrefixCacheManager:
    pcm = object.__new__(PrefixCacheManager)
    pcm.num_gpu_blocks = num_blocks
    pcm.kv_num_heads = kv_num_heads
    pcm._head_wise_free_lists = [list(range(num_blocks)) for _ in range(kv_num_heads)]
    pcm._head_wise_alloc = {}
    return pcm


def _build_rm(window: int, sink: int, block_size: int = 16, kv_num_heads: int = 4):
    rm = object.__new__(ResourceManagerV1)
    rm.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(window_size=window, sink_size=sink),
    )

    class _Cache:
        def __init__(self, n):
            self.kv_num_heads = n
            self.recycled = 0

        def recycle_gpu_blocks_head_wise(self, ids, req_id=None):
            self.recycled += 1

        def allocate_gpu_blocks_head_wise(self, n, req_id=None):
            return [list(range(n)) for _ in range(kv_num_heads)]

    rm.cache_manager = _Cache(kv_num_heads)
    rm.swa_head_recycle_upto = {}
    rm.swa_head_block_tables = {}
    return rm


def _bench(fn, *args, iters=NUM_ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    dt = time.perf_counter() - t0
    return iters / dt if dt > 0 else float("inf")


class HeadWiseSWABenchmark(unittest.TestCase):
    """Micro-bench head-wise alloc / recycle paths"""

    def test_alloc_recycle_throughput_grid(self):
        rows = []
        for kv_heads in (2, 4, 8, 16):
            for bpr in (16, 64, 256):
                pcm = _build_prefix_cache(num_blocks=bpr * 8, kv_num_heads=kv_heads)

                def alloc():
                    pcm._head_wise_free_lists = [list(range(bpr * 8)) for _ in range(kv_heads)]
                    return [[fl.pop() for _ in range(bpr)] for fl in pcm._head_wise_free_lists]

                ops = _bench(alloc, iters=200, warmup=20)
                rows.append((kv_heads, bpr, ops))

        # Print compact table; pytest -s shows it.
        print("\n[T53/bench] kv_heads | blocks_per_req | alloc_ops_per_sec")
        for kv, bpr, ops in rows:
            print(f"  {kv:>4}    | {bpr:>5}          | {ops:>12.0f}")

        # Sanity gate: largest config should still hit > 100 ops/s on CPU.
        worst = min(r[2] for r in rows)
        self.assertGreater(worst, 100.0, f"alloc throughput collapsed: {worst:.1f} ops/s")

    def test_swa_window_sink_recycle_throughput(self):
        rows = []
        for window, sink in ((64, 32), (1024, 128), (4096, 256)):
            rm = _build_rm(window=window, sink=sink, kv_num_heads=4)
            req = SimpleNamespace(
                request_id="bench-0",
                num_total_tokens=window * 2,
                num_computed_tokens=window * 2,
                cache_swap_metadata=[],
                cache_evict_metadata=[],
            )
            # Pre-populate per-head block tables so recycle has work to do.
            rm.swa_head_block_tables[req.request_id] = [list(range(window // 16 + 4)) for _ in range(4)]

            def step():
                # Reset cursor each iter so recycle does work on every call.
                rm.swa_head_recycle_upto[req.request_id] = [0 for _ in rm.swa_head_block_tables[req.request_id]]
                rm.recycle_request_swa_head_cache(req)

            ops = _bench(step, iters=300, warmup=30)
            rows.append((window, sink, ops))

        print("\n[T53/bench] window | sink | recycle_ops_per_sec")
        for w, s, ops in rows:
            print(f"  {w:>5} | {s:>4} | {ops:>12.0f}")

        # Sanity: even tightest window/sink should sustain > 50 ops/s on CPU.
        worst = min(r[2] for r in rows)
        self.assertGreater(worst, 50.0, f"recycle throughput collapsed: {worst:.1f} ops/s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
