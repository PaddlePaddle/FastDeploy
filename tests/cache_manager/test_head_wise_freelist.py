# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
Head-wise KV cache free-list tests for ``PrefixCacheManager``.

Approach: instances are built via ``object.__new__(PrefixCacheManager)`` plus
manual attribute setup. Real ``__init__`` requires a fully-wired ``FDConfig``
plus running IPC signals which cannot be brought up on a CPU-only workstation
without GPU paddle. The ``object.__new__`` pattern is the same one used by
H10 task-20 ``common_engine`` tests for the identical reason.
"""

import heapq
import logging
from types import SimpleNamespace

import pytest

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager


class _DummyMetric:
    def __init__(self):
        self.values = []

    def set(self, value):
        self.values.append(value)

    def inc(self, value=1):
        self.values.append(("inc", value))

    def dec(self, value=1):
        self.values.append(("dec", value))


class _DummyMainMetrics:
    def __init__(self):
        self._metrics = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._metrics:
            self._metrics[name] = _DummyMetric()
        return self._metrics[name]


def _build_manager(num_gpu_blocks=8, kv_num_heads=4, head_wise=True):
    """Construct a bare ``PrefixCacheManager`` and run the head-wise initializer."""
    mgr = object.__new__(PrefixCacheManager)
    mgr.cache_config = SimpleNamespace(enable_prefix_caching=False)
    mgr.num_gpu_blocks = num_gpu_blocks
    mgr.num_cpu_blocks = 0
    mgr.kv_num_heads = kv_num_heads
    mgr.head_wise = head_wise
    mgr.total_head_wise_cache_ids = 0
    mgr.gpu_free_block_list = []
    if head_wise:
        mgr._init_head_wise_free_list()
    return mgr


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    """Replace the module-level metrics singleton with a recording dummy."""
    dummy = _DummyMainMetrics()
    monkeypatch.setattr(
        "fastdeploy.cache_manager.prefix_cache_manager.main_process_metrics",
        dummy,
    )
    return dummy


def test_head_wise_free_list_size():
    """#1 — initializer fills heap with num_gpu_blocks * kv_num_heads ids; smallest pops first."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)
    assert mgr.total_head_wise_cache_ids == 32
    assert len(mgr.gpu_free_block_list) == 8 * 4
    # heapq is a min-heap → smallest id pops first.
    assert heapq.heappop(mgr.gpu_free_block_list) == 0


def test_head_wise_allocate_returns_2d():
    """#2 — alloc returns [kv_num_heads][N], ids in valid range, no duplicates across heads."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)
    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-2d")

    assert len(allocated) == 4  # one row per kv head
    for row in allocated:
        assert len(row) == 3

    flat = [cid for row in allocated for cid in row]
    assert len(flat) == 12
    assert len(set(flat)) == 12  # no duplicates anywhere
    for cid in flat:
        assert 0 <= cid < mgr.total_head_wise_cache_ids


def test_head_wise_recycle_round_trip():
    """#3 — alloc → recycle returns the heap to its initial size; subsequent alloc succeeds."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)
    initial_free = len(mgr.gpu_free_block_list)

    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-rt")
    assert len(mgr.gpu_free_block_list) == initial_free - 12

    mgr.recycle_gpu_blocks_head_wise(allocated, req_id="req-rt")
    assert len(mgr.gpu_free_block_list) == initial_free

    # Heap invariant preserved.
    again = mgr.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-rt-2")
    assert sum(len(row) for row in again) == 12


def test_head_wise_recycle_dedup_and_range_check(caplog):
    """#4 — duplicates and out-of-range ids are dropped (warned), only valid ids re-enter the heap."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)

    # Drain a few ids so we can recycle a known-valid one back.
    drained = mgr.allocate_gpu_blocks_head_wise(num_blocks=1, req_id="req-drain")
    valid_id = drained[0][0]  # an id we now own
    duplicate = valid_id  # used twice in the recycle list
    out_of_range = mgr.total_head_wise_cache_ids + 17  # beyond the valid window

    free_before_recycle = len(mgr.gpu_free_block_list)

    # ``get_logger`` may produce a non-propagating logger; force propagation so
    # caplog can observe the warnings emitted by the recycle path.
    pcm_logger = logging.getLogger("prefix_cache_manager")
    prior_propagate = pcm_logger.propagate
    pcm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            mgr.recycle_gpu_blocks_head_wise(
                [valid_id, duplicate, out_of_range],
                req_id="req-dedup",
            )
    finally:
        pcm_logger.propagate = prior_propagate

    # Only the single valid id should have been pushed back.
    assert len(mgr.gpu_free_block_list) == free_before_recycle + 1
    # Warnings should mention either a dropped duplicate or an out-of-range id.
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert ("duplicate" in log_text) or ("out-of-range" in log_text)
