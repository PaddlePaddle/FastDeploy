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
"""T53 PR1 head-wise SWA extend-validation tests for ``PrefixCacheManager``.

Case #9 from the architecture brief: extending a request's head-wise
allocation at decode time must satisfy four invariants

  * a zero-block extend is a no-op (returns ``[[]] * kv_num_heads``,
    free heap unchanged),
  * extending past head-wise capacity raises (``assert needed <= len(...)``
    in ``allocate_gpu_blocks_head_wise`` makes this an ``AssertionError``),
  * successive extends to the same request yield disjoint ids per head
    (the allocator drains via ``heappop`` from a single shared heap so
    ids cannot be reissued before recycle),
  * after a partial recycle, the next extend reuses recycled ids first
    (heap is a min-heap; recycled ids are pushed back via ``heappush``).

Same ``object.__new__`` construction pattern as ``test_head_wise_freelist.py``.
"""

import heapq
from types import SimpleNamespace

import pytest

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager


class _DummyMetric:
    def set(self, *_a, **_k):
        pass

    def inc(self, *_a, **_k):
        pass

    def dec(self, *_a, **_k):
        pass


class _DummyMainMetrics:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _DummyMetric()


@pytest.fixture(autouse=True)
def _patch_metrics(monkeypatch):
    monkeypatch.setattr(
        "fastdeploy.cache_manager.prefix_cache_manager.main_process_metrics",
        _DummyMainMetrics(),
    )


def _build_manager(num_gpu_blocks=8, kv_num_heads=4):
    mgr = object.__new__(PrefixCacheManager)
    mgr.cache_config = SimpleNamespace(enable_prefix_caching=False)
    mgr.num_gpu_blocks = num_gpu_blocks
    mgr.kv_num_heads = kv_num_heads
    mgr.head_wise = True
    mgr.total_head_wise_cache_ids = 0
    mgr.gpu_free_block_list = []
    mgr.gpu_free_head_wise_block_list = []
    mgr._init_head_wise_free_list()
    return mgr


def test_extend_with_zero_blocks_is_noop():
    """#9a — alloc(0) returns empty per-head rows, free heap unchanged."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)
    initial_free = len(mgr.gpu_free_head_wise_block_list)

    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=0, req_id="req-zero")

    assert len(allocated) == 4
    for row in allocated:
        assert row == []
    assert len(mgr.gpu_free_head_wise_block_list) == initial_free


def test_extend_more_than_available_raises():
    """#9b — requesting more blocks than head-wise capacity raises ``AssertionError``."""
    mgr = _build_manager(num_gpu_blocks=4, kv_num_heads=4)
    # Capacity = 4 blocks per head. Request 5 → needed=20 > free=16.
    with pytest.raises(AssertionError):
        mgr.allocate_gpu_blocks_head_wise(num_blocks=5, req_id="req-overflow")


def test_extend_preserves_per_head_disjointness():
    """#9c — successive extends to the same req yield non-overlapping ids per head."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)

    first = mgr.allocate_gpu_blocks_head_wise(num_blocks=2, req_id="req-extend")
    second = mgr.allocate_gpu_blocks_head_wise(num_blocks=2, req_id="req-extend")

    # Across the two calls, every id ever issued (irrespective of head) must
    # be unique — the allocator pops from a single shared heap.
    flat = [cid for row in first for cid in row] + [cid for row in second for cid in row]
    assert len(flat) == 16
    assert len(set(flat)) == 16, "no id may be issued twice without a recycle in between"


def test_extend_after_partial_recycle_uses_recycled_ids():
    """#9d — recycled ids re-enter the heap and are returned by the next alloc (min-heap)."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)

    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-cycle")
    flat_first = sorted(cid for row in allocated for cid in row)

    # Recycle the lowest 4 ids only.
    to_recycle = flat_first[:4]
    mgr.recycle_gpu_blocks_head_wise(to_recycle, req_id="req-cycle")

    # Snapshot the heap; the 4 smallest values must be exactly the recycled ids.
    snapshot = list(mgr.gpu_free_head_wise_block_list)
    smallest_4 = []
    for _ in range(4):
        smallest_4.append(heapq.heappop(snapshot))
    assert sorted(smallest_4) == sorted(to_recycle), "recycled ids must be the next to pop"

    # Real next alloc should issue exactly those recycled ids first.
    again = mgr.allocate_gpu_blocks_head_wise(num_blocks=1, req_id="req-cycle-2")
    flat_again = sorted(cid for row in again for cid in row)
    assert flat_again[:4] == sorted(to_recycle)
