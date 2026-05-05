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
Per-head independent heap invariants for ``PrefixCacheManager`` (T53 PR2).

These tests pin the contract introduced by RFC-PR2-reanchored §3.1-§4.1:
``kv_num_heads`` independent min-heaps, each over the per-head value space
``[0, num_gpu_blocks)``. The kernel
(``custom_ops/.../multiquery_attention_c16_impl.cuh``) consumes block ids
in ``{-1} \u222a [0, num_gpu_blocks)``; this contract eliminates the prior
shared-heap aliasing class entirely (no modulo HOTFIX needed).

Construction approach mirrors ``test_head_wise_freelist.py``:
``object.__new__(PrefixCacheManager)`` + manual attribute wiring, because
the real ``__init__`` requires a fully-wired ``FDConfig`` and running IPC
signals which cannot be brought up on a CPU-only workstation. No
``MagicMock`` per repo policy.
"""

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
    mgr.gpu_free_block_list = list(range(num_gpu_blocks - 1, -1, -1))
    mgr.gpu_free_head_wise_block_list = []
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


def test_allocated_ids_within_per_head_value_space():
    """#1 — every allocated id lies in the per-head value space [0, num_gpu_blocks)."""
    num_gpu_blocks = 8
    kv_num_heads = 4
    mgr = _build_manager(num_gpu_blocks=num_gpu_blocks, kv_num_heads=kv_num_heads)

    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-vs")

    assert len(allocated) == kv_num_heads
    for row in allocated:
        assert len(row) == 3
        for cid in row:
            assert 0 <= cid < num_gpu_blocks, f"id {cid} outside per-head value space [0, {num_gpu_blocks})"


def test_allocated_ids_unique_within_head():
    """#2 — within a single head's allocation, no id collides with another."""
    mgr = _build_manager(num_gpu_blocks=8, kv_num_heads=4)
    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=5, req_id="req-uw")

    for h, row in enumerate(allocated):
        assert len(row) == len(set(row)), f"head {h} produced duplicate block ids: {row}"


def test_allocated_ids_may_collide_across_heads():
    """#3 — distinct heads share the same value space and may legitimately
    return identical ids; this is the whole point of the per-head heap design.

    With num_gpu_blocks=4 and kv_num_heads=4 each pulling 4 blocks, every head
    drains its own heap and ends up holding ``[0, 1, 2, 3]``. Cross-head id
    collision is therefore not just allowed — it is mandatory.
    """
    num_gpu_blocks = 4
    kv_num_heads = 4
    mgr = _build_manager(num_gpu_blocks=num_gpu_blocks, kv_num_heads=kv_num_heads)

    allocated = mgr.allocate_gpu_blocks_head_wise(num_blocks=num_gpu_blocks, req_id="req-cross")

    # Sets across heads: every head should hold the full per-head value space.
    expected = set(range(num_gpu_blocks))
    for h, row in enumerate(allocated):
        assert set(row) == expected, f"head {h} did not drain its full per-head heap: {row}"

    # And therefore at least one id is shared between head 0 and head 1.
    assert set(allocated[0]) & set(
        allocated[1]
    ), "expected non-empty intersection across heads under per-head heap design"


def test_free_returns_to_correct_head_heap():
    """#4 — recycled ids return to the originating head's heap, not somewhere else.

    Allocate, recycle a known id from head 0, then allocate again with that head's
    heap fully drained except for the freed id; assert head 0 hands that id back.
    """
    num_gpu_blocks = 4
    kv_num_heads = 2
    mgr = _build_manager(num_gpu_blocks=num_gpu_blocks, kv_num_heads=kv_num_heads)

    # Drain every head completely: each head holds [0,1,2,3].
    drained = mgr.allocate_gpu_blocks_head_wise(num_blocks=num_gpu_blocks, req_id="drain")
    for h in range(kv_num_heads):
        assert len(mgr.gpu_free_head_wise_block_lists[h]) == 0

    # Recycle exactly one id from head 0 only.
    head_0_returned = drained[0][0]
    nested_recycle = [[] for _ in range(kv_num_heads)]
    nested_recycle[0] = [head_0_returned]
    mgr.recycle_gpu_blocks_head_wise(nested_recycle)

    assert mgr.gpu_free_head_wise_block_lists[0] == [head_0_returned]
    for h in range(1, kv_num_heads):
        assert (
            mgr.gpu_free_head_wise_block_lists[h] == []
        ), f"head {h} heap should remain empty (nothing recycled to it)"

    # Next allocation of 1 block per head must FAIL fast for the still-empty
    # heads (head 1..N-1) — proves the freed id lives in head 0's heap and
    # cannot be cross-pollinated.
    with pytest.raises(RuntimeError):
        mgr.allocate_gpu_blocks_head_wise(num_blocks=1, req_id="post-recycle")
