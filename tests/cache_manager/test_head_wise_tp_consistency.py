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
"""T53 PR1 head-wise SWA tensor-parallel consistency tests (P13 fix, commit 5).

Case #10 from the feature spec: when the model runs under tensor
parallelism, the head-wise free list MUST shard predictably across ranks.
The fix in commit 5 computes per-rank ``kv_num_heads`` as

    kv_num_heads = max(1, kv_num_heads_global // tp_size)
                   if kv_num_heads_global >= tp_size else 1

inside ``PrefixCacheManager.__init__``. The free list size is then
``num_gpu_blocks * kv_num_heads`` per rank, and the heap is a deterministic
descending range so two ranks built with the same parameters emit the same
allocation order.

We mirror that formula in a small helper and then build managers via
``object.__new__`` (same rationale as ``test_head_wise_freelist.py``).
The constructor itself cannot run on a CPU-only workstation because it
requires a fully-wired ``FDConfig`` plus running IPC signals.
"""

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


def _kv_heads_per_rank(kv_num_heads_global, tp_size):
    """Mirror commit 5 P13 fix from PrefixCacheManager.__init__ exactly."""
    if kv_num_heads_global >= tp_size:
        return max(1, kv_num_heads_global // tp_size)
    return 1


def _build_for_rank(kv_num_heads_global, tp_size, num_gpu_blocks=8):
    """Bare PrefixCacheManager with the per-rank head count baked in."""
    mgr = object.__new__(PrefixCacheManager)
    mgr.cache_config = SimpleNamespace(enable_prefix_caching=False)
    mgr.num_gpu_blocks = num_gpu_blocks
    mgr.kv_num_heads = _kv_heads_per_rank(kv_num_heads_global, tp_size)
    mgr.head_wise = True
    mgr.total_head_wise_cache_ids = 0
    mgr.gpu_free_block_list = list(range(num_gpu_blocks - 1, -1, -1))
    mgr.gpu_free_head_wise_block_list = []
    mgr._init_head_wise_free_list()
    return mgr


def test_tp_size_1_uses_full_kv_heads():
    """#10a — single-rank manager carries the full kv_num_heads_global heads."""
    mgr = _build_for_rank(kv_num_heads_global=4, tp_size=1, num_gpu_blocks=8)
    assert mgr.kv_num_heads == 4
    assert mgr.total_head_wise_cache_ids == 8 * 4
    assert len(mgr.gpu_free_head_wise_block_list) == 32


def test_tp_size_2_splits_kv_heads_evenly():
    """#10b — two ranks each carry kv_num_heads/2; sum across ranks equals the global total."""
    rank0 = _build_for_rank(kv_num_heads_global=4, tp_size=2, num_gpu_blocks=8)
    rank1 = _build_for_rank(kv_num_heads_global=4, tp_size=2, num_gpu_blocks=8)
    assert rank0.kv_num_heads == 2
    assert rank1.kv_num_heads == 2
    total_ids = len(rank0.gpu_free_head_wise_block_list) + len(rank1.gpu_free_head_wise_block_list)
    assert total_ids == 8 * 4, f"sum across ranks must equal num_gpu_blocks * kv_num_heads_global; got {total_ids}"


def test_tp_uneven_split_truncates_via_floor_div():
    """#10c — non-divisible split uses integer floor (4 heads / 3 ranks → 1 head per rank).

    The source code does NOT raise on uneven splits; it deterministically
    truncates via ``//``. That means one head's worth of capacity is
    "lost" per rank in this configuration — but the loss is predictable
    and identical across ranks, which is the property we assert here.
    """
    rank = _build_for_rank(kv_num_heads_global=4, tp_size=3, num_gpu_blocks=8)
    assert rank.kv_num_heads == 1, "4 // 3 == 1; commit 5 P13 fix is a deterministic floor"
    assert len(rank.gpu_free_head_wise_block_list) == 8

    # Edge case: more ranks than heads → clamp to 1 head per rank (else branch).
    over = _build_for_rank(kv_num_heads_global=2, tp_size=4, num_gpu_blocks=8)
    assert over.kv_num_heads == 1
    assert len(over.gpu_free_head_wise_block_list) == 8


def test_tp_alloc_order_deterministic_across_ranks():
    """#10d — same construction params on two ranks produce identical allocation order."""
    rank0 = _build_for_rank(kv_num_heads_global=4, tp_size=2, num_gpu_blocks=8)
    rank1 = _build_for_rank(kv_num_heads_global=4, tp_size=2, num_gpu_blocks=8)

    a0 = rank0.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="rank0")
    a1 = rank1.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="rank1")
    assert a0 == a1, "same heap construction must yield identical pop sequence per head"

    # And the second alloc (after the first drained the smallest ids) is still
    # deterministic across ranks.
    b0 = rank0.allocate_gpu_blocks_head_wise(num_blocks=2, req_id="rank0-b")
    b1 = rank1.allocate_gpu_blocks_head_wise(num_blocks=2, req_id="rank1-b")
    assert b0 == b1
