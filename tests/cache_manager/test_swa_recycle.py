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
"""T53 PR1 head-wise SWA recycle tests for ``ResourceManagerV1``.

These tests cover the three §4 cases from the architecture brief:

* #5 — ``test_swa_recycle_respects_sink_and_window``: sink/window math
  releases only fully-aged blocks and ``swa_head_recycle_upto`` is monotone.
* #6 — ``test_swa_recycle_skips_when_swap_inflight``: a request whose
  per-request ``cache_swap_metadata`` queue still has unfinished swaps
  targeting one of its own blocks is left untouched (recycle is a no-op).
* #7 — ``test_mutual_exclusion_with_prefix_caching``: ``PrefixCacheManager``
  refuses to construct when both ``enable_prefix_caching`` and
  ``FD_HEAD_WISE_KV_CACHE`` are on (assertion landed in commit 2).

Approach: ``ResourceManagerV1`` is built via ``object.__new__`` because its
real ``__init__`` requires a fully-wired ``FDConfig``, IPC signals, and a
running ``CacheManager`` that the workstation cannot bring up. This mirrors
the pattern used in ``test_head_wise_freelist.py`` (commit 2) and the H10
task-20 ``common_engine`` tests (no MagicMock, real objects only).
"""

from types import SimpleNamespace

import pytest

from fastdeploy.cache_manager.v1.metadata import CacheSwapMetadata
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


class _FakeCacheManager:
    """Minimal cache manager exposing the head-wise APIs the SWA recycle calls."""

    def __init__(self, kv_num_heads=2):
        self.kv_num_heads = kv_num_heads
        self.recycled = []  # list of (req_id, ids) recorded per call

    def recycle_gpu_blocks_head_wise(self, cache_ids, req_id=None):
        self.recycled.append((req_id, list(cache_ids)))

    def allocate_gpu_blocks_head_wise(self, num_blocks, req_id=None):
        return [list(range(num_blocks)) for _ in range(self.kv_num_heads)]


def _build_manager(window=64, sink=32, block_size=16, kv_num_heads=2, head_wise_swa_ratio=1.0):
    """Build a bare ``ResourceManagerV1`` with just the SWA recycle state wired."""
    rm = object.__new__(ResourceManagerV1)
    rm.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            window_size=window,
            sink_size=sink,
            num_key_value_heads=kv_num_heads,
            head_wise_swa_ratio=head_wise_swa_ratio,
        ),
    )
    rm.cache_manager = _FakeCacheManager(kv_num_heads=kv_num_heads)
    rm.swa_head_recycle_upto = {}
    rm.swa_head_block_tables = {}
    rm.swa_legacy_recycle_upto = {}
    rm.swa_legacy_recycled_blocks = {}
    return rm


def _fake_request(req_id="req-0", num_total_tokens=512, swap_meta=None, evict_meta=None):
    return SimpleNamespace(
        request_id=req_id,
        num_total_tokens=num_total_tokens,
        num_computed_tokens=num_total_tokens,
        cache_swap_metadata=list(swap_meta or []),
        cache_evict_metadata=list(evict_meta or []),
    )


@pytest.mark.parametrize(
    ("kv_num_heads", "head_wise_swa_ratio", "expected"),
    [
        (4, 1.0, 4),
        (4, 0.5, 2),
        (4, 0.0, 0),
        (1, 0.5, 1),
        (1, 1.0, 1),
        (1, 0.0, 0),
        (8, 0.25, 2),
        (3, 0.5, 2),
        (2, 0.5, 1),
    ],
)
def test_num_swa_heads_clamps_positive_ratios(kv_num_heads, head_wise_swa_ratio, expected):
    rm = _build_manager(kv_num_heads=kv_num_heads, head_wise_swa_ratio=head_wise_swa_ratio)

    assert rm._num_swa_heads() == expected


# ---------------------------------------------------------------------------
# Case #5 — sink/window math
# ---------------------------------------------------------------------------
def test_swa_recycle_respects_sink_and_window(monkeypatch):
    """Only blocks in ``[ceil(sink/bs), floor((T-window)/bs))`` are released; cursor is monotone."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    rm = _build_manager(window=64, sink=32, block_size=16, kv_num_heads=2)
    # 32 blocks per head, total tokens = 32 * 16 = 512.
    rm.swa_head_block_tables["req-0"] = [list(range(100, 132)), list(range(200, 232))]
    req = _fake_request(req_id="req-0", num_total_tokens=512)

    released = rm.recycle_request_swa_head_cache(req)
    # window_blocks = ceil(64/16) = 4; sink_blocks = ceil(32/16) = 2.
    # recycle_upto = (512 - 4*16) // 16 = 28; floor = 2; per-head release = 26 blocks.
    assert released == 26 * 2, f"expected 52 blocks released, got {released}"
    # Sink (idx 0,1) and tail window (idx 28..31) must remain untouched.
    cursor = rm.swa_head_recycle_upto["req-0"]
    assert cursor == [28, 28], f"per-head recycle_upto must equal 28, got {cursor}"
    # Verify the recycled IDs match the open interval [2, 28) on each head.
    head0_ids = list(range(100 + 2, 100 + 28))
    head1_ids = list(range(200 + 2, 200 + 28))
    recorded = [ids for (_, ids) in rm.cache_manager.recycled]
    assert head0_ids in recorded and head1_ids in recorded

    # Second call with the same total_tokens must be a no-op (monotone cursor).
    rm.cache_manager.recycled.clear()
    released_again = rm.recycle_request_swa_head_cache(req)
    assert released_again == 0
    assert rm.swa_head_recycle_upto["req-0"] == [28, 28]


def test_swa_recycle_only_recycles_swa_heads(monkeypatch):
    """Only the first ``round(kv_heads * ratio)`` rows are recycled; full-attention rows stay intact."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    rm = _build_manager(window=64, sink=32, block_size=16, kv_num_heads=4, head_wise_swa_ratio=0.5)
    rm.swa_head_block_tables["req-swa-only"] = [
        list(range(100, 132)),
        list(range(200, 232)),
        list(range(300, 332)),
        list(range(400, 432)),
    ]
    req = _fake_request(req_id="req-swa-only", num_total_tokens=512)

    released = rm.recycle_request_swa_head_cache(req)

    assert released == 26 * 2
    assert rm.swa_head_recycle_upto["req-swa-only"] == [28, 28, 2, 2]
    recorded = [ids for (_, ids) in rm.cache_manager.recycled]
    assert list(range(100 + 2, 100 + 28)) in recorded
    assert list(range(200 + 2, 200 + 28)) in recorded
    assert all(not set(ids).intersection(range(300, 432)) for ids in recorded)


def test_swa_recycle_fires_only_on_block_boundary(monkeypatch):
    """Decode-step recycle is throttled to block boundaries to avoid per-token O(H*B) scans."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    rm = _build_manager(window=64, sink=32, block_size=16, kv_num_heads=2)
    rm.swa_head_block_tables["req-boundary"] = [list(range(100, 132)), list(range(200, 232))]
    req = _fake_request(req_id="req-boundary", num_total_tokens=511)

    released = rm.recycle_request_swa_head_cache(req)

    assert released == 0
    assert "req-boundary" not in rm.swa_head_recycle_upto


# ---------------------------------------------------------------------------
# Case #6 — overlap with in-flight swap
# ---------------------------------------------------------------------------
def test_swa_recycle_skips_when_swap_inflight(monkeypatch):
    """An unfinished ``CacheSwapMetadata`` touching the request's blocks blocks the recycle."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    rm = _build_manager(window=64, sink=32, block_size=16, kv_num_heads=2)
    rm.swa_head_block_tables["req-1"] = [list(range(100, 132)), list(range(200, 232))]
    # Pending swap touching block 105 (which is in the recycle range for head 0).
    pending = CacheSwapMetadata(src_block_ids=[105], dst_block_ids=[999], success=False)
    req = _fake_request(req_id="req-1", num_total_tokens=512, swap_meta=[pending])

    released = rm.recycle_request_swa_head_cache(req)
    assert released == 0, "recycle must skip when an in-flight swap targets owned blocks"
    assert "req-1" not in rm.swa_head_recycle_upto, "cursor must not advance on skip"
    assert rm.cache_manager.recycled == []


# ---------------------------------------------------------------------------
# Case #7 — mutual exclusion vs prefix caching
# ---------------------------------------------------------------------------
def test_mutual_exclusion_with_prefix_caching(monkeypatch):
    """``PrefixCacheManager`` must refuse when both head-wise and prefix caching are on."""
    monkeypatch.setattr("fastdeploy.cache_manager.prefix_cache_manager.envs.FD_HEAD_WISE_KV_CACHE", 1)
    monkeypatch.setattr("fastdeploy.cache_manager.prefix_cache_manager.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1)
    from fastdeploy.cache_manager import prefix_cache_manager as pcm_module

    cache_config = SimpleNamespace(
        enable_prefix_caching=True,
        total_block_num=4,
        prefill_kvcache_block_num=4,
        num_cpu_blocks=0,
        model_cfg=SimpleNamespace(num_key_value_heads=2),
    )
    fake_fd_config = SimpleNamespace(
        cache_config=cache_config,
        speculative_config=SimpleNamespace(),
    )
    with pytest.raises((AssertionError, ValueError)):
        pcm_module.PrefixCacheManager(
            config=fake_fd_config,
            tensor_parallel_size=1,
            splitwise_role="mixed",
            local_data_parallel_id=0,
        )
