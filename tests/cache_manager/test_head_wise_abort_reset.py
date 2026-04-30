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
"""T53 PR1 head-wise SWA abort-reset tests for ``ResourceManagerV1._free_blocks``.

Case #8 from the architecture brief: when a request is aborted mid-flight the
``_free_blocks`` hook (gated by ``FD_HEAD_WISE_KV_CACHE``) MUST

  * release every per-head block id back into the head-wise free heap,
  * clear the per-request cursor in ``swa_head_recycle_upto``,
  * clear the per-request table in ``swa_head_block_tables``,
  * remain idempotent under repeated abort calls (no duplicate heap entries,
    no KeyError, no exception).

Approach mirrors ``test_head_wise_freelist.py`` and ``test_swa_recycle.py``:
both ``PrefixCacheManager`` and ``ResourceManagerV1`` are constructed via
``object.__new__`` because their real ``__init__`` requires a wired
``FDConfig`` plus running IPC signals that cannot be brought up on the
workstation. No MagicMock anywhere — the cache manager is the real
``PrefixCacheManager`` so the heap invariant and dedup logic exercised by
``recycle_gpu_blocks_head_wise`` are the real production code paths.
"""

import heapq
from types import SimpleNamespace

import pytest

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


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


def _build_pcm(num_gpu_blocks=8, kv_num_heads=4):
    """Real PrefixCacheManager with head-wise free list initialized."""
    pcm = object.__new__(PrefixCacheManager)
    pcm.cache_config = SimpleNamespace(enable_prefix_caching=False)
    pcm.num_gpu_blocks = num_gpu_blocks
    pcm.kv_num_heads = kv_num_heads
    pcm.head_wise = True
    pcm.total_head_wise_cache_ids = 0
    pcm.gpu_free_block_list = []
    pcm._init_head_wise_free_list()
    # _free_blocks falls through to enable_cache_manager_v1 branch below; give
    # the PCM a no-op request_finish so the legacy code path does not crash.
    pcm.request_finish = lambda _req: None
    return pcm


def _build_rm(pcm):
    """Bare ResourceManagerV1 wired to ``pcm`` with the legacy V1 path active."""
    rm = object.__new__(ResourceManagerV1)
    rm.cache_manager = pcm
    rm.config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(splitwise_role="mixed"),
        model_config=SimpleNamespace(window_size=64, sink_size=32),
    )
    rm.swa_head_recycle_upto = {}
    rm.swa_head_block_tables = {}
    rm.enable_cache_manager_v1 = True  # forces request_finish branch
    rm.using_extend_tables_req_id = set()
    rm.reuse_block_num_map = {}
    rm.need_block_num_map = {}
    return rm


def _fake_request(req_id="req-A"):
    return SimpleNamespace(
        request_id=req_id,
        block_tables=[],
        extend_block_tables=[],
        num_total_tokens=0,
        num_computed_tokens=0,
        cache_swap_metadata=[],
        cache_evict_metadata=[],
    )


def test_abort_releases_head_wise_blocks_back_to_free_list(monkeypatch):
    """#8a — aborted req's head-wise ids return to the free heap; heap invariant preserved."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    pcm = _build_pcm(num_gpu_blocks=8, kv_num_heads=4)
    rm = _build_rm(pcm)
    initial_free = len(pcm.gpu_free_block_list)

    # Allocate 3 blocks per head and stash on the per-request map.
    allocated = pcm.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-A")
    rm.swa_head_block_tables["req-A"] = allocated
    assert len(pcm.gpu_free_block_list) == initial_free - 12

    rm._free_blocks(_fake_request("req-A"))

    assert len(pcm.gpu_free_block_list) == initial_free, "all 12 ids must return to free heap"
    # Heap invariant: smallest id pops first; sequence must be sorted.
    snapshot = list(pcm.gpu_free_block_list)
    pops = [heapq.heappop(snapshot) for _ in range(len(snapshot))]
    assert pops == sorted(pops), "free list must remain a valid min-heap after abort"


def test_abort_clears_swa_recycle_cursor(monkeypatch):
    """#8b — abort drops the per-request entry in ``swa_head_recycle_upto``."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    pcm = _build_pcm()
    rm = _build_rm(pcm)
    rm.swa_head_recycle_upto["req-B"] = [10, 10, 10, 10]
    # No head_blocks for req-B → no recycle call, but the cursor still must be popped.

    rm._free_blocks(_fake_request("req-B"))

    assert "req-B" not in rm.swa_head_recycle_upto


def test_abort_clears_swa_head_block_tables(monkeypatch):
    """#8c — abort drops the per-request entry in ``swa_head_block_tables``."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    pcm = _build_pcm()
    rm = _build_rm(pcm)
    allocated = pcm.allocate_gpu_blocks_head_wise(num_blocks=2, req_id="req-C")
    rm.swa_head_block_tables["req-C"] = allocated
    rm.swa_head_recycle_upto["req-C"] = [0, 0, 0, 0]

    rm._free_blocks(_fake_request("req-C"))

    assert "req-C" not in rm.swa_head_block_tables
    assert "req-C" not in rm.swa_head_recycle_upto


def test_double_abort_is_idempotent(monkeypatch):
    """#8d — second abort is a no-op; free heap size unchanged, no exception, no duplicates."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    pcm = _build_pcm(num_gpu_blocks=8, kv_num_heads=4)
    rm = _build_rm(pcm)
    initial_free = len(pcm.gpu_free_block_list)

    allocated = pcm.allocate_gpu_blocks_head_wise(num_blocks=3, req_id="req-D")
    rm.swa_head_block_tables["req-D"] = allocated

    rm._free_blocks(_fake_request("req-D"))
    free_after_first = len(pcm.gpu_free_block_list)
    assert free_after_first == initial_free

    # Second abort must not raise and must not push any id again.
    rm._free_blocks(_fake_request("req-D"))
    assert len(pcm.gpu_free_block_list) == free_after_first
    # No duplicate ids in the heap.
    assert len(set(pcm.gpu_free_block_list)) == len(pcm.gpu_free_block_list)
